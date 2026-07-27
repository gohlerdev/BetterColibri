"""Integration test: the REAL engine's mux serve protocol (SERVE_BATCH=1).

Audit finding I7: the byte protocol (SUBMIT/DATA/DONE/ERROR framing over
stdin/stdout) is the least-typed, most platform-sensitive seam in the system
(#139, #195, #401) and shipped on manual testing — test_openai_server.py covers
the gateway against a MOCK engine; nothing executed run_serve_mux itself.

This suite drives the actual `colibri` binary against the glm_tiny fixture:
handshake, generation round-trip, KV prefix reuse, CANCEL, SLOT_BUSY,
DUPLICATE_ID, BAD_FRAME recovery, CONTEXT_EXCEEDED, and EOF shutdown.

Skips (like test_inefficiency) when the engine or fixture is absent: CI's
`make check` builds no fixture by design (#140). The tiny tokenizer
(tests/tok_o200k_tiny.json) is copied into the fixture on setup; its ids can
exceed the tiny model's 256-token vocab, which the engine handles by zeroing
OOB embeddings (#SEC-5) — fine here, the contract under test is the protocol,
not output quality.
"""
import os
import select
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
TINY = C_DIR / "glm_tiny"
TOK_FIXTURE = HERE / "tok_o200k_tiny.json"

READY = b"\x01\x01READY\x01\x01"


def _find_engine() -> Path:
    for name in ("colibri", "colibri.exe", "glm", "glm.exe"):
        cand = C_DIR / name
        if cand.exists():
            return cand
    return C_DIR / "colibri"


ENGINE = _find_engine()


def _present() -> bool:
    return ENGINE.exists() and (TINY / "config.json").exists() and TOK_FIXTURE.exists()


def _skip_reason() -> str:
    if not ENGINE.exists():
        return "engine not built (run: make)"
    if not (TINY / "config.json").exists():
        return "glm_tiny fixture absent (run tools/make_glm_oracle.py)"
    return "tok_o200k_tiny.json fixture missing"


@unittest.skipUnless(_present(), _skip_reason() or "prerequisites present")
class MuxProtocolTest(unittest.TestCase):
    """Each test speaks the wire format from docs/serve_protocol.md directly."""

    @classmethod
    def setUpClass(cls):
        # the serve modes need SNAP/tokenizer.json; keep the fixture self-contained
        tok = TINY / "tokenizer.json"
        if not tok.exists():
            shutil.copy(TOK_FIXTURE, tok)

    def spawn(self, **env_overlay):
        env = os.environ.copy()
        env.update({"SNAP": str(TINY), "SERVE": "1", "SERVE_BATCH": "1",
                    "KV_SLOTS": "2", "NGEN": "32", "CTX": "128",
                    "COLI_NO_OMP_TUNE": "1", "KVSAVE": "0", "PROF": "0"})
        env.update({k: str(v) for k, v in env_overlay.items()})
        proc = subprocess.Popen([str(ENGINE), "8"], env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)
        os.set_blocking(proc.stdout.fileno(), False)   # all reads go through _fill
        proc._rxbuf = bytearray()                      # protocol receive buffer
        self.addCleanup(self._reap, proc)
        self._read_until(proc, READY, timeout=30)
        return proc

    @staticmethod
    def _reap(proc):
        if proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    @staticmethod
    def _fill(proc, deadline):
        """Pull whatever is available into proc._rxbuf (select-bounded)."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        r, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.5))
        if r:
            chunk = proc.stdout.read(1 << 20)
            if chunk == b"" and proc.poll() is not None:
                raise AssertionError("engine exited unexpectedly")
            if chunk:
                proc._rxbuf += chunk
                return True
        return False

    def _read_line(self, proc, timeout=15.0):
        """One protocol line from the buffered stream, deadline-bounded."""
        deadline = time.monotonic() + timeout
        while True:
            nl = proc._rxbuf.find(b"\n")
            if nl >= 0:
                line = bytes(proc._rxbuf[:nl])
                del proc._rxbuf[:nl + 1]
                return line
            if time.monotonic() >= deadline:
                raise AssertionError(f"timeout waiting for a line (buf={bytes(proc._rxbuf[:120])!r})")
            self._fill(proc, deadline)

    def _read_exact(self, proc, n, timeout=15.0):
        deadline = time.monotonic() + timeout
        while len(proc._rxbuf) < n:
            if time.monotonic() >= deadline:
                raise AssertionError(f"timeout reading {n}-byte payload")
            self._fill(proc, deadline)
        data = bytes(proc._rxbuf[:n])
        del proc._rxbuf[:n]
        return data

    def _read_until(self, proc, token, timeout=15.0):
        deadline = time.monotonic() + timeout
        seen = []
        while time.monotonic() < deadline:
            line = self._read_line(proc, timeout=max(0.1, deadline - time.monotonic()))
            seen.append(line)
            if token in line:
                return seen
        raise AssertionError(f"never saw {token!r}; lines: {seen[-5:]}")

    def submit(self, proc, rid, slot, prompt, max_tokens=8, temp=0.0, top_p=1.0):
        payload = prompt.encode()
        frame = (f"SUBMIT {rid} {slot} {len(payload)} {max_tokens} "
                 f"{temp:.8g} {top_p:.8g}\n").encode() + payload + b"\n"
        proc.stdin.write(frame)
        proc.stdin.flush()

    def collect_turn(self, proc, rid, timeout=30.0):
        """Read frames until this rid's DONE or ERROR; return (kind, text, stats_or_code)."""
        deadline = time.monotonic() + timeout
        text = bytearray()
        while time.monotonic() < deadline:
            line = self._read_line(proc, timeout=max(0.1, deadline - time.monotonic()))
            fields = line.split()
            if not fields:
                continue
            if fields[0] == b"DATA" and fields[1] == str(rid).encode():
                n = int(fields[2])
                chunk = self._read_exact(proc, n)
                self.assertEqual(self._read_exact(proc, 1), b"\n", "DATA terminator")
                text += chunk
            elif fields[0] == b"DONE" and fields[1] == str(rid).encode():
                self.assertEqual(fields[2], b"STAT", "DONE carries STAT")
                self.assertGreaterEqual(len(fields), 9, f"short DONE: {line!r}")
                return ("done", text.decode("utf-8", "replace"), fields[3:])
            elif fields[0] == b"ERROR" and fields[1] == str(rid).encode():
                return ("error", text.decode("utf-8", "replace"),
                        b" ".join(fields[2:]).decode())
            # telemetry lines (HWINFO/TIERS/EMAP/HITS/PROF/STAT) are skipped —
            # the forward-compat rule this repo's gateway now honors too
        raise AssertionError(f"no DONE/ERROR for id {rid}")

    # ---- the contracts -----------------------------------------------------

    def test_handshake_then_generate_and_prefix_reuse(self):
        proc = self.spawn()
        self.submit(proc, 1, 0, "hello world", max_tokens=6)
        kind, text, stats = self.collect_turn(proc, 1)
        self.assertEqual(kind, "done")
        emitted = int(stats[0])
        self.assertGreaterEqual(emitted, 1, "engine emitted no tokens")
        prompt_tokens = int(stats[4])
        self.assertGreater(prompt_tokens, 0)
        # second turn on the same slot extends the conversation: the engine
        # must reuse the KV prefix (prompt_tokens counts the FULL prompt; the
        # reuse is internal, but the turn must succeed and return promptly)
        self.submit(proc, 2, 0, "hello world and more", max_tokens=4)
        kind2, _, stats2 = self.collect_turn(proc, 2)
        self.assertEqual(kind2, "done")
        self.assertGreaterEqual(int(stats2[0]), 1)

    def test_greedy_determinism_across_processes(self):
        outs = []
        for _ in range(2):
            proc = self.spawn()
            self.submit(proc, 7, 0, "determinism probe", max_tokens=8, temp=0.0)
            kind, text, _ = self.collect_turn(proc, 7)
            self.assertEqual(kind, "done")
            outs.append(text)
            self._reap(proc)
        self.assertEqual(outs[0], outs[1], "greedy output differs across engine restarts")

    def test_cancel_is_acknowledged(self):
        proc = self.spawn()
        # long turn, then CANCEL while it decodes
        self.submit(proc, 3, 0, "cancel target", max_tokens=4096)
        proc.stdin.write(b"CANCEL 3\n")
        proc.stdin.flush()
        kind, _, code = self.collect_turn(proc, 3)
        # a fast tiny-model turn may legitimately finish before CANCEL lands:
        # the protocol allows either DONE or ERROR 3 CANCELLED, never a hang
        if kind == "error":
            self.assertIn("CANCELLED", code)

    def test_slot_busy_and_duplicate_id(self):
        proc = self.spawn()
        # two SUBMITs written back-to-back: the second targets the same slot
        # while the first is still active -> SLOT_BUSY
        p1 = b"first turn on slot zero"
        f1 = b"SUBMIT 10 0 %d 4096 0 1\n" % len(p1) + p1 + b"\n"
        p2 = b"second on same slot"
        f2 = b"SUBMIT 11 0 %d 8 0 1\n" % len(p2) + p2 + b"\n"
        proc.stdin.write(f1 + f2)
        proc.stdin.flush()
        kind, _, code = self.collect_turn(proc, 11)
        self.assertEqual(kind, "error")
        self.assertIn("SLOT_BUSY", code)
        # duplicate id on the OTHER slot while 10 is in flight
        p3 = b"dup id probe"
        proc.stdin.write(b"SUBMIT 10 1 %d 8 0 1\n" % len(p3) + p3 + b"\n")
        proc.stdin.flush()
        # the engine answers ERROR 10 DUPLICATE_ID; the original 10 continues.
        deadline = time.monotonic() + 30
        saw_dup = saw_done = False
        while time.monotonic() < deadline and not (saw_dup and saw_done):
            line = self._read_line(proc)
            f = line.split()
            if not f:
                continue
            if f[0] == b"DATA" and len(f) == 3:
                n = int(f[2]); self._read_exact(proc, n + 1)
            elif f[0] == b"ERROR" and f[1] == b"10" and b"DUPLICATE_ID" in line:
                saw_dup = True
            elif f[0] == b"DONE" and f[1] == b"10":
                saw_done = True
        self.assertTrue(saw_dup, "no DUPLICATE_ID for reused id")
        self.assertTrue(saw_done, "original request did not complete")

    def test_bad_frame_reported_then_clean_shutdown(self):
        proc = self.spawn()
        # correct byte count but a wrong trailing delimiter ('X' instead of
        # '\n'): the engine must answer ERROR <id> BAD_FRAME. Per the framing
        # contract a corrupted delimiter makes the stream UNRECOVERABLE (no
        # resync point in a length-prefixed protocol), so the engine then
        # drains and exits cleanly — never hangs, never crashes.
        # (A byte count LARGER than the actual payload is indistinguishable
        # from a slow writer and legitimately blocks — not tested.)
        proc.stdin.write(b"SUBMIT 20 0 5 8 0 1\nhelloX")
        proc.stdin.flush()
        proc.stdin.close()                     # X + EOF: delimiter check fails
        kind, _, code = self.collect_turn(proc, 20, timeout=20)
        self.assertEqual(kind, "error")
        self.assertIn("BAD_FRAME", code)
        rc = proc.wait(timeout=15)
        self.assertEqual(rc, 0, "engine must exit cleanly after a framing error")

    def test_context_exceeded_is_reported(self):
        proc = self.spawn(CTX=48)
        long_prompt = "word " * 500          # >> 48 tokens
        self.submit(proc, 30, 0, long_prompt, max_tokens=4)
        kind, _, code = self.collect_turn(proc, 30)
        self.assertEqual(kind, "error")
        self.assertIn("CONTEXT_EXCEEDED", code)
        # engine stays alive: a small turn still works
        self.submit(proc, 31, 0, "small", max_tokens=2)
        kind2, _, _ = self.collect_turn(proc, 31)
        self.assertEqual(kind2, "done")

    def test_eof_is_graceful_shutdown(self):
        proc = self.spawn()
        self.submit(proc, 40, 0, "final turn", max_tokens=4)
        kind, _, _ = self.collect_turn(proc, 40)
        self.assertEqual(kind, "done")
        proc.stdin.close()
        rc = proc.wait(timeout=15)
        self.assertEqual(rc, 0, "EOF on stdin must exit cleanly")


if __name__ == "__main__":
    unittest.main()
