# Fix Plan — Deep-Dive Findings (2026-07-26)

Derived from `DEEP_DIVE_REPORT.md`. Baseline validated before any change:
`make -C c portable` clean, `make -C c test` = 141 Python tests OK (21 skipped) + all C tests pass.

Local toolchain: gcc 14 + Python 3.13 on Linux x86-64. **No nvcc / hipcc / macOS**:
CUDA fixes are made by careful review and validated by CI's `engine-cuda-syntax` /
`engine-hip-syntax` jobs; Metal-only work is deferred (see §Deferred).

Validation gate for every phase: `make -C c portable && make -C c test` must stay green.
Engine changes additionally keep the repo's byte-identical-output discipline: any change
that could alter numerics is either provably bit-identical (same operations, same order)
or explicitly gated.

---

## Phase 1 — Engine correctness fences (C, locally tested)

| ID | Fix | Sites | Approach |
|----|-----|-------|----------|
| B1/B2 | fmt=6 (E8) must never reach the generic decoders or unrotated matmul | colibri.c qt_load, qt_addrow :2096, qt_matvec_rows :2134, embed_row :1288, matmul_qt_ex :548 | Load-time fence: `qt_load()` (dense/resident tensors: embed, lm_head, attention, shared experts) refuses fmt=6 with a clear error — E8 is expert-only, experts load via `expert_load_impl` and get the FWHT rotation in `moe()`. Plus defensive aborts in the three per-row decoders so any future unknown fmt fails loudly instead of decoding as int2. |
| B3 | Dead duplicate `fmt==4` branch | colibri.c :2120-2126 | Delete the unreachable second branch. |
| H4 | fmt=6 sets `cuda_failed` + error spam through the CUDA gate | colibri.c :537 | Add `w->fmt!=6` to the gate (same treatment as fmt 5). |
| B4 | mlock/munlock wrong byte ranges for grouped/int3/E8 under COLI_MMAP | colibri.c :5750-5772 | New `qt_scale_bytes()` helper with the same per-format switch as `qt_bytes()`/`st_read_f32_cap`; both wire/unwire use it. Fixes over-wiring past the weight mapping and the munlock leak on REPIN promote. |

## Phase 2 — io_uring parity (C, locally tested incl. tests/test_uring)

| ID | Fix | Sites | Approach |
|----|-----|-------|----------|
| O1a | URING path ignores the dual-SSD mirror | colibri.c uring_load_add :1740-1805 | Compute `rep=expert_route(layer,eid)` with the same partial-mirror fallback as `expert_load_impl` (:1477-1478); route buffered and O_DIRECT fds through `rep_bfd`/`st_direct_fd_rep`. |
| O1b | URING path skips I/O accounting | uring_finalize_load :1828-1852 | Count `g_prof_io` and `g_mir_bytes/g_mir_nread` at finalize. `g_edisk_ns` (thread-seconds of read service) has no meaningful uring equivalent — documented as intentionally absent rather than faked. |

## Phase 3 — CUDA correctness (review + CI syntax validation; no local GPU)

| ID | Fix | Sites | Approach |
|----|-----|-------|----------|
| H1 | TC_INT4 launches empty WMMA kernels on HIP / sm<7.5 | backend_cuda.cu :823-826 | Mirror the W4A16 gate: `COLI_GPU_HAS_WMMA && (major>7 || (major==7 && minor>=5))`. |
| H2 | Ragged attention kernel uses per-row scales for fmt=4 | backend_cuda.cu :1064+ | Reject `fmt==4` in the host wrapper → clean CPU fallback (threading gs/ng through the kernel is follow-up work needing GPU validation). |
| H5 | `group_pending` cleared before the stream sync it protects | backend_cuda.cu :986-993 | Clear the flag only after a successful sync; on failure leave it set so the next issue refuses reuse of the possibly-wedged stream/buffers. |
| H8 | Dead duplicate `fmt==4` branch in `row_bytes` | backend_cuda.cu :95 | Delete. |

## Phase 4 — Gateway robustness (Python, locally tested)

| ID | Fix | Sites | Approach |
|----|-----|-------|----------|
| P1 | Any unknown engine stdout line permanently kills the gateway | openai_server.py :881-882 | Log-and-continue (rate-limited), making serve_protocol.md's forward-compat rule true. DATA framing checks stay strict. |
| P2 | `/health` reports "ok" after engine death | :1121-1133 | `status: "degraded"` when the dispatcher recorded an error or the process exited. Auto-restart is out of scope (documented follow-up): restart-with-warm-KV works but needs crash-loop guards. |
| I3 | Client disconnect during prefill undetected until first token | :930-940 | Poll `cancelled()` on a 1 s `events.get` timeout; send CANCEL before the first DATA. Frees the scheduler slot minutes earlier on long prefills. |
| I4 | Keepalive thread leaks on mid-stream errors; HTTP-500 body written into an open SSE stream | generation() | `try/finally` around the stream loop sets `ka_stop`; when headers are already sent, emit an SSE `data: {"error": …}` frame instead of `send_json`. |
| I5 | `/profile` skips the #SEC-8 auth gate `/health` and `/experts` have | :1144-1149 | Gate turn data behind `_is_authed()`; update the test that enshrined the old behavior. |

## Phase 5 — Performance (C, bit-identical only, locally tested)

| ID | Fix | Sites | Approach |
|----|-----|-------|----------|
| P#1 | `step_all` runs S separate S=1 lm_head matmuls (re-streams the vocab tensor per verify position) | colibri.c :4283-4285 | Batch rmsnorm rows + one `matmul_qt(...,S)` **when the kernel family cannot change with S**: fmt 0/1/3/4/5, or fmt 2 with `g_i4s<=1`, or inside `spec_pinned()`. lm_head is fmt 0 in default builds → always batched; the guarded fallback keeps byte-identity everywhere else. |
| P#3 | DSA indexer mallocs 4 buffers per position inside the OMP region | colibri.c :2458-2486 | Hoist to per-thread scratch sized once per call (max nk over the batch). Allocation-only change — the scalar math and its order are untouched (SIMD would reassociate; deferred). |
| P#2 | `matmul_e8` re-decodes every weight block per activation row | quant.h :1188-1213 | Hoist block expansion out of the S loop (decode once per row into scratch, dot per activation row). Same values, same dot order → bit-identical; validated by tests/test_e8_kernel against the reference fixture. |

## Phase 6 — Docs truth (serve_protocol.md, loader comment)

- serve_protocol.md: PERF/ENTROPY/GPUS/TOPK/REPIN are not emitted on stdout today —
  document what the engine actually emits (PROF on stdout; REPIN on stderr), fix the
  `/experts` field list, note KV_SLOTS bounds (gateway ≤16, engine mux ≤512), and keep
  the forward-compat rule (now actually honored by the gateway after P1).
- backend_loader.c vs colibri.c comment drift on optional `tensor_upload_g` (H7):
  make the symbol genuinely optional at resolve time so old DLLs degrade per-tensor
  as the engine comment promises.

## Deferred (needs hardware, weights, or measured A/B this box cannot provide)

| Item | Why deferred |
|------|--------------|
| CUDA `quant_matmul` vectorization (I1) | Needs GPU to measure; kernel perf work without a run is guesswork. |
| GPU kernels for fmt 5/6 (I2) | New kernels need numerics-parity validation on hardware (#298/#334 discipline). |
| Metal: resolve() index, scatter kernel, RESSET default (I5/I6) | macOS-only; cannot compile or test here. |
| attention_absorb_batch shared-memory tiling (I4-gpu) | GPU measurement required. |
| run_serve ⇄ run_serve_mux dedup (I6-serve) | Large refactor of the most delicate serve code; needs the protocol integration test (I7) first. |
| Engine auto-restart in gateway (I2-serve) | Feature work with crash-loop risk; degraded /health (P2) covers the observability gap now. |
| Heuristic-counter atomics (eheat/elast/used) for TSan | Benign today on 64-bit; blanket `_Atomic` churn deserves its own reviewed pass. |
| AVX-512/VNNI fmt=4 kernels, qt_matvec_rows vectorization, DSA SIMD dot | All change accumulation order → not bit-identical; the repo's rule is measure-then-merge, and there is no model on this box to measure against. |
| slot_of_eid residency map, O(E·K) selection heap, pinned-staging H2D, mirror share re-derivation | Perf refactors whose win shows only at prefill/scale; same measure-first rule. |
| Protocol integration test vs the real engine (I7), Tauri CI job (I8) | Valuable; separate PR-sized effort each. |

## Commit sequence

1. `docs: add fix plan` (this file)
2. `fix(engine): fence fmt=6 out of generic decode paths` (B1/B2/B3/H4)
3. `fix(engine): correct mmap wire byte ranges per quant format` (B4)
4. `fix(engine): route io_uring expert reads through the dual-SSD mirror` (O1a/O1b)
5. `fix(cuda): gate TC_INT4 on WMMA, reject fmt=4 ragged, harden group_pending` (H1/H2/H5/H8)
6. `fix(server): survive unknown engine lines, honest health, early cancel, SSE errors, auth /profile` (P1/P2/I3/I4/I5)
7. `perf(engine): batch step_all lm_head when kernel-family-safe` (P#1)
8. `perf(engine): hoist DSA indexer scratch out of the OMP loop` (P#3)
9. `perf(quant): hoist E8 block expansion out of the row loop` (P#2)
10. `docs: align serve_protocol.md with the code; optional _g DLL symbol` (Phase 6)

Each engine commit passes `make -C c portable && make -C c test` before it lands.
