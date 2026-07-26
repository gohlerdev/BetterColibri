# The serve protocols — engine ⇄ server wire format

The engine speaks two line-oriented protocols over stdin/stdout. Both are plain text
plus byte-counted payload frames; every outbound line is written with a trailing
`fflush`, one line per write. On Windows both ends of the pipe are switched to binary
mode at startup — the CRT's CRLF translation otherwise corrupts the sentinels and
stalls byte-counted reads (#195).

| protocol | entry | selected by | used by |
|---|---|---|---|
| **mux** (continuous batching, up to 512 KV slots engine-side; the gateway currently drives ≤16) | `run_serve_mux` | `SERVE_BATCH=1` | `openai_server.py`, `coli web` |
| **legacy** (single slot, interactive) | `run_serve` | `SERVE=1` (without `SERVE_BATCH`) | `coli chat` |

This document is the reference for the **mux** protocol; the legacy protocol is
summarized at the end. Line formats below are quoted from the emitting `printf`s in
`glm.c` — if this document and the code disagree, the code wins and this file needs a PR.

## Startup handshake (engine → server)

```
\x01\x01READY\x01\x01
STAT 0 0.00 0.0 <rss_gb>
HWINFO <cores> <ram_total_gb> <ram_avail_gb> <ngpu> <vram_total_gb> <cpu_name>|<gpu_name>
TIERS <vram_experts> <ram_experts> <disk_experts> <vram_gb> <ram_gb>
EMAP <rows> <cols> <hex>
```

The server must not send requests before `READY`. `HWINFO`/`TIERS`/`EMAP` are
telemetry (see below) and may grow — **servers must ignore line kinds they do not
recognize**; that is the protocol's forward-compatibility rule.

## Requests (server → engine)

```
SUBMIT <id> <slot> <bytes> <max_tokens> <temperature> <top_p>\n<payload>\n
CANCEL <id>\n
```

- `id` — non-zero u64, unique among in-flight requests.
- `slot` — KV slot index, `0 … KV_SLOTS-1` (`KV_SLOTS` env, 1–16, default 1). A slot
  holds one conversation's KV; the engine matches the tokenized payload against the
  slot's history and reuses the common prefix (truncate-and-extend), so stateless
  HTTP turns keep their cache.
- `bytes` — exact byte length of `payload` (UTF-8, may contain newlines). The engine
  reads exactly that many bytes after the header line, then one trailing `\n`.
- `payload` — the fully rendered prompt (the server owns the chat template).
- EOF on stdin = graceful shutdown: in-flight requests finish first.

Prefill is serial; decode is continuously batched — every active slot contributes
one row per forward.

## Responses (engine → server)

Per request, in order:

```
DATA <id> <n>\n<n bytes of UTF-8>\n          # a decoded token's text; repeated
HITS <rows> <cols> <hex>                    # routed-expert bitmap since the last HITS
DONE <id> STAT <emitted> <tok_s> <hit_pct> <rss_gb> <prompt_tokens> <length_limited>
```

Immediately before each `DONE` the engine emits a telemetry block for the finished
turn: `HWINFO`, `PROF`, `TIERS`, `EMAP`, `HITS` (formats below). `.coli_usage` is
persisted at every turn end, not only at exit.

Errors replace the stream: `ERROR <id> <CODE>` with codes `BAD_FRAME`, `BAD_REQUEST`,
`SLOT_BUSY`, `DUPLICATE_ID`, `EMPTY_PROMPT`, `CONTEXT_EXCEEDED <used> <limit>`,
`NOT_FOUND` (CANCEL of unknown id), `CANCELLED`. A `CANCEL` is acknowledged by
`ERROR <id> CANCELLED` after the slot's KV is persisted.

> Planned-but-not-yet-emitted line kinds (`PERF`, `ENTROPY`, `GPUS`, `TOPK`) were
> documented here before they existed; they are NOT currently printed. `REPIN`
> events go to **stderr** (`REPIN_VERBOSE=1`), not the protocol stream. Servers
> must ignore unknown line kinds regardless — that rule is what lets these lines
> land later without breaking deployed gateways.

## Telemetry lines

| line | format | meaning |
|---|---|---|
| `TIERS` | `TIERS <vram> <ram> <disk> <vram_gb> <ram_gb>` | expert count per tier + resident bytes |
| `HWINFO` | `HWINFO <cores> <ram_total> <ram_avail> <ngpu> <vram_total> <cpu>\|<gpu>` | host snapshot (GBs are floats) |
| `EMAP` | `EMAP <rows> <cols> <hex>` | one byte per expert, row-major over `rows×cols` (sparse layers +MTP × experts): `byte = (tier<<6) \| heat` — 2-bit tier (0 disk / 1 RAM / 2 VRAM), 6-bit log₂-bucketed usage heat |
| `HITS` | `HITS <rows> <cols> <hex>` | 1 bit per expert, experts routed since the previous `HITS` |
| `PROF` | `PROF <wall_s> <prompt> <completion> <edisk> <ewait> <emm> <attn> <head> <n_fw>` | this turn's phase wall times, seconds |

All telemetry is advisory: servers render what they know and skip the rest.

## HTTP surface (`openai_server.py`)

- `POST /v1/chat/completions`, `POST /v1/completions` — OpenAI-compatible;
  `POST /v1/messages` — Anthropic Messages API over the same generation path.
- `GET /health` — `{status: ok|degraded}`; scheduler/tiers/hwinfo details when authed.
- `GET /experts` — the latest `EMAP`/`HITS` state: `{rows, cols, map, hits, seq}` (authed).
- `GET /profile` — rolling per-turn `PROF` snapshots: `{seq, turns[≤120]}` (authed).
- `GET /*` — static hosting of `web/dist` (SPA fallback, path-traversal-safe), plus
  `experts.json` if published there (the measured expert atlas, #175/#218).

## Legacy protocol (`run_serve`, `coli chat`)

Interactive lines are prompts; control frames: `\x02RESET` (clear history),
`\x02MORE` (continue an NGEN-truncated answer), and
`\x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n`
(the pre-mux API mode). Responses are raw text terminated by `\x01\x01END\x01\x01`
plus a `STAT` line. New integrations should use the mux protocol.
