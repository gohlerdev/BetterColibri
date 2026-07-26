# BetterColibri — Full Deep-Dive Report (2026-07-26)

Four-agent parallel analysis with coordinator cross-verification. Every cited
line number was read from source at commit 81f08a0 (v1.1.1).

Contents:
1. [Executive Summary](#executive-summary)
2. [Compute Core](#colibri-compute-core--deep-dive-report) — forward pass, MLA, MoE, quant kernels, speculative decoding
3. [Storage / Caching / Concurrency](#colibri-storagecachingconcurrency-deep-dive-read-only) — expert streaming, PIPE/PILOT, thread-safety audit
4. [GPU Backends](#colibri-gpu-backends--deep-dive-report) — CUDA/HIP/Metal
5. [Serving / Tooling / Frontend](#colibri-servingtoolingfrontend-layer--deep-dive-report) — gateway, byte protocol, web UI, CI

---

## Executive Summary

### How it works

- **Compute**: token → embed → 93 layers of [rmsnorm → MLA attention → residual → rmsnorm → MoE → residual] → lm_head. MLA stores **576 floats/token** (512-d latent + 64-d roped key) instead of 32,768; the "absorb" path scores queries directly in latent space, O(T·kv_lora) per head.
- **Storage**: each expert's 3 matrices are contiguous on disk → **one ~19 MB pread** into a reusable slab. Misses flow through a lock-free gen-tagged CAS job queue (PIPE) or io_uring; PILOT predicts next layer's routing (71.6% recall) and prefetches under a documented two-part safety invariant. Dual-SSD mirror splits reads by deterministic bandwidth-weighted hash.
- **GPU**: CUDA uploads experts whole to the least-loaded device and can release host RAM after upload (rematerializes from disk on CPU fallback); a resident pipeline keeps residuals on-device across layers, device router included. Metal uses bindless GPU-address arrays over zero-copy unified-memory slabs.
- **Serving**: Python gateway spawns the engine and speaks a length-prefixed SUBMIT/DATA/DONE byte protocol over stdin/stdout. KV slots prefix-match resent conversations → warm cross-turn prefix cache; continuous batched decode across slots.

### Latent bugs found (silent-wrong-answer class first)

| # | Bug | Where |
|---|-----|-------|
| B1 | fmt=6 (E8) falls through to the int2 decoder in three fallback paths → garbage + OOB scale read | colibri.c:2096, 2134, 1288 |
| B2 | fmt=6 dispatched by matmul_qt_ex without the required activation rotation | colibri.c:548 |
| H1 | COLI_CUDA_TC_INT4 not gated on WMMA availability → empty kernels return stale scratch on HIP / sm<75 | backend_cuda.cu:823-837 |
| H2 | Ragged attention kernel ignores fmt=4 grouped scales → silent wrong answers with g64 kv_b | backend_cuda.cu:427-457 |
| B4 | qt_wire_mmap/qt_unwire_mmap compute wrong mlock byte ranges for grouped/int3/E8 formats | colibri.c:5750-5772 |
| O1 | io_uring expert path never routes to the dual-SSD mirror and skips g_prof_io / DISK-CLASS accounting | colibri.c:1777-1801 |
| P1 | Gateway dispatcher treats any unknown engine stdout line as fatal, contradicting serve_protocol.md's forward-compat rule | openai_server.py:881-882 |
| P2 | /health reports "ok" after engine death; no restart path exists | openai_server.py:1125 |
| B3/H8 | Dead duplicate fmt=4 branches invite divergent edits | colibri.c:2104/2120, backend_cuda.cu:93/95 |
| H5 | expert_group_take clears group_pending before its stream sync → invariant violated on sync failure | backend_cuda.cu:986-993 |

### Top performance levers

1. **Batch step_all's lm_head** — currently S separate S=1 matmuls re-stream the ~0.5 GB vocab tensor once per verify position; step_decode_batch already does it right (colibri.c:4283-4285 vs 4319).
2. **Vectorize CUDA quant_matmul** — the workhorse GEMV reads weights unvectorized; the identical Metal kernel measured 1.5-2.1× faster with vec4 loads (backend_cuda.cu:128-161 vs backend_metal.mm:68-69).
3. **GPU kernels for fmt 5 (int3-g64) and fmt 6 (E8)** — these formats currently lose the entire GPU tier (the difference between ~6 tok/s and ~1 tok/s class).
4. **SIMD + per-thread scratch for the DSA indexer** — scalar dot plus 4 mallocs per position inside an OMP region (colibri.c:2458-2486).
5. **AVX-512/VNNI paths for fmt=4 grouped-int4** — the default modern container format runs at half vector width on Zen 4/5 (colibri.c:469-520, quant.h:168-202).
6. **Detect client disconnect during prefill** — currently unnoticed until the first DATA token, wasting minutes of prefill while holding a scheduler slot (openai_server.py:930-940).
7. **Per-layer slot_of_eid map** — replaces O(npin) linear residency scans run per lookup (colibri.c:3032-3035).
8. **Reuse batched kernels in qt_matvec_rows** — the hot absorb-path matvec is scalar for fmt 1/4/5 (colibri.c:2100-2137).

### Maintainability

- run_serve vs run_serve_mux: ~200-line duplication with drifted behavior — the legacy path still silently truncates over-long prompts, the exact #401 bug class the mux path fixed (colibri.c:5621 vs 5353-5370).
- serve_protocol.md documents line kinds (PERF/ENTROPY/GPUS/TOPK) that would today crash the gateway if emitted.
- Desktop Tauri shell is never built in CI.
- Thread-safety: core invariants (PIPE cursor, PILOT handshake, uring rings) are rigorous and ARM-portable. Residual formal races on heuristic counters (eheat/elast/ESlot.used) are benign on 64-bit but block TSan adoption and can tear on 32-bit ARM.

### Suggested fix order

1. H1 (one-line WMMA gate — silent wrong answers on AMD)
2. H2 (ragged fmt=4 scales — silent wrong answers)
3. B1/B2 (fmt=6 correctness fence outside the expert path)
4. B4 (mmap wire ranges before anyone runs MLOCK+MMAP with grouped containers)
5. P1/P2 (gateway resilience: ignore-unknown-lines + health degradation)
6. O1 (uring mirror routing + accounting)
7. Perf items in profile order on the target host (PROF=1 instrumentation already measures every relevant phase).

---


All line numbers were read directly from the source in this session. Files: `c/colibri.c` (6751 L), `c/quant.h` (1219 L), `c/sample.h`, `c/grammar.h`, `c/decode_batch.h`.

---

## 1. MECHANISM MAP: token → logits

### 1.1 Entry points

| Path | Function | Use |
|---|---|---|
| `step` | colibri.c:4261 | prefill / single decode, returns last-pos logits |
| `step_all` | colibri.c:4276 | speculative verify, returns logits for all S positions |
| `step_decode_batch` | colibri.c:4291 | mux server: one token per independent sequence, ragged per-row `KVState`/`positions` |

All three: `embed_row` per token → `layers_forward[_rows]` → final `rmsnorm` (colibri.c:800) with `final_norm` → `lm_head` matmul.

**Embed** (colibri.c:1263-1290): dequant-on-read of one row of `model.embed_tokens.weight` for fmts 0/1/2/4/5 (+int2 fallback). Out-of-range token → zero row (SEC-5, :1265).

### 1.2 Per-layer flow — `layer_forward_rows` (colibri.c:4056-4132)

1. `rmsnorm(in_ln)` per row (:4121)
2. `attention_rows` (:4122) + residual add (:4123)
3. optional `pilot_prefetch` for layer L+1 (:4124)
4. `rmsnorm(post_ln)` (:4129)
5. `moe()` if sparse else `dense_mlp` (:4130, dense_mlp at :3536) + residual add (:4131)

First `first_k_dense_replace` layers are dense (`l->sparse` set at :1135).

### 1.3 MLA attention with compressed KV — `attention_rows` (colibri.c:2330-2698)

**Projections** (:2397-2401, batched over S rows, `allow_idot=0` — exact int4 kernel because IDOT costs +0.117 nat/tok on attention projections, comment :2378-2385):
- `q_a` [q_lora=2048, D] → rmsnorm(`q_a_ln`) → `q_b` → Q[S, H·qk_head], RoPE on the last `qk_rope`=64 dims of each head (:2407, `rope_interleave` :817 with a thread-local cos/sin cache).
- `kv_a` [kv_lora+qk_rope, D] → per position (:2403-2419): first `kv_lora`=512 floats rmsnorm'd (`kv_a_ln`) into `Lc[layer][pos]` (the compressed latent), last `qk_rope`=64 floats RoPE'd into `Rc[layer][pos]` (`k_rot`, shared across all 64 heads).

**That is the 576 floats/token** (512+64) vs 64 heads × (192 nope + 64 rope + 256 v) = 32768 if K/V were materialized — the compression that makes GLM-5.2 context fit in 15 GB (comment :184-186).

**kv_b reconstruction.** `kv_b` is [H·(qk_nope+v_head), kv_lora] = [H·448, 512]. Row block `h·448..h·448+191` is W_K^h (k_nope), rows `+192..+447` are W_V^h. Two consumers:

- **Dense/prefill path** (:2660-2696): one batched `matmul_qt(kvb_all, Lc, kv_b, Tk-stL)` (:2664) rebuilds `k_nope|value` for every context token, then a standard causal `q·k` + softmax + `Σ a_t v_t` loop (collapse(2) over S×H, :2671-2693), then `o_proj` (:2695).

- **Absorb path** (:2491-2659, used when `kvs!=NULL` — ragged decode — or S≤4, gate :2505): by linearity,
  `q·k_nope_t = (W_K^hᵀ q_nope) · L_t` and `ctx^h = W_V^h (Σ_t a_t L_t)`.
  Per (s,h): `qabs[kvl] = Σ_d q_nope[d]·kv_b_row(rbase+d)` via `qt_addrow` (:2596, AVX-512 axpy at quant.h:66); score = `dot(qabs, Lt) + dot(q_rot, k_rot_t)` (SIMD'd, :2611-2626) scaled by `attn_scale = 1/√qk_head` (:933); softmax (:2628); `clat = Σ softmax·Lt` (SIMD AXPY, :2629-2650); finally `ctx^h = W_V^h clat` via `qt_matvec_rows` (:2651). Cost O(T·kv_lora) per head instead of O(T·(nope+vh)) plus the full kvb matmul. `o_proj` at :2655.
  The per-thread score buffer is sized `sc_cap` = max nt over the batch (:2514-2522) — the comment records a previous heap overflow when it was sized `Tk+1` with ragged positions.

### 1.4 DSA lightning indexer (:2420-2490)

Active when `has_dsa` and layer indexer weights present (:1216-1243). Layers are FULL (`idx_type=1`, compute selection) or SHARED (reuse the last FULL layer's selection, since `m->dsa_sel/dsa_nsel` persist across layers, :2489).

FULL layer: index keys `k_idx = layernorm(ix_wk·x)` + RoPE, cached in `Ic[layer][pos·index_hd]` (:2434-2443). Selection only when `nk > index_topk` (:2456, NO-OP on short prompts, matching the file header claim). Per query (OMP dynamic, :2451-2487):
- `qi = ix_wq · QR_s` (q_lora residual reuse) + RoPE per index head (:2459-2460)
- head weights `w32 = ix_wp · x_s` (:2462)
- score per context token t: `Σ_h w32[h]·ReLU(qi_h·k_t/√hd)/√nh` (:2465-2472)
- top-keep via quickselect threshold (`partial_select_desc` :2168, replacing a full qsort, #356), then two position-order scans build `dst[]` bit-identically (:2479-2484).

Both attention paths then iterate `tlist` instead of `st0..pos` (:2599-2602, :2678-2681).

### 1.5 MoE — `moe()` (colibri.c:2718-3534)

**FASE A (routing, :2751-2938):** one batched router matmul `logits_all = x·router` (:2789); per position: `sigmoid(logit)` then `choice = sigmoid + e_score_correction_bias` (:2793 — noaux_tc: the bias enters *selection* only); top-K by `choice`, gate weights taken from the *bias-free* sigmoid `logit[best]` (:2881-2884); optional `norm_topk` renorm (:2911) then `× routed_scale` (:2912). `n_group=1` enforced at load (:934). Pre-routed shortcut (`g_pre_idx`, GPU-side router from Metal layer-CB or CUDA `pipe_layer_sparse` :3970-3993) copies the selection and still bumps usage/heat/recency clocks (#417 fix, :2760-2788). Opt-in variants: CACHE_ROUTE max-rank fill (:2796-2879), TOPP per-position truncation (:2893-2898).

**FASE B (batch-union, :2939-2945):** dedupe `idxs[]` across all S positions into `uniq[]` via `seen[E]`. Each unique expert's weights are read from disk/cache once and matmul'd against all its rows — this is what makes prefill and MTP verify affordable. `EXPERT_BUDGET` (decode-only, S≤4, :2957-3011) can drop lowest-aggregate-gate misses, keeping cache hits free.

**FASE C/D (resolve + compute, :3028-3495):** blocks of 64 uniques; per slot resolve pin → LRU ecache → miss into `ws[]` (:3031-3037). Misses load via PIPE async workers, io_uring, or blocking OMP `expert_load` (:3100-3112, load impl :1441-1656 — coalesced contiguous pread of gate/up/down into one slab). Compute per expert: `expert_gate_up` fused pair kernel (:270-276, kernels quant.h:205/469), `silu(g)·u` (:3346), `down` matmul (:3348), weighted accumulate into `out` (:3349). GPU tiers (CUDA groups incl. async issue/take overlap :3162-3230/3356-3384, Metal resident/miss subsets :3040-3158) and the XEXP single-OMP-region path (:3243-3298) are alternates with CPU fallback. LRU promotion swaps `ws[]` slabs into ecache at block end (:3488-3494).

**FASE E (shared expert, :3496-3526):** one S-row gate/up/silu·mul/down over `sh_*`, added to `out` — unless already fused on GPU (`g_pre_sh` :3501).

**fmt=6 (E8/IQ3)** experts require the activation FWHT rotation: `xe = e8_rot_rows(x)` built once per moe() call and shared across routed experts (:3329-3332), and the down input rotated per expert (:3347). Kernel `matmul_e8` quant.h:1188.

### 1.6 lm_head

`step`: rmsnorm(final_norm) on last row, `matmul_qt(logit, last, lm_head, 1)` (:4268-4270). `step_all`: **per-position S=1 loop** (:4283-4285 — see Improvement #1). `step_decode_batch`: properly batched `matmul_qt(..., S)` (:4319).

### 1.7 MTP draft/verify contract and SPEC_PIN

- **Draft** `mtp_draft` (:4347-4383): DeepSeek-V3 chain — `cat = [enorm(emb(tok)) ; hnorm(rmsnorm(final_norm, h))]` → `eh_proj` → one forward through the MTP layer (row `n_layers` in the KV arrays, `kv_start[n_layers]` marks its decode-only window :4350) → `shared_head.norm` → `lm_head` argmax. Chains G tokens by feeding back its own hidden.
- **Verify** `spec_decode` (:4536-4615): batch `[next, draft…]` through `step_all` (:4587); accept while argmax matches (greedy) or via Leviathan rejection sampling (`accept = rndu() < p(draft)`, resample with draft banned, :4592-4601). Lossless by construction.
- **Absorb** `mtp_absorb` (:4387-4408): after verification, the *true* hiddens `m->h_all` of accepted positions are pushed through the MTP layer once (batch) so its KV stays in sync — the draft-time KV rows for those positions are overwritten with truth. `m->hlast` is fixed up to the last *accepted* position (:4606).
- **Why SPEC_PIN exists** (:256-259, :2379-2385, :3244, :4541-4546): several kernel choices switch on S (`g_i4s` gate in `matmul_qt_ex` :549, fused-pair :271, Metal GEMM :531). Draft forwards are S=1 but verify forwards are S=1+g; without pinning, the same weights would run through *different* quantization kernels (int8-activation IDOT vs exact f32) at draft vs verify time, so the model would verify against slightly different logits than it drafted from, killing acceptance and reproducibility (#163). `spec_pinned()` freezes the S=1 kernel family for the whole draft+verify window; `g_spec_live` is raised only inside `spec_decode` (:4543/:4611).
- `run_serve_mux` sets `g_draft=0` (:5400) because MTP/n-gram speculation is not ragged-safe; grammar drafts pull a slot out of the shared batch for one single-sequence `step_all` (:5469-5495), matching how prefill is already serial (:5379).

---

## 2. IMPROVEMENTS (evidenced)

**#1 — `step_all` lm_head is S separate S=1 matmuls.** colibri.c:4283-4285. `lm_head` is [vocab≈155k, 6144] — ~0.5 GB at int4. Every verify forward with g drafts re-streams that entire tensor 1+g times, once per position, and each S=1 call is a fresh OMP region. `step_decode_batch` already does it right (:4314-4319, one `matmul_qt(..., S)`). Batch the rmsnorm into `[S,D]` and do one S-row matmul. Caveat: with SPEC_PIN this changes the S seen by `matmul_qt_ex`'s IDOT gate, but `spec_pinned()` already forces the S=1 family regardless of S (:549), so drop-in inside the spec window; outside it, byte-differences would need an A/B. Expected impact: large for MTP-heavy decode — lm_head weight traffic per verify forward drops ×(1+g)→×1; `t_head` is a tracked phase so it's directly measurable. Risk: low-medium (numerics identical under the pin; verify with TOKENS=1 A/B).

**#2 — `matmul_e8` re-expands every weight block per activation row and has no SIMD.** quant.h:1195-1213. The `e8_expand_sub` decode (grid lookups, parity, sign math) runs inside the `s` loop, so at prefill S=512 every super-block is decoded 512 times; the inner dot is a scalar 32-float loop. Hoist expansion out of the S loop (expand row once into a `float[I]` scratch when S>1, or tile S), and SIMD the FMA (the expanded `w[32]` is contiguous). Expected impact: E8 prefill matmul several× faster; even S=1 gains from vectorizing the dot. Risk: low, pure kernel change validated by the existing oracle fixture (test_e8_kernel.c referenced at quant.h:855).

**#3 — DSA indexer inner loop is scalar and mallocs per position.** colibri.c:2458-2486. Per (position, context-token, head): scalar `d0 += qhp[i]*kt[i]` over hd dims (:2469) — the same shape the absorb path SIMD'd at :2612-2626 (#442) — and four `falloc`/`free` pairs *per position inside an OMP region* (:2458,:2461,:2464,:2479 → :2486), hammering the allocator lock at prefill. AVX2 the dot, and hoist the scratch to per-thread buffers like `sc_all` (:2523). At long contexts (nk up to `max_t`, nh=index heads) this is the dominant DSA cost. Risk: low; reassociation in the dot changes scores at ~1 ulp, which can flip top-k tie ordering — keep the scalar tail order or accept the flip as the absorb path did.

**#4 — O(E·K) selection loops with O(K) inner dedupe scans, copy-pasted 4×.** The pattern `for kk<K { for e<E { for j<kk if(idx[j]==e) ... } }` is O(E·K²) and appears at colibri.c:2881-2884 (router), :3584-3587 and :3598-3601 (la_predict twice), :2802-2817 (CACHE_ROUTE, doubly), plus the masked-argmax variant in `pilot_prefetch` :3880-3882. With E=256, K=8 it's tolerable, but this runs per position per sparse layer (93 layers × prefill S). A single pass with an 8-element insertion heap is O(E·log K), and one shared helper kills the duplication (these copies have already drifted: pilot uses `ch[best]=-2e30` mutation, router uses the dedupe scan). Risk: low; keep tie-break order (first-index-wins) to stay bit-identical.

**#5 — Row-count discovery scans idxs[] once per (expert, block) — O(nb·S·K) repeated 3-4×.** moe() re-derives "which rows use expert e" by scanning all S×K routing slots per expert: :3316-3317 (CPU loop), :3055-3057 + :3074-3078 inside MB_BUILD (twice per Metal block), :3179-3180 (CUDA early-issue). At prefill (S≈512, nu≈100+) that's millions of comparisons per layer per block. Building a per-expert row list once right after FASE B (inverse index: E buckets, one S×K pass) makes every consumer O(rows). Risk: low, mechanical.

**#6 — `matmul_i4_grouped_pair` / `matmul_i4_grouped` ignore the AVX-512 and VNNI kernels.** colibri.c:469-520 and quant.h:168-202 only have an AVX2 16-wide path, while per-row int4 has `dot_i4f_avx512` (quant.h:49) and the whole IDOT VNNI family. On this host (Zen 5, AVX-512 + VNNI) fmt=4 experts run at half the available vector width, and grouped scales compose fine with a 512-bit accumulator when gs is a multiple of 32 (candidates at :965 are). Also `matmul_qt_ex` :549 excludes fmt=4 from IDOT entirely — a grouped-scale idot variant (per-group integer accumulate × scale) is straightforward. Impact: fmt=4 is the default container format for modern packs (#242/#334); expert matmul is the top decode phase. Risk: medium (numerics change per-group accumulation order; gate like `g_i4_acc512`).

**#7 — `qt_matvec_rows` per-row `double` loop resists vectorization; hot in absorb.** colibri.c:2100-2137. Runs H=64 times per (s, layer) on the absorb path (:2651) computing vh=256 rows × kvl=512. Only fmt=2 has the AVX-512 shortcut (:2114-2116, and only when `I%32==0` and the toggle is on); fmt 1/4/5 rows are scalar. This is per-token decode work in the hottest attention path (`t_acore`). Reuse the batched kernels (it is just a matmul over a contiguous row slice — `matmul_i4(y, clat, q4+rbase·rb, s+rbase, 1, I, vh)` shape) instead of the bespoke scalar loop. Risk: low-medium (accumulation-order change; same tradeoff already accepted for #442).

**#8 — Code duplication between run_serve and run_serve_mux / step-family.** Both serve loops implement independently: prompt-prefix KV reuse and truncation (:5372-5377 vs :5620-5630), per-request temp/top-p save/restore (:5387 vs :5596/5683), STAT/PROF emission (:5284-5296 vs :5673-5682), grammar lifecycle (:5348-5349 vs :5661). The prefix-match logic has already drifted (mux refuses over-long prompts loudly :5364-5371; run_serve raw mode silently truncates at `maxctx-8-g_draft` :5621 — the exact #401 failure class the mux comment describes). Similarly `step`/`step_all` differ only in what they keep (:4261-4287). Extract shared helpers (prefix_sync, emit_stat, step_core with an `all_logits` flag). Impact: maintainability plus one live inconsistency fixed. Risk: low.

Bonus observations: `softmax` (:811-812) and `rmsnorm` (:800-803) are scalar and run per (s,h) in attention — vectorizing softmax's exp loop is a further small win; `attn_pipe_prefill` mallocs `chost` per layer per call (:2244) where a scratch would do.

---

## 3. LATENT BUGS

**B1 — fmt=6 falls through to the int2 decoder in `qt_addrow`, `qt_matvec_rows`, and `embed_row` → silent numerical corruption.** `qt_addrow` (colibri.c:2065-2098) handles fmt 0/4/5/1/2 and then *unconditionally* treats the remainder as int2 (:2096-2097). Same in `qt_matvec_rows` (:2134-2135) and `embed_row` (:1288-1289). fmt=6 stores 98-byte lattice blocks; decoded as int2 nibble pairs this produces garbage *and* reads `t->s[row]` (:2088) which for fmt=6 is a 1-float tag allocated at :1028 (`qsalloc(1)`) — an out-of-bounds read for any row>0. Reachable the moment an E8-packed `kv_b`, `lm_head` or `embed` ships (the loaders at :1027-1029 and `qt_resolve_fmt` :988 happily produce fmt=6 for *any* tensor, not just experts). This is exactly the bug class the comment at :2068-2073 documents for fmt=4 (#298). Fix: explicit fmt==6 branch or a loud abort.

**B2 — `matmul_qt_ex` dispatches fmt=6 without the required activation rotation.** colibri.c:548: `if(w->fmt==6){ matmul_e8(y,x,...); return; }`. quant.h:1142-1148 states fmt=6 stores `W@Q` and *activations must be transformed* (`e8_rot_rows`) before the matmul; moe() does this explicitly (:3329-3332, :3347). Any fmt=6 tensor reaching the generic path (dense attention projections, shared expert, lm_head — all loaded through the same `qt_from_disk`) is multiplied against unrotated x → wrong output, no error. Either rotate inside the dispatch (cost: per-call FWHT, defeating the shared-x optimization) or hard-refuse fmt=6 outside the expert path at load time.

**B3 — Dead duplicate `fmt==4` branch in `qt_matvec_rows`.** colibri.c:2104 and :2120 both test `t->fmt==4`; the second is unreachable. Harmless today, but the two bodies differ (the live one accumulates a float `acc` per group into a double; the dead one is the "per-gruppo, come matmul_i4_grouped" variant) — a future edit to the wrong copy silently no-ops. Delete one.

**B4 — `qt_wire_mmap`/`qt_unwire_mmap` compute wrong byte ranges for grouped/int3/E8 formats.** colibri.c:5750-5772: `scale_b = O*4` and `weight_b = qt_bytes(t) - scale_b`. For fmt=4, scales are `O*ceil(I/gs)*4` (qt_bytes :119-121), for fmt=5 `O*ng*4`, for fmt=6 they're in-block. So mlock/munlock wire `weight_b` *larger* than the actual weight mapping (locking unrelated adjacent file pages or failing) and lock only the first `O` floats of the scale array. On the REPIN promote path `qt_unwire_mmap` under-unlocks, leaking locked pages over a session — the precise failure mode the comment at :5744-5746 says this code exists to prevent. Fix: derive per-format scale bytes (same switch as `st_read_f32_cap`'s cap at :1036-1039).

**B5 (minor, latent) — `pilot_worker` ring-index advance under URING.** colibri.c:3766: with `g_uring`, `pilot_uring_batch` consumes entries and stores `pilot_r` itself (:3730), then the worker `continue`s — correct — but the non-uring `pilot_realload` path (:3768) processes entry `r` and *then* stores `r+1` (:3771); if `g_pilot_real` was toggled per-entry this would double-process, and more practically a crash inside `pilot_realload` leaves `pilot_r` stale forever. Currently safe because the flags are set once at startup, but the invariant is implicit; worth an assertion or unifying the advance.

**B6 (design smell, not a bug) — `g_temp`/`g_nuc` are process globals mutated per request** (:5387, :5481, :5506). Single-threaded scheduler makes it safe today, but any future parallel sampling (or a signal-time read) races. The `GrDraft` struct comment (:636-640) shows the codebase already migrated grammar state per-request for exactly this reason; sampling params deserve the same treatment.

---

**Summary of highest-leverage actions:** batch `step_all`'s lm_head (#1) for immediate MTP-decode wins; guard fmt=6 outside the expert path (B1/B2) as a correctness fence; fix the mmap wire ranges (B4) before anyone runs MLOCK+MMAP with grouped containers; then the kernel work (#2, #6, #7) in order of the profile on your target host.

---


# Colibri storage/caching/concurrency deep-dive (read-only)

All line numbers verified by reading the files. `st.h`, `uring.h`, `tier.h`, `kv_persist.h`, `telemetry.h` cited by their own line numbers; unqualified lines are `c/colibri.c`.

---

## 1. MECHANISM MAP: lifecycle of an expert read

### 1.1 Indexing and fd universe
- `st_init_multi` (st.h:248-360) scans `*.safetensors` in snap_dir plus `COLI_MODEL_DIRS` split dirs (dedup by basename → search-path semantics, st.h:217-240), parses each JSON header with hostile-input guards (hlen bound st.h:287, offset bounds st.h:318, shape-overflow st.h:325-333, numel/nbytes cross-check st.h:343-346), and builds an open-addressing hash `hidx` (st.h:352-359) because linear scan over ~120k tensors cost tens of seconds/token (st.h:46-48).
- Each file gets a buffered fd plus an **O_DIRECT twin** opened eagerly (st.h:92-98); lookup via `st_direct_fd` (st.h:109-111).
- **Dual-SSD mirror** (`COLI_MODEL_MIRROR`): `st_mirror_init` (st.h:134-178) accepts a mirror file only if size and full header are byte-identical to the primary (so data_offsets match by construction), populating `mfds/mdfds`. Mirror is never written; partial mirrors are allowed per-file.

### 1.2 Miss detection
`moe()` FASE C/D (3028-3038): for each of the ≤64 unique experts of the block, scan `pin[layer]` (3033, counts `hit_pin`), then `ecache[layer]` (3034-3035, counts `hit_ecache` and bumps `used` from the global `eclock`), else assign a `ws[]` slot and count a miss (3036). Misses are dispatched to PIPE (3101-3107) or blocking OMP loads (3108-3111).

### 1.3 Slab layout: why 3 matrices in one pread
`expert_load_impl` (1441-1657): gate/up/down tensor names are formatted (1451-1453); in the pre-quantized container the three weight tensors are contiguous in the file, so the loader sorts by offset (1590-1591), checks contiguity (1592-1594), and issues **one ~19 MB coalesced pread into `slab`** (1611) instead of three; scales `.qs` go into `fslab` (1626-1630). The QT structs are then zero-copy *views* into the slab (1648-1655). One large sequential read is what a cold NVMe wants; the comment at 1380 documents this.

### 1.4 fd selection: primary vs mirror vs O_DIRECT
- Replica choice: `expert_route(layer,eid)` (1344-1349) is a **deterministic hash** mapped against `g_mir_share`/256. Determinism is load-bearing: WILLNEED prefetch and the demand pread must hit the same fd/page-cache and no expert may be cached on both drives (1340-1343). `g_mir_share` is derived from `COLI_DISK_WEIGHTS` or a startup O_DIRECT bandwidth probe (`mirror_probe_bw` 5809-5835, `mirror_setup` 5841-5868).
- Fallback: partial mirror → rep forced to 0 (1478); read error on mirror → one-time warning + retry on primary (`mir_pread` 1363-1377), inheriting the EINTR/short-read honest loop of `pread_full` (1394-1412).
- O_DIRECT: only when `DIRECT=1` and the block is contiguous; offset/len are 4K-aligned into the slab (1598-1608), fallback to buffered on short direct read (1610).
- `COLI_MMAP=1` bypasses all of this: experts become views into `mmap`s of the shards (`map_of_fd` 1307-1326 under `g_map_mtx`), with CPU pre-touch page faulting + `MADV_WILLNEED` so the GPU never demand-faults (1495-1508).

### 1.5 Concurrency invariants

**PIPE gen-tagged cursor** (design comment 1859-1885, code 1908-2009):
- `cur = (gen<<8)|index`, single atomic (1909). Main thread is sole writer of `gen` (monotonic → no ABA). `pipe_dispatch` writes `njobs/layer/eids/ready` relaxed, then **release-stores** `cur=(g+1)<<8` (1982-1987) and broadcasts.
- Workers **acquire-load** `cur` (1929), so all batch state is visible; a job is grabbed only by a winning `CAS(cur, cur+1)` whose comparand carries the gen (1934-1935); `eids[i]` is read **after** the winning CAS (1937). A straggler resumed after a new generation always fails its CAS and re-reads. Per-slot completion: `ready[i]` release-store (1939), main `pipe_wait` acquire-spin or condvar (1990-2009). Per-expert waits before the end-of-block LRU swap make every grab complete within its generation (3304-3310, 3484-3487); under Metal a block-level drain is mandatory before handing slabs to the GPU (3134-3145).

**PILOT two-part safety invariant** (stated 765-775):
1. **MATMUL path**: the pilot only loads into `ecache[layer]` with `layer > g_cur_moe_layer` (check under `g_pilot_mx` at 3626-3628, URING variant 3684-3687); `moe()` takes ownership of its layer by locking `g_pilot_mx`, release-storing `g_cur_moe_layer=layer`, and waiting for `g_pilot_inflight[layer]==0` (2723-2727). Hence no half-loaded slot is ever matmul'ed. Ownership resets to -1 per forward (4139-4143).
2. **SCAN path**: `pilot_prefetch` residency scans on the *future* layer — exactly the layer the worker mutates — so those scans take the same `g_pilot_mx` (3890-3897; couple variant 3828-3835), preventing torn reads of `ecn[]/eid`.
- Slot publication: pthread pilot hides the slot as `eid=-1` while the pread runs *outside* the lock, and only bumps `ecn` after success (3652-3666). URING pilot uses a *visible reservation* `eid=-(eid+2)` (3718) so duplicate enqueues are detected (3691, 3834, 3896) while the slot is never treated as resident or evictable (3699).

### 1.6 Eviction policy
- **Working-set → LRU promotion**: end-of-block swap-buffer exchange with the LRU slot chosen by minimal `used` (3488-3494).
- **LRU vs LFRU**: the plain ecache eviction is pure LRU on `used` (3492, 3636, 3700). **LFRU** (`tier_lfru_score` tier.h:30-33: `(heat<<8)|recency`, so one frequency count outweighs max recency) is used for (a) the **pilot evict guard** (#441/#490): a speculation may not evict a resident that is warm (≥2 accesses) *and* hotter by the 25%+4-freq hysteresis (3645-3651, URING 3702-3714); (b) **repin** victim/candidate selection via `tier_pick_lfru` (tier.h:35-54, called at 5064-5065).
- **RSS guard** (#403): at safe points every ~16 emitted tokens, frees LRU slabs *in place* under `g_pilot_mx` (free must stay under the lock or the pilot could pread into a freed slab — 5118-5124) and ratchets `ecap` down (5131).

### 1.7 Pin promotion/demotion
- **Startup**: `pin_load` (5925-…) reads `.coli_usage` (written atomically by `stats_dump_q` tmp+rename, telemetry.h:168-176; loaded by `usage_load` telemetry.h:180-186), sorts by count, clamps to RAM/VRAM budget, loads pins OMP-parallel (6014-6016). On NUMA, per-layer **arenas** replace per-slab mbind to avoid VMA explosion (#419, `pin_arena_bind` 5885-5922); arena slices are marked with `aslab/afslab` and never freed (ESlot 156-162).
- **Live repin**: `repin_pass_limit` (5137-5220), between turns only, max ~16 swaps, using decaying `eheat` (tier_decay 5219) + LFRU with hysteresis; GPU swaps move the CUDA tensor handle instead of re-uploading a slot (5164-5188). `expert_host_release/ensure` (2012-2038) detach/reattach host slabs for VRAM-tier experts under `CUDA_RELEASE_HOST`, with arena slices detached but not freed (2027).
- **KV persistence** (kv_persist.h): `.coli_kv` append-only, `nrec` written last for crash consistency (kv_persist.h:81-83), header-mismatch rejection on load (92-94).

---

## 2. THREAD-SAFETY AUDIT

Threads: main/compute (+OpenMP teams), PIPE workers (≤16), pilot worker (1), URING kernel io-wq.

| Structure | Guard | Assessment |
|---|---|---|
| `PipePool.cur/njobs/eids/ready` | atomics; release publish 1987 / acquire consume 1929; CAS acq_rel 1934 | **Sound**, portable to ARM. Correctness never depends on the mutex (parking only). |
| `ws[64]` slabs | Invariant: pipe_wait per dispatched slot before LRU swap (3310, 3484-3487); Metal drain 3143-3145 | Sound; documented in-code. |
| `ecache[l]/ecn[l]` when `g_pilot_real` | `g_pilot_mx` + layer-ownership handshake (2723-2727 / 3626 / 3684) | Sound. **But**: `moe()`'s hit scans (3034) and next-block prefetch scans (3117-3122) run *without* the lock, relying on the ownership invariant — safe only for `layer <= g_cur_moe_layer`. The next-block scan reads `ecache[layer]` for the *current* layer, which the pilot no longer touches: OK. |
| `expert_is_resident` (2710-2716) used from CACHE_ROUTE FASE A (2830) | **No lock** | Reads `ecn[layer]`/`eid` of the current layer during FASE A, which happens *after* the ownership store at 2724 — the pilot has already stopped writing this layer. OK, but subtle and undocumented at the call site. |
| `pilot_q` ring, `pilot_w/pilot_r` | `volatile unsigned` + `__atomic` load/store with acquire/release (3837-3841, 3899-3903, 3761-3771) | SPSC-per-role in practice, but there are **two producers** (pilot_prefetch on main, couple_prefetch on main — same thread, so OK) and one consumer. The `volatile` qualifier is redundant; the atomics carry the ordering. Sound on ARM. |
| `g_pilot_inflight[256]` | `g_pilot_mx` (779) | Sound. |
| `m->eheat/elast/eaccess_clock` | **None** — written plain by main in FASE A (2778, 2783, 2908-2909), read by pilot worker inside `g_pilot_mx` (3645-3650, 3706-3711) and by `couple_prefetch` | **Formal C11 data race.** Consequences are heuristic-only (evict-guard decisions), values are monotone-ish u32s, so on x86 TSO and AArch64 this yields at worst a stale decision. Not a correctness bug, but ThreadSanitizer will flag it; `_Atomic`/relaxed accessors would make it clean at zero cost. |
| `ESlot.used` (uint64) | Written under lock by pilot (3661, 3744) and plain by main (3035, 3493); `eclock` bumped with `__atomic_add_fetch` relaxed | The **read** of `Sl[z].used` in the pilot's LRU pick (3636, 3700) is under the mutex, but main's write at 3035 is not mutex-paired → formal race. On 64-bit x86/ARM a torn read is impossible for aligned u64; **on 32-bit ARM a torn `used` read is possible**, producing a bogus LRU victim (still only a placement error). Flag for portability. |
| `m->hits/miss/hit_ecache/ereq` | Plain increments, main thread only in moe() | OK (single writer, read at safe points). But `mux_done`/PROF read them concurrently only from the same thread — fine. |
| `m->ecap` | Written by `rss_guard` at safe points; read by pilot at 3635/3694 without lock | rss_guard runs at request boundaries with no moe in flight, but the **pilot worker is not quiesced** at that safe point — a pilot batch could be running (its residency checks would drop layers ≤ current, but between requests `g_cur_moe_layer` is at its last value, not −1... actually it is reset only at forward start, 4139-4143, so between turns it holds the *max* layer = drops nearly everything; benign in practice, racy in theory). `ecap` decrement while the pilot compares `nn<m->ecap` is a benign off-by-transient. |
| `g_mir_bytes/nread`, `g_prof_io`, `g_edisk_ns`, `g_pilot_loads/drops`, DISK-CLASS counters | `_Atomic`, relaxed (1374-1375, 1631, 1669) | Sound; stats only. |
| DISK-CLASS busy-wall | dedicated `g_dc_wall_mx` (401-427) | Sound, deliberately mutexed for auditability. |
| `g_maps[]` mmap table | `g_map_mtx` (1298, 1308-1325) | Sound. |
| io_uring SQ/CQ | acquire/release helpers (uring.h:31-36) on head/tail; single-owner rings (`g_ub_pipe` main, `g_ub_pilot` pilot worker) | Sound and ARM-portable. **Invariant not enforced**: nothing prevents a future caller from touching a batch from two threads; ownership is by convention (uring.h:4-5). |
| `st_open_fd`/`shards` | Built at init, then read-only ("lookup poi thread-safe", st.h:93) | Sound. `st_mirror_init` re-init closes fds while readers could be in flight (135-138) — only called at startup, but worth a comment. |
| KV disk (`kv_disk_append`) | Single-threaded at turn boundaries | Sound; crash-safety by nrec-last (kv_persist.h:81-82). |
| `pipe_wait` blocking mode | ready release-store *before* lock+broadcast; waiter rechecks under lock (1997-2006) | No lost wakeup; sound. |

**x86-TSO dependence summary**: the engine is largely clean (explicit acquire/release in the PIPE cursor, uring ring, pilot ring). The residual risks on ARM are (a) plain u64 `used` / u32 `eheat/elast` cross-thread accesses (torn/stale reads → wrong heuristic victim only), and (b) `g_pilot_inflight` is fine (mutex). Nothing output-affecting was found to depend on TSO.

---

## 3. IMPROVEMENTS (evidence, impact, risk)

1. **URING path ignores the dual-SSD mirror.** `uring_load_add` reads via `l->tw[k]->fd` / `st_direct_fd` only (1777, 1787, 1793, 1800) — no `expert_route`/`st_fd_rep`, unlike `expert_load_impl` (1477-1478, 1598, 1611). With `URING=1` + a mirror, 100% of expert bytes hit the primary drive, forfeiting up to ~2x cold-read bandwidth the mirror exists for. Fix: compute `rep` in `uring_load_add` and substitute `rep_bfd`/`st_direct_fd_rep`. Impact: up to +80-100% cold-decode disk bandwidth on dual-SSD URING hosts. Risk: low — same offsets by mirror construction (st.h:127-133); needs the same partial-mirror fallback as 1478.

2. **URING path also skips I/O accounting and DISK-CLASS.** `uring_finalize_load` (1828-1852) never adds to `g_prof_io` or `g_edisk_ns` and never classifies, while the pread path does (1631, 1669, 1586-1641). PROF=1 reports under URING silently under-report disk service to ~0, corrupting the very numbers used to tune this subsystem (`edisk`/`ewait` verdicts, 4784-4801). Fix: account bytes at finalize; time from submit to finalize per load. Impact: correct tuning data. Risk: minimal, measurement-only.

3. **Synchronous next-block WILLNEED from the compute thread.** moe() issues `expert_prefetch` inline for the next 64-expert block (3113-3124) — six `posix_fadvise` calls per expert. The PILOT section itself documents that fadvise submit blocks ~0.5 ms per call under a saturated disk queue (3614-3617) and therefore moved pilot hints to a dedicated thread. The batch-union prefill path (where nu>64 actually happens, S large) still pays it inline. Fix: route these through the existing `pilot_q` ring (hint mode), or gate on queue depth. Impact: removes up to tens of ms per prefill block on saturated disks. Risk: low; hints are advisory and output-preserving (2047-2052).

4. **`IOSQE_ASYNC` unconditionally punts every read to io-wq** (uring.h:104). Correct for cold buffered reads (comment 99-103), but for page-cache-warm reads it forces a worker-thread hop where inline completion would be ~free; with `DIRECT=1` on NVMe, modern kernels complete O_DIRECT without blocking the submitter anyway. Fix: set IOSQE_ASYNC only for buffered fds, or make it env-tunable. Impact: lower per-read latency and fewer io-wq wakeups on warm/direct workloads (URING pipe reads are 0.5-3 ms scale, so shaving the ~10-30 µs wq handoff matters mostly at high hit rates). Risk: low; needs A/B per kernel version.

5. **Lock ping-pong in couple/pilot residency pre-checks.** `couple_prefetch` takes and releases `g_pilot_mx` once per candidate (COUPLE_K × depth per position, 3828-3835), and `pilot_prefetch` once per K prediction (3891-3897) — up to ~10-16 lock round-trips per layer per token on the *compute* thread, contending with a pilot worker that holds the lock across slot selection. Fix: hoist one lock per (layer) around the whole candidate loop, or snapshot `(eid)` arrays of pin+ecache under one lock into a stack bitmap. Impact: fewer main-thread stalls when PILOT_REAL is loading; micro but on the token critical path. Risk: low — same invariant, coarser section (keep it short: the scan is O(npin+ecn)).

6. **O(npin) linear residency scans on every lookup.** Every hit-check walks `pin[layer]` and `ecache[layer]` arrays (3032-3035, 2710-2716, 5041-5065). With PIN_GB=all class loads (~250 pins/layer, 692-693) and CACHE_ROUTE calling `expert_is_resident` per rank-window candidate per position (2826-2833), this is O(S·Mwin·(npin+ecn)) integer compares per layer. Fix: per-layer `int16 slot_of_eid[E]` map (E ≤ 512, 1 KB) maintained at pin/promo/evict points — all of which already run under existing locks/safe points. Impact: measurable FASE C and CACHE_ROUTE routing cost reduction on big-pin hosts; also removes the repin `ids[4096]` copy (5061-5063). Risk: moderate — must be updated at every eid mutation site (moe swap 3488-3493, pilot 3652/3718/3743, rss_guard 5113, repin 5196); the kind of invariant this codebase documents well.

7. **LRU slab capacity churn between layer classes.** `ws[]` slots are reused across layers, and MTP int8 experts are 2x int4 size (154-155); a slot bouncing between classes triggers free+posix_memalign of ~19-38 MB (1527-1541), and the swap-buffer promotion (3493) moves those slabs into ecache, so LRU slabs also come in two sizes and get reallocated on reuse for the other class. glibc serves >128 KB via mmap/munmap (as the rss_guard comment notes, 5079-5080), i.e., each churn is a TLB-shootdown-bearing syscall pair plus page faults on first touch. Fix: size ws/LRU slabs to the max class once (`wtot_max+8192`), or keep per-class free lists. Impact: removes recurring mmap/munmap + soft-fault cost per MTP-boundary miss (every draft/verify cycle when MTP is on). Risk: low — `slab_cap` logic already tolerates oversized slabs; costs a bounded RAM overcommit that `cap_for_ram` (6126-6214) would need to account with the max-class `eb`.

Bonus observations: (a) mirror hash balance — `expert_route`'s avalanche (1346-1348) is fine, but the share is computed once at startup from an 8-block probe (5816, 5822-5832); a slow-warming drive (thermal/SLC cache) skews the split for the whole run — periodic re-derivation from `g_mir_bytes/g_mir_nread` service times would self-correct. (b) `st_read_f32` at st.h:409 computes `esz` before checking `dtype==3`, so a U8 tensor read through the f32 path would pass the esz=2 guard only accidentally — currently unreachable (raw path used for quant), worth an assert. (c) `pilot_worker` idles on `usleep(200)` (3763) — a futex/condvar park would cut idle wakeups; trivial.

---

**Bottom line**: the core concurrency design (gen-tagged PIPE cursor, two-part PILOT invariant, reservation-eid URING pilot) is rigorous and documented to an unusual standard, with correct acquire/release chains that hold on ARM. The real gaps are *coverage* asymmetries: the URING path lost the mirror and the accounting that every other path has, and a few heuristics rely on formally-racy plain reads that are benign today but block TSan adoption and 32-bit ARM.

---


# Colibri GPU Backends — Deep-Dive Report

All line numbers below were read directly from the files cited.

---

## 1. MECHANISM MAP

### 1.1 CUDA expert tier: upload, budget clamp, release-host, placement

**Tensor upload & caching.** Each `QT` carries an opaque `ColiCudaTensor*` (`colibri.c:107-113`). `qt_cuda_upload` (`colibri.c:293-303`) routes fmt=4 through `coli_cuda_tensor_upload_g` (extra `gs` arg via a thread-local `g_upload_gs`, `backend_cuda.cu:597-653`, kept thread-local because pin-load uploads happen from parallel OpenMP threads) and refuses fmt 5/6 outright (`colibri.c:294`). Upload is lazy-once: if `*tensor` already exists, the call is a shape/format revalidation only (`backend_cuda.cu:605-614`) — crucially checked **before** `!weights`, so a slot whose host pointers were freed by `CUDA_RELEASE_HOST` still passes (the comment at cu:606-610 records the earlier bug where host-released experts silently never computed on GPU). int4 weights are converted offset-binary→signed in place on device by `offset_to_signed_s4` (cu:124-126, launched at cu:630-632).

**VRAM budget clamp.** In `pin_load` (`colibri.c:5962-5991`): per device, `remaining[i] = free_VRAM − g_cuda_dense_projected[i] − CUDA_RESERVE_GB` (default 2 GB, `colibri.c:6500`, computed at 5972-5978). `CUDA_EXPERT_GB=auto` sets budget = sum of headroom (5984); an explicit budget is honored even beyond headroom, degrading per-expert instead of clamping (#491 comment, 5979-5983). Experts are placed **whole** on the least-loaded device that fits (`best<0||placed_b[i]<placed_b[best]`, 6029-6043); an upload failure zeroes that device's `remaining` and tries the next (6046-6050).

**Release-host & rematerialization.** With `CUDA_RELEASE_HOST` (default on multi-GPU, `colibri.c:6503`), `expert_host_release` (`colibri.c:2012-2032`) munlocks and frees (or arena-detaches, #419) the slab and NULLs all QT host pointers. The reverse, `expert_host_ensure` (2033-2038), re-attaches the arena slice and calls `expert_load` with demand=0. It is invoked on every CPU-fallback edge: the plain expert loop (`colibri.c:3343`), Inc.4 take-failure fallback (3374), sync-group fallback (3466), and REPIN gpu_swap restore (5151-5153, parallelized because serial 20 MB reads cost ~0.7 s/pass). Budget accounting trick: with release-host, the RAM pin budget is **increased** by the VRAM prefix estimate (`npin += prefix_est`, 5985-5988) since prefix RAM is transient staging returned after upload — the fix for the 6-GPU host that pinned only leftovers (comment 5963-5968).

**Multi-GPU round-robin + colocate.** Dense tensors round-robin by projected bytes: `slot = g_cuda_rr++ % g_cuda_ndev` (`colibri.c:1048-1052`). Then `qt_cuda_colocate` (1063-1072) forcibly moves the whole attention chain (o, q_a, q_b, kv_a → kv_b's device, 1127-1131) and the shared-expert chain (sh_gate/up/down, 1147-1150) onto the layer "home device", adjusting the projected-bytes ledger. Optional `COLI_CUDA_ATTN_SHARD` splits kv_b by heads across devices (`layer_cuda_shard_kvb`, 1073-1087). Per-thread device caching (`g_current_device`, `backend_cuda.cu:76-88`) avoids the measured 14.3s→25.4s regression from redundant `cudaSetDevice` when the expert loop alternates devices.

**Grouped expert execution.** `coli_cuda_expert_group` (cu:775-924) packs ≤64 same-shape experts into one `GroupDesc[]` (cu:51-55) upload plus one activation H2D, then dispatches one of five kernel families by format census (`all_s4/all_q4/any_g4`, cu:794-797): W4A4 tensor-core (`COLI_CUDA_TC_INT4`, quantize-activations-to-s4 + `grouped_s4_wmma`, cu:827-837), per-expert W4A16 WMMA above `COLI_CUDA_TC_W4A16_MIN` rows with naive small-row fallback (cu:838-877, the #431 launch-flood fix), packed-W4 grouped (`grouped_hidden_w4_dual`+`grouped_down_w4`, cu:878-887), grouped-int4 fmt=4 (`*_g4_*`, cu:888-894), and a generic `weight_at` path that explicitly **rejects** stray fmt=4 (cu:895-904). Async issue/take split (Inc.4, cu:936-993): decode-scale (total≤8 rows, cu:955), one outstanding issue per device (`group_pending`, cu:956), removes the measured ~0.45 ms/call host-side sync tax vs ~0.18 ms GPU work (comment cu:927-935). moe() issues before its CPU loop and takes after (`colibri.c:3161-3230` pass-1 early issue, 3355-3384 take with per-device CPU recompute fallback).

### 1.2 Resident pipeline (COLI_CUDA_PIPE)

`pipe_layer_sparse` (`colibri.c:3917-4053`) keeps the residual `x_dev` on the layer's home device across the whole layer: persistent numbered scratch slots (27 per device, `pipe_buf`, cu:43, `coli_cuda_pipe_scratch` cu:1228-1233 — grow-only, "78 × ~10 alloc/req were pure churn"), cached layernorm weights per layer (`ln_dev`, `colibri.c:3925-3939`), device rmsnorm → resident attention (`attn_pipe_prefill`, 2224-2327, projections + rope + KV assembly all on device, host KV stays canonical via a small `chost` download 2264-2269) → residual add → post-ln → **device router** (#431 PR-A: `pipe_router_logits` E-block GEMV + sigmoid, single-thread `pipe_router_select` for exact CPU tie-break parity, packed `[idx|w|keff]` read back in one tiny D2H, cu:1286-1361; consumed via the `g_pre_idx` shortcut, `colibri.c:4024`) → shared expert issued async on GPU **before** moe() runs on CPU (overlap comment 3996-4015) → routed result uploaded and added async; no end-of-layer sync — the next layer's sync `pipe_download` is the implicit ordering point (4006-4009). Layer-boundary hops use `pipe_peer_copy` (cu:1528-1534); the decode gate is device-count aware (S≥1 single-GPU, S≥8 multi-GPU, `colibri.c:4156-4161`, +49% on 5070 Ti per #273). CPU fallback restores from an on-device snapshot (slot 14, 3944-3946, 4191-4192).

**Resident expert accumulation** (#431 PR-C0, cu:1362-1462): for S=1, `resident_issue` P2Ps the input row from home, broadcasts to `count` rows (`bcast_row`), runs the grouped-W4 kernels on the owning device's stream, reduces with `weighted_sum_rows` in **fixed expert order** (no atomics — "the 9.20.7 lesson", cu:1216-1218), P2Ps the partial into a per-issue slot on home and records `ev_done`. `resident_take` makes the home **legacy stream** wait on every issue event then `sum_slots` in issue order (cu:1451-1462) — deterministic, zero host bytes. Failure at take is deliberately **fatal** (`exit(1)`, `colibri.c:4041-4045`): dropping routed experts would be a wrong answer, not a slow one. Note the fixed-cap sizing in issue (cu:1418-1425): buffers reserved for 64 experts, not `count`, because a `reserve()` realloc could free memory that the previous layer's still-queued stream work reads.

**KV device shadow** (`kv_dev_sync`, `colibri.c:2199-2217`): append-only mirror of Lc/Rc on kv_b's device, invalidated by rewinds (`kv_dev_valid[layer]=pos_base` on rewrite, 2270); decode absorb uses `attention_absorb_kvdev` (cu:1573-1590) uploading only q (~KB).

### 1.3 Metal: bindless moe_gemv slab resolution

Host slabs (expert LRU slabs, KV pools, weight file maps) are wrapped zero-copy iff 16 KB page-aligned (`wrap`, mm:408-413) and registered in `g_slabs` under `g_slab_mtx` (mm:322-325, 479-507; registration sites `colibri.c:745-748, 1318-1319, 1528-1563, 4237-4246`). `resolve()` (mm:509-515) linearly maps a host pointer to `(MTLBuffer, gpuAddress+offset)`. `moe_submit` (mm:904-965) resolves each expert's 6 pointers into `waddr[]/saddr[]` arrays baked into small shared buffers; the `moe_gemv` kernel (mm:70-99) then dereferences `device const ulong* waddr` per row: `w = (device uchar*)(waddr[erow[gr]])` — true bindless GPU-address indirection. One SIMDGROUP per output row, 4 rows/threadgroup, 8-value vectorized loads (measured 1.5-2.1×, 358-389 GB/s, mm:68-69). Because the buffers are only indirectly referenced, each unique slab buffer must be declared with `useResource:` per command buffer (mm:943-945) — unless `COLI_METAL_RESSET=1` attaches one persistent `MTLResidencySet` to the queue (macOS 15+, mm:327-391), moving residency off the dispatch path; deferred-commit adds are flushed before any submit that skips `useResource:` (mm:377-383, 910-912), and `resset_remove` commits **immediately** before the caller frees slab memory (mm:361-373). The overlap trick: resident experts submit to GPU *before* miss experts' disk preads (`moe_block_begin` at `colibri.c:3083-3094`, misses in a second sync submit at 3146-3151), with a mandatory PIPE drain barrier before handing miss slabs to the GPU (3134-3145). Layer-CB decode (`coli_metal_layer_decode`, mm:724-824) fuses in_ln→attention→residual→post_ln→shared expert→router→exact top-K (serial `r_top8` or the exact-match 32-lane parallel `r_top8_par`, ~93× faster, mm:222-278) in one command buffer, hard-gated to GLM-5.2 shapes (`colibri.c:4068-4072`).

### 1.4 Known CPU-vs-GPU numerics divergence

Documented in `GPU_BACKENDS.md` (Known behavior notes): GPU float MACs round differently than the CPU IDOT int8-dot kernels, so **greedy output is not token-identical across backends** (#100/#163 shape-dependence class), and MTP acceptance drops ~40%→~31% on GPU-heavy configs. The CUDA device router explicitly clones CPU tie-breaking exactly, leaving only dot/expf rounding in the divergence class (cu:1286-1295). Metal, by contrast, claims byte-identical greedy output (docs/metal.md "dequant→f32-MAC ... byte-identical") and enforces `r_top8_par` bitwise equality by memcmp in metal-test (mm:229-238). The async CUDA group path is byte-identical **to the sync path** by construction (same small-batch kernels, cu:933-935), not to the CPU. Note also CPU `attention` uses `double` accumulation in `qt_matvec_rows` (`colibri.c:2102`) and rmsnorm on GPU uses double partials only in `pipe_rmsnorm_rows` (cu:1186) — Metal's `a_rmsnorm` accumulates in float (mm:111).

---

## 2. IMPROVEMENTS (evidenced, with impact/risk)

**I1. `quant_matmul` has fully uncoalesced weight access — the workhorse kernel.**
`backend_cuda.cu:128-161`: thread `i` of a 256-block strides `weight_at(weights, fmt, row, i)` with stride 1 element along `I`, but for fmt=2/4 each thread reads byte `q[i>>1]` — adjacent threads hit adjacent bytes, so coalescing is actually acceptable for int4, but **each byte is fetched twice** (both nibble owners) and fmt=1/0 reads are 1-byte/4-byte per thread with no vectorization, unlike the Metal shader's `uchar4`/`float4`8-value loads (mm:42-45) or the CPU AVX paths. The `grouped_hidden_w4*` family (cu:300-334) already proves the fix (consume each packed byte once via `unpack_s4`); `quant_matmul` is still used by *every* pipe_gemm, o_proj, attention projection, and the async decode group path (cu:969-975). Impact: this kernel dominates the resident pipeline's GEMV time at decode; a `uchar4`-vectorized variant would plausibly gain 1.5-2× (matching the Metal measurement at mm:68-69). Risk: low — same accumulation order per thread can be preserved; validate with the existing cuda-test.

**I2. fmt 5 (int3-g64) and fmt 6 (E8) fall back to CPU everywhere on GPU — quantifiably expensive.**
`colibri.c:294` (`qt_cuda_upload` returns 0 for fmt 5/6), `colibri.c:537` (`w->fmt!=5` gate in matmul_qt_ex — note fmt 6 is *not* gated there, see H4 below), Metal `mm_gemv` supports only fmt 0-3 (mm:564 `fmt<0||fmt>3`; moe path fmt 1-2 only, mm:909). Cost: an fmt=5/6-quantized model loses the **entire** expert tier, resident pipeline, fused attention and Metal MoE — i.e. the difference between the documented 5.8-6.8 tok/s (6×5090 pipeline) and the pure CPU baseline (~1 tok/s class), because `cuda_eligible` experts are what feed every group path (`colibri.c:3321`). The int3 decode loop is simple (2-bit plane + 1-bit plane + per-group scale, `colibri.c:2081-2087`); a `grouped_*_i3` kernel is structurally identical to the existing `grouped_*_g4` family (cu:341-370) with a different `weight_at`. E8 is harder (rotation + lattice decode) but the rotation is already hoisted per layer on CPU (`xe`/`e8_rot_rows`, `colibri.c:3329-3332`). Impact: high for #132's "quality/size sweet spot" format; risk: medium (new kernels need the numerics-parity discipline of #298/#334).

**I3. Double-copy through pinned staging on every H2D/D2H in the group/W4A16 paths.**
`coli_cuda_expert_group` async mode does `memcpy(ctx->host_x, x, xb)` then `cudaMemcpyAsync` (cu:817-819), and on the way back `cudaMemcpyAsync`→sync→`memcpy(y, ctx->host_y, xb)` (cu:908-912); same in `shared_mlp_w4a16` (cu:758-771) and `expert_group_issue` (cu:962-963). The extra CPU memcpy is pure overhead once the *caller's* buffers could be pinned: `group_x`/`group_y` in moe() (`colibri.c:3022-3023`) are `falloc`'d pageable memory rebuilt per call. Registering (or allocating) those as pinned once per model would let the H2D start without staging and would let issue() return without the host memcpy on the critical path. Impact: at decode the staged copy is small, but at prefill (`group_enabled && S<=4096`, `colibri.c:3021`) `xb` is up to 4096·K·D·4 bytes per block — the H2D ms visible in `COLI_CUDA_PROFILE` (cu:587-592). Risk: low; pinned allocs are already plumbed (`reserve_pinned`, cu:483-486).

**I4. `attention_absorb_batch_kernel` serializes score compute over K=512 per thread and softmax scans over T sequentially.**
cu:393-423: the score loop (`for k=0;k<K;k++ a+=qa[k]*lt[k]`) is a length-512 serial dot per (t, thread) with `latent` re-read from global memory each time (only `qa` is in shared memory); `cl` accumulation (cu:417-418) re-reads all T×K latents again. For decode with T in the thousands this kernel reads T·K·4 bytes twice per head-block from global. Tiling `latent` through shared memory (it is reused across all 256 threads of the block) or using one warp per t with warp-shuffle reduction would cut global traffic ~2× and vectorize the dot. Impact: this is the decode attention core for the kv_dev path (cu:1584) and the pipeline (cu:1483, 1547, 1566); risk: medium (must preserve the documented softmax ordering to stay in the accepted divergence class).

**I5. Metal `useResource:` cost — make RESSET the default and drop the O(n) dedup loop.**
`moe_submit` deduplicates `use` with a linear scan per resolve (`add_use`, mm:916) and issues one `useResource:` per unique slab per CB (mm:943-945) — the cost "scales with LRU cache size (mechanism history v5)" (mm:936). The residency-set path exists but is opt-in experiment E5 (mm:457). Additionally `resolve()` itself is a linear scan of `g_slabs` under a mutex, run **6× per expert per submit** (mm:917-925) — with hundreds of registered slabs that's thousands of locked scans per layer. A sorted/interval-tree slab index (or per-ESlot cached resolve) is a straightforward win. Impact: measured via `g_t_setup` (mm:962); risk: low for the index, moderate for defaulting RESSET (needs macOS 15 gate + the documented hazard-tracking argument at mm:934-942).

**I6. Metal scatter-add is single-threaded CPU.**
`moe_finish` (mm:979-982): `for gr<R for dd<D out+=w*hh` — R·D f32 FMAs on one thread while `g_t_scatter` is separately tracked precisely because it shows up. Either dispatch an `a_add`-style weighted scatter kernel into a shared out buffer (rows are per-expert-unique already), or at least OpenMP the loop. Impact: at prefill R can be ~S·K; risk: low.

**I7. `cuda_failed` is a permanent one-shot disable per tensor, with no error classification.**
`colibri.c:536-543`: any single `coli_cuda_matmul` failure (including a transient `cudaErrorMemoryAllocation` from a scratch `reserve()` during a temporary VRAM spike, cu:468-476) sets `w->cuda_failed=1` forever; the resident copy is *not* freed, so VRAM stays occupied while compute goes CPU for the rest of the process. Options: distinguish sticky errors (ECC, illegal address → disable + free) from resource errors (retry with backoff, or free the tensor to reclaim VRAM). Also note `qt_cuda_reset` clears `cuda_failed` (`colibri.c:289-292`) only when an expert slot is recycled — dense tensors never recover. Risk: low; the fault-injection hook `COLI_GPU_FAIL_AFTER` (cu:674-682) already exists to test exactly this.

**I8. HIP parity: no WMMA and no async-quality validation.**
`backend_gpu_compat.h:21` pins `COLI_GPU_HAS_WMMA=0` under HIP, so AMD always uses the naive kernels — rocWMMA is named as follow-up (h:12-14). But also: the compat macro list (h:22-59) covers exactly the symbols used; any new CUDA API use (e.g. `cudaMemcpy2DAsync`, graphs) silently breaks HIP compile — the CI syntax job catches it, but a comment-enforced checklist or a `#pragma` poison of raw `cuda*` names in `.cu` would harden the "one source, two vendors" rule stated in GPU_BACKENDS.md. Also `hipDeviceProp_t` `major` on gfx11 is 11 — the `compute_major>=7` runtime checks (cu:752, 838) would pass, which is exactly why the compile-time flag is load-bearing; keep it, but the `tc` gate at cu:823-826 checks **only env + shapes, not `COLI_GPU_HAS_WMMA`** — see H1 below.

---

## 3. LATENT BUGS AND PORTABILITY HAZARDS

**H1. `COLI_CUDA_TC_INT4` path is not gated on WMMA availability or compute capability — silent garbage on HIP/sm<75.**
cu:823-837: `tc = getenv("COLI_CUDA_TC_INT4") && all_s4 && shape checks` — unlike the W4A16 branch (cu:838: `all_s4 && COLI_GPU_HAS_WMMA && ctx->compute_major>=7 && ...`), the tc branch never checks `COLI_GPU_HAS_WMMA` or `compute_major`. `grouped_s4_wmma`'s body is compiled away below `__CUDA_ARCH__ 750` (cu:248) and always under HIP, so on gfx GPUs or sm_70, enabling the env var launches **empty kernels** and the D2H returns stale scratch as expert outputs — the exact failure mode the compat header's comment warns about (h:9-14). `quantize_s4_rows` (cu:233-244) would still run, but the wmma stages produce nothing. Fix: mirror the W4A16 gate (`COLI_GPU_HAS_WMMA && compute_major>7 || (==7&&minor>=5)`).

**H2. Ragged attention kernel ignores grouped scales (fmt=4) — the #298 bug survives on one path.**
`attention_absorb_ragged_kernel` (cu:427-457) uses `wscale[rbase+d]` and `wscale[row]` directly (cu:438, 456) — the per-row (fmt=2) semantics — with **no `absorb_scale`/gs/ng** parameters, and `coli_cuda_attention_project_ragged` (cu:1064+) does not reject `w->fmt==4`. The batch/single kernels were fixed for exactly this ("the per-row semantic that crashed #298's g64 kv_b", cu:112-122). A g64-grouped kv_b reaching the SERVE ragged path (`colibri.c:2526-2543`, gated only on `COLI_CUDA_ATTN` + eligibility) computes with wrong scales — a silent wrong answer, since `qt_cuda_upload` happily uploads fmt=4 kv_b. Fix: either thread gs/ng through the ragged kernel like the batch one, or reject fmt==4 at cu:1068.

**H3. Race window between expert release/eviction and in-flight GPU work (the asked-about class).**
Three related shapes:
- **CUDA REPIN swap during in-flight async group.** `repin_pass_limit` (`colibri.c:5160-5216`) calls `qt_cuda_update` (VRAM refresh, memcpy into the *same* device buffer, cu:655-672) with **no stream synchronization against a pending `expert_group_issue`** on that device. `coli_cuda_tensor_update` uses synchronous `cudaMemcpy` on the default stream while the group runs on `ctx->stream` (non-blocking, cu:525) — the default legacy stream does not order against non-blocking streams, so a repin refresh can overwrite weights the queued group kernels are still reading. In practice repin runs between tokens and issue/take is bracketed inside one moe() call, so the window needs `COLI_GROUP_ASYNC` plus an adaptation pass interleaving — narrow, but nothing structural prevents it. The resident path has the same exposure for `pipe_*` allocations, but its fixed-cap reserve comment (cu:1418-1421) shows the authors are aware of the general class.
- **Metal slab unregister vs in-flight CB.** Acknowledged in-tree (mm:938-942): a slab unregistered+freed+reused while an async CB still reads it is a CPU-side race outside Metal hazard tracking, "held by the engine's own slot lifecycle". The engine does honor it (`moe_block_end` before LRU swap, drain barrier at `colibri.c:3134-3145`; rss_guard frees only under `g_pilot_mx` with the eid=-1 hiding protocol, 5101-5128 — but rss_guard **does not** check for an outstanding `ColiMetalMoeHandle` on that slab; `moe_block_begin`'s CB may still be executing when rss_guard's `coli_metal_unregister`+`free` runs from the same thread between blocks. Today rss_guard is called from `repin_pass_limit` (5138) between forwards, so the invariant holds only by call-site timing, not by construction.
- **`expert_host_release` vs concurrent CPU readers**: release NULLs `q4/s` pointers non-atomically (2029-2030) while the pilot/lookahead threads may hold an `ESlot*`; the pin tier is never evicted so this is only reachable through the repin gpu_swap path, again serialized by call-site timing only.

**H4. fmt=6 is not excluded from the CUDA matmul gate.**
`colibri.c:537` gates only `w->fmt!=5`. For fmt=6 the code passes `w->q4` (E8 lattice blocks) as `weights` with `fmt=6` into `coli_cuda_matmul` → `row_bytes(6,I)` returns 0 (cu:90-97) → `!rb` fails the upload (cu:618) → returns 0 → **`w->cuda_failed=1` is set permanently and an error line printed** (`colibri.c:541-543`) even though nothing is wrong. Benign outcome but wrong bookkeeping and log spam; also any *future* fmt added to `row_bytes` without kernel support would silently compute garbage since `quant_matmul`'s `weight_at` falls through to the int2 decode for unknown fmts (cu:103-109) — the same trap #334 closed for fmt=4 in the group path (cu:895-899) is still open in `quant_matmul` itself.

**H5. `coli_cuda_expert_group_take` clears `group_pending` before the sync it depends on.**
cu:986-993: `ctx->group_pending=0` is set **before** `select_ctx`/`cudaStreamSynchronize`; if the sync fails, the caller gets NULL and falls back to CPU recompute (fine), but the pending flag is already cleared so a subsequent `issue` can immediately reuse `ctx->host_x/host_y` and the stream while the failed/incomplete prior work may still be queued. Combined with `reserve()`'s free-then-alloc realloc (cu:468-476), a growth realloc in the next issue can free buffers the wedged stream still references. Low likelihood (requires a sync failure), but the failure path violates the one-outstanding-issue invariant it exists to protect.

**H6. Metal `coli_metal_matmul` leaks tensor bookkeeping and `ensure()` leaks buffers.**
`ensure` (mm:395-398) returns a new buffer without releasing the old one when growing — under ARC the old `id<MTLBuffer>` assigned over (`g_gg = ensure(g_gg,...)`, mm:996-999) is released, OK — but `attn_scratch_init`'s `ascore_` growth path (mm:702, 755) also relies on this; fine under ARC. The real issue: `coli_metal_tensor_free` decrements `g_tensor_count`/`g_tensor_bytes` without any lock (mm:893-897) while `coli_metal_matmul` increments them (mm:573) — both can run from different threads (matmul_qt is called inside OpenMP-parallel regions gated by `!omp_in_parallel()`, but tensor_free is not). Stats-only corruption, but the same unlocked pattern applied to `g_moe_fb++` inside `resolve` failures (mm:919-924) runs on the caller thread only — OK today, fragile tomorrow. Similarly the CUDA side's `g_gpu_calls++` in `fault_injected` (cu:679-682) is an unsynchronized `long` on a multi-threaded path (benign: test hook only).

**H7. Windows loader: partial-symbol DLLs and stale-ABI hazard.**
`backend_loader.c` resolves ~48 symbols, failing hard if any is missing (bl:199-201) — good — but `tensor_upload_g` alone gets an extra null-check at the wrapper (bl:339-342) implying older DLLs are expected, while `RESOLVE(tensor_upload_g, ...)` at bl:216 makes a missing symbol fatal — the graceful path is dead code and an old DLL simply disables CUDA entirely, contradicting the `colibri.c:297-301` comment ("an old DLL without the _g symbol returns 0 and the tensor simply stays CPU-side"). Cosmetic inconsistency, but worth aligning: either make `_g` optional at resolve time or drop the misleading comments. The DLL-hijack mitigation (absolute path + `LOAD_WITH_ALTERED_SEARCH_PATH`, bl:160-183) is sound.

**H8. `row_bytes` dead branch.** cu:90-97: the `fmt==4` case at cu:95 is unreachable (already handled at cu:93). Harmless, but it invites divergence if someone edits one branch.

---

### Priority ranking
1. **H1** (ungated TC_INT4 WMMA — silent wrong answers on HIP/sm_70, one-line fix)
2. **H2** (ragged fmt=4 scales — silent wrong answers, small fix)
3. **I1** (quant_matmul vectorization — largest broad perf lever)
4. **I2** (fmt 5/6 GPU kernels — unlocks the tier for the recommended quant format)
5. **H5/H3** (async lifecycle hardening), then I3/I5/I6/I7.

---


# Colibri Serving/Tooling/Frontend Layer — Deep-Dive Report

Read-only analysis. Lines cited were actually read.

---

## 1. MECHANISM MAP

### 1.1 Request lifecycle: `POST /v1/chat/completions` → tokens

**HTTP entry** — `ThreadingHTTPServer` subclass `APIServer` (openai_server.py:972-986), one Python thread per connection (`daemon_threads = True`, :973). `APIHandler.do_POST` (:1176-1205) does, in order: Host-header DNS-rebinding guard `_check_host` (:1043-1061), auth via constant-time `hmac.compare_digest` on `Authorization: Bearer` or `x-api-key` (:1026-1036), `read_json` with a 4 MiB body cap (`MAX_BODY = 4 << 20`, :41, enforced :1068), model-id check (:1078-1081), then routes to `chat_completion` (:1413), `completion` (:1617), or `anthropic_messages` (:1437).

**Prompt rendering** — the gateway owns the chat template. `render_chat` (:374-449) emits the GLM-5.2 template literally: `[gMASK]<sop>`, `<|system|>`, `<|user|>`, `<|assistant|><think></think>`, tool declarations byte-matching `chat_template.jinja` (:393-406), tool results as `<|observation|><tool_response>…</tool_response>` (:439-442). The Anthropic path (`/v1/messages`) is a pure translation layer: `anthropic_to_openai` (:475-549) + `anthropic_tools` (:552-595) reshape the request, then reuse the same `render_chat`/`generation_options` pipeline (:1453-1465), so generation logic stays single-sourced.

**Admission** — `GenerationScheduler.admit` (:116-180). Capacity = KV slots (:983). A waiter is admitted when its target slot is free AND no strictly earlier waiter wants that slot (per-slot fairness replacing strict FIFO head, :137-153, "#B2"). Queue overflow → 429 `queue_full` with `Retry-After: 1` (:125-128); timeout (default 300 s) → 429 `queue_timeout` (:161-166); client-gone while queued → `ClientCancelled` (:155-159). Wait time is surfaced as `x-colibri-queue-wait-ms` (:1248).

**Engine subprocess** — `Engine.__init__` (:757-782) spawns the C binary with `SERVE=1 SERVE_BATCH=1 NGEN=<max> KV_SLOTS=<n>` and `bufsize=0` pipes. `SERVE_BATCH=1` selects `run_serve_mux` in the engine (colibri.c:6657). Startup handshake: gateway blocks in `read_engine_turn(stdout, READY, …)` (:779) waiting for the `b"\x01\x01READY\x01\x01\n"` sentinel (:40), which the engine prints after model load (colibri.c:5422) followed by a `STAT 0 …` line. A dedicated daemon thread `colibri-stdout` (`_dispatch_stdout`, :815-886) then owns stdout forever.

**Submit framing (gateway → engine)** — `Engine.generate` (:888-949):
```
SUBMIT <id> <slot> <bytes> <max_tokens> <temp:.8g> <top_p:.8g> [<gbytes>]\n
<prompt bytes><grammar bytes>\n
```
built at :916-918, written atomically under `write_lock` (:920-924). NUL bytes rejected client-side (:893-897) because the engine's `getline`-based framing can't carry them. Engine-side parse: `coli_submit_parse` (decode_batch.h:26-45) — strict field count with trailing-char rejection, `bytes ≤ 16 MiB`, `gbytes ≤ 1 MiB`, temp ∈ [0,2], top_p ∈ (0,1]. Payload read with `fread(bytes)` + mandatory trailing `\n` (colibri.c:5324-5337, `BAD_FRAME` if the delimiter is wrong).

**Engine mux loop** — `run_serve_mux` (colibri.c:5397-5520): non-blocking `select()` on stdin when slots are active, blocking when idle (:5439-5459; Windows uses `PeekNamedPipe` per #139/#195). `mux_submit` (:5303-5394) tokenizes with one-token headroom to *detect* overflow instead of silently truncating (the #401 fix, :5353-5370), prefix-matches against the slot's history (:5372), prefills only new tokens serially (:5379-5380), then the slot joins **continuous batched decode**: each active slot contributes one `DecodeRow` per forward through `step_decode_batch` (:5463-5514). Grammar-forced greedy slots temporarily leave the batch for a single-sequence draft-verify forward (:5469-5496).

**Token streaming (engine → gateway)** — per token, `mux_data` prints a length-prefixed frame (colibri.c:5263-5267):
```
DATA <id> <n>\n<n bytes UTF-8>\n
```
Turn end: `mux_done` (:5269-5298) emits telemetry (`HWINFO`,`TIERS`,`EMAP`,`HITS` via telemetry.h:78-166), a `PROF` phase-timing line (:5284-5288), then
```
DONE <id> STAT <emitted> <tok/s> <hit%> <rss_gb> <prompt_tokens> <length_limited>
```
(:5289-5291), and persists the slot's KV to disk (`kv_disk_append`, :5292). Errors: `ERROR <id> <CODE>` with `BAD_REQUEST/BAD_FRAME/SLOT_BUSY/DUPLICATE_ID/EMPTY_PROMPT/CONTEXT_EXCEEDED/NOT_FOUND/CANCELLED` (:5311-5346, 5363-5370).

**Gateway dispatch** — `_dispatch_stdout` (openai_server.py:815-886) reads line kinds: `DATA` (size-checked ≤ 65536, exact read + `\n` terminator, :825-836), `DONE` → stats dict (:837-843), telemetry lines cached on the Engine object (`HWINFO`:844, `EMAP`:851, `HITS`:853, `PROF` into a 120-turn deque:856-869, `TIERS`:870), `ERROR` → typed exception (`CONTEXT_EXCEEDED` → HTTP 400 `context_length_exceeded`, :74-89). Frames route to per-request `queue.Queue` objects keyed by request id (:767, :915). The requesting HTTP thread consumes the queue in `Engine.generate`'s event loop (:930-949), incrementally UTF-8-decoding across frame boundaries (:898-903, tested at test_openai_server.py:392).

**SSE back to the client** — `generation` (:1213-1394): headers `text/event-stream`, `X-Accel-Buffering: no` (:1273-1280). A keepalive thread emits a `{"reasoning_content":"."}` delta if no write for 10 s (cold prefill can be minutes; :1282-1319), all writes serialized by `ka_lock` including the final `[DONE]` (:1388-1393, "#B9"). With tools, streamed content holds back `len("<tool_call>")-1` chars so a marker split across chunks is still suppressed (:1333-1357); authoritative tool_calls are parsed from the full reply post-hoc (`parse_tool_calls`, :312-371, with unclosed-tail recovery and opt-in `COLI_TOOL_SALVAGE` de-mangling). Anthropic streaming mirrors this with first-class `ping`/`message_start`/`content_block_*` events (:1517-1615).

**Cancellation** — client disconnect detected by `MSG_PEEK` recv (:1396-1404); `generate` sends `CANCEL <id>\n` after the next DATA event (:936-940); engine acknowledges with `ERROR <id> CANCELLED` after persisting the slot's KV (colibri.c:5308-5318), which the gateway maps to `ClientCancelled` (:946-947).

### 1.2 KV slot prefix cache across stateless HTTP turns

- The client (or dashboard) passes `cache_slot` (validated :1228-1233); the scheduler treats a pinned slot as a distinct capacity lane (:137-153) and `SUBMIT` carries it as field 2.
- Engine-side, each slot has a `ServeCtx {kv, hist, len}` (colibri.c:5229). On submit, the newly tokenized prompt is compared token-by-token against `hist`: `prefix = longest common prefix` (:5372). If the new prompt diverges, history is truncated to the divergence point (`kv_disk_truncate`, :5373-5374) and only `nt - len` new tokens are prefilled (:5375-5380). Since OpenAI clients resend the whole conversation each turn and the rendered template is deterministic, the entire prior conversation is a prefix hit — visible in the `[API] KV slot %d prefix %d/%d token, prefill %d` stderr line (:5377).
- KV additionally persists to disk per slot (`<SNAP>/.coli_kv.<slot>`, :5239-5241) with crash-consistent append-only records (nrec written last, :5221-5227), so a restarted engine resumes warm.

### 1.3 Live expert/tier telemetry to the dashboard

Engine emits at startup and after every turn (colibri.c:5272-5276, 5423-5425): `TIERS` (expert counts per VRAM/RAM/disk tier, telemetry.h:98-113), `EMAP` (one byte per expert: `(tier<<6)|log2-heat`, hex-encoded, :115-146), `HITS` (1 bit per expert routed since last emit, :148-166), `HWINFO` (:78-96), `PROF` (per-turn phase wall times, colibri.c:5284-5288). The gateway dispatcher caches the latest of each and bumps `hits_seq`/`profile_seq` (openai_server.py:851-873). HTTP surface:

- `GET /health` → scheduler snapshot + `tiers` + `hwinfo` (auth-gated internals, :1121-1134)
- `GET /experts` → `{rows, cols, map, hits, seq}` (auth-gated, :1135-1143)
- `GET /profile` → `{seq, turns[≤120]}` (:1144-1149)

Frontend: `App.tsx` polls `/health` every 5 s when connected (App.tsx:113-127) and renders tier bars/scheduler counters (:265-292). `Brain.tsx` polls `/experts` every 1.5 s (Brain.tsx:55-82); a `seq` change loads the `hits` bitmap into a per-expert pulse array, rendered as a canvas grid where tier sets hue, heat sets luminance, and routing hits flash white with 0.94/frame decay (:96-122). `Profiling.tsx` polls `/profile` every 2 s and derives `other_s = wall - Σphases` for stacked phase bars (Profiling.tsx:17-21, 82-95). `coli web` serves the built SPA from the same port via `serve_static` (openai_server.py:1083-1114).

---

## 2. PROTOCOL ROBUSTNESS

**Engine crash mid-turn.** `_dispatch_stdout` sees stdout EOF → `RuntimeError("colibri engine exited unexpectedly")` (:818-820) → `_fail_pending` pushes the error to every in-flight request queue (:797-802) and records `dispatcher_error` (:885). In-flight non-streaming requests get 500; mid-stream requests get a JSON error blob written **into the open SSE stream** (do_POST's generic handler :1198-1205 sends `send_json` on a connection whose headers were already sent — the client sees a malformed tail, not a proper SSE error event). Every subsequent request fails at :909-912 (`dispatcher stopped`). **There is no engine restart path and `/health` still returns `{"status":"ok"}`** (:1125 — the status literal is unconditional), so orchestrators never see the outage. Contrast `coli chat`, which at least diagnoses OOM-kills (`engine_diag`, coli:418-444).

**Client disconnect mid-stream.** Detected two ways: SSE write `OSError` sets `connected=False` (:1309-1310), and the `cancelled` callback (`lambda: not connected`) sent to `generate`. But the cancel check runs **only when a DATA event arrives** (:933-940). During prefill — minutes on this hardware, zero DATA frames — a vanished client keeps its scheduler slot and the engine keeps prefilling to completion. The keepalive pump does detect the dead socket within ~10 s (its `event()` write fails), but nothing converts that into a CANCEL until the first token. Also the keepalive thread in `generation` is only stopped after `generate` returns *normally* (:1379); if `generate` raises (engine death), `ka_stop` is never set — the daemon thread exits via its `connected` check only if the socket also died, otherwise it dribbles keepalives until the handler thread closes the socket.

**Concurrency > KV_SLOTS.** Well-handled: capacity = kv_slots, per-slot fair queueing (:137-153), bounded queue (429 `queue_full`), bounded wait (429 `queue_timeout`), and the engine independently rejects `SLOT_BUSY`/`DUPLICATE_ID` (colibri.c:5342-5347) as a second line of defense. Note the gateway allows `kv_slots ≤ 16` (:1639-1640) while the mux engine accepts up to 512 (colibri.c:5406) — headroom exists but the two bounds are silently different, and the legacy `run_serve` caps at 16 (:5550).

**Malformed JSON / bodies.** Clean: Content-Length validation, 4 MiB cap, JSON+dict check (:1063-1076), extensive per-field 400s in `generation_options` (:613-725). Unsupported params are refused loudly, not ignored (:659-666).

**Oversized prompts.** Two layers: gateway `MAX_BODY` 4 MiB, engine `bytes ≤ 16 MiB` (decode_batch.h:39), and semantic overflow via the #401 fix — `tok_encode` at `maxctx-1` detects truncation and returns `ERROR <id> CONTEXT_EXCEEDED <used> <limit>` (colibri.c:5353-5370), mapped to an actionable 400 `context_length_exceeded` (openai_server.py:74-89). This is the correct design (client can compact and retry).

**Protocol fragility (the core weakness).** The dispatcher treats **any unrecognized stdout line as fatal**: `raise RuntimeError(f"invalid engine response: …")` (:881-882), killing every in-flight and future request. This directly contradicts serve_protocol.md:28-30 ("servers **must ignore line kinds they do not recognize**; that is the protocol's forward-compatibility rule"). One stray `printf` to stdout in the engine — a debug line, a new telemetry kind, a library writing to fd 1 — permanently bricks the gateway until manual restart. The engine is careful (`prof_report` goes to stderr, colibri.c:5296, 5586), but the invariant is enforced nowhere and one violation is unrecoverable. Also: stdout and the framing share a single stream with no escaping — a `DATA` payload is length-prefixed (safe), but header lines are `split()`-parsed with no version/magic, and `int(fields[2])` on a corrupt size raises the same fatal error.

**Docs/code drift.** serve_protocol.md documents `TOPK`, `PERF`, `ENTROPY`, `GPUS`, `REPIN` stdout lines (:57-71, 82-87) and a `data: {"colibri": {...}}` SSE frame plus `/experts` fields `gpus/entropy/repin` (:92-99). None exist: REPIN goes to stderr (colibri.c:5214), the dispatcher has no PERF/ENTROPY/GPUS/TOPK arms (:825-882 — and per :881 they would be *fatal* if the engine ever emitted them), `/experts` returns only `{rows,cols,map,hits,seq}` (:1136). The doc says "the code wins and this file needs a PR" — it does.

---

## 3. IMPROVEMENTS (evidenced)

**I1 — Make the dispatcher ignore unknown line kinds (protocol forward-compat).**
`openai_server.py:881-882`. Change the final `else: raise` to log-and-continue (optionally rate-limited), keeping strict checks only for `DATA` payload framing. *Impact:* converts a whole-server permanent outage into a log line; makes the documented forward-compat rule true, unblocking new telemetry kinds (PERF/ENTROPY/GPUS are already spec'd in serve_protocol.md:82-87 and would today crash the gateway). *Risk:* low — genuinely corrupt framing inside a DATA payload still fails via size/terminator checks (:829-832); a skipped garbage line between frames was previously fatal anyway.

**I2 — Engine crash: supervise/restart and degrade `/health`.**
`openai_server.py:815-886` (fatal dispatcher), :1125 (`{"status":"ok"}` unconditional), no respawn anywhere in `serve` (:1626-1669). Add: (a) `/health` reports `"degraded"` when `dispatcher_error` is set or `process.poll() is not None`; (b) optional bounded auto-restart of the Engine (the KV disk persistence at colibri.c:5292 means a restart resumes warm — the machinery already exists). *Impact:* a serve deployment currently requires human intervention after any engine SIGKILL (OOM-killer is a documented, recurring event — coli:421-435, #305, #403); orchestrators and `coli web`'s opener poll `/health` and are actively lied to. *Risk:* medium — restart must not double-spawn (guard with a lock) and must fail-fast after N attempts to avoid OOM crash loops.

**I3 — Check client-disconnect during prefill, not just per-DATA.**
`openai_server.py:930-940`: cancellation is evaluated only inside the `kind == "data"` arm. Replace the blocking `events.get()` with `events.get(timeout=1)` and poll `cancelled()` on timeout, sending CANCEL even before the first token. *Impact:* on this workload prefill dominates (minutes); today a client that gives up during prefill wastes the full prefill **and holds a scheduler slot**, delaying every queued request behind it — the exact head-of-line cost the #B2 scheduler work tried to eliminate. The engine already handles early CANCEL (colibri.c:5308-5318). *Risk:* low; one caveat — the engine's mux loop only reads stdin between forwards, so CANCEL during a long single prefill call is still deferred engine-side (a second improvement would check `g_intr`/pending input inside prefill).

**I4 — Fix mid-stream error handling: emit an SSE error event, stop the keepalive thread on exception.**
`openai_server.py:1198-1205` sends a JSON 500 body after SSE headers are already out (invalid HTTP; clients see garbage after their event stream), and the `except` path never sets `ka_stop` (:1291, :1379), leaking a pinging thread per failed stream. Wrap the streaming section in `try/finally: ka_stop.set()` and, when headers are sent, emit `data: {"error": …}\n\n` + close instead of `send_json`. *Impact:* correct client-visible failure semantics for every engine error during streaming (the most common failure mode); removes thread leak. *Risk:* low, localized.

**I5 — Auth-gate `/profile` like `/health` internals and `/experts`.**
`openai_server.py:1144-1149` returns full per-turn timings (prompt/completion token counts, wall time, phase breakdown of the last 120 turns) with no `_is_authed()` check, while the same class of telemetry on `/health` (:1126-1132) and `/experts` (:1138) was deliberately gated per #SEC-8. Cross-origin *reads* are blocked by CORS for browsers, but any local process or non-browser client on a `COLI_ALLOW_INSECURE_BIND` deployment can watch usage patterns. One-line fix mirroring :1138; also note `test_profile_reports_recent_turns_without_auth` (test_openai_server.py:489) currently enshrines the inconsistent behavior. *Risk:* trivial; dashboard already sends the key (web/src/lib/api.ts `getProfile` passes headers).

**I6 — Deduplicate `run_serve` vs `run_serve_mux` (and align their limits).**
The legacy path (colibri.c:5522-5695) re-implements everything the mux path has: tokenize-with-headroom, prefix match + truncate (:5620-5633 vs :5372-5376), byte-counted prompt framing (`\x02PROMPT` :5597-5612 vs `SUBMIT` :5320-5341), Windows binary-mode setup (:5533-5537 vs :5412-5419), KV persistence and STAT emission — with subtle divergences: legacy caps `nb ≤ 16 MiB` but *silently swallows* an over-long prompt via `tok_encode(…, maxctx-8-g_draft)` with no CONTEXT_EXCEEDED signal (:5621 — the exact #401 bug class the mux path fixed at :5353-5370), tolerates a missing trailing `\n` (`ungetc`, :5607) where mux hard-fails (`BAD_FRAME`, :5334), and enforces `KV_SLOTS ≤ 16` vs mux's 512 (:5550 vs :5406). Extract shared helpers (prefix-match+truncate; framed-payload read; overflow-detecting encode) or migrate `coli chat` onto the mux protocol (serve_protocol.md:107 already says "new integrations should use the mux protocol"). *Impact:* removes a live bug-parity gap (legacy still silently truncates) and halves the maintenance surface of the most delicate code in the engine. *Risk:* medium — `coli chat`'s stream_turn/`\x02MORE` UX depends on legacy sentinels; do it behind the existing protocol tests plus a new legacy-path test (none exists today).

**I7 — Add an integration test for the real engine mux protocol and for dispatcher resilience.**
Coverage today: excellent Python unit tests (test_openai_server.py: scheduler FIFO/fairness/timeout :141-223, dispatcher interleave/UTF-8 split/corruption :295-424, HTTP surface :450-660) and a genuine subprocess e2e — but against a **mock** engine (test_openai_tools_e2e.py:1-50). The C side tests `coli_submit_parse` arithmetic only (test_decode_batch.c:35-40). Nothing anywhere executes `run_serve_mux`'s actual loop: CANCEL mid-decode, SLOT_BUSY, DUPLICATE_ID, BAD_FRAME recovery, prefix-truncate correctness, EOF drain, or the select/PeekNamedPipe gate — the code with the richest platform history (#139, #195). A tiny-model or stub-weights harness (the repo already has `tok_o200k_tiny.json` fixtures) driving the real binary would close this. Also missing: a test that the dispatcher survives (post-I1) an unknown telemetry line, and any test of `serve_static`'s SPA fallback beyond traversal (:597). *Impact:* the byte protocol is the least-typed, most platform-sensitive seam in the system and currently ships on manual testing. *Risk:* CI runtime; keep it Linux-only initially.

**I8 — CI blind spots: desktop shell and gateway-vs-engine protocol drift.**
`.github/workflows/ci.yml` covers engine build+tests, CUDA/HIP/MSVC syntax/builds, web build+vitest, Python unittests (:9-165); check.yml adds 3-OS `make check`. Not covered: (a) `desktop/src-tauri` — never built anywhere (`cargo build`/`tauri build` absent from all four workflows), so the Tauri 2 config/capabilities (tauri.conf.json, capabilities/default.json) can rot unnoticed; (b) no job cross-checks serve_protocol.md against the emitting printfs (the drift documented in §2 is already real); (c) release.yml builds engines but never runs even a smoke `SUBMIT/DONE` round-trip against the artifact it ships. Add a `cargo check` job for src-tauri and fold the I7 protocol smoke test into release. *Impact:* prevents shipping a desktop shell that no longer compiles and catches protocol drift pre-merge. *Risk:* none beyond CI minutes.

**Security posture (context for I5):** overall strong for a local server — fail-closed non-loopback bind without a key (#SEC-6, :1641-1650), Host-header rebinding guard incl. preflight (#SEC-7, :1043-1061, :1163-1174), constant-time key compare (:1032-1036), allowlisted CORS with proper `Vary: Origin` (:1011-1022), traversal-safe static serving via `resolve()`+`relative_to` (:1090-1098, tested :597), untrusted-tokenizer hardening in tok.h (:115-136) and depth-bounded JSON parsing (json.h:29-35). Remaining nits beyond I5: static files are served **before** auth (:1150-1152) by design ("same trust level as /health") — fine for the SPA but worth documenting; `serve_static` reads whole files into memory per request with no caching headers (:1106-1113), a minor DoS/perf nit; and the slowloris `timeout = 30` (:991) does not bound total body-dribble time, only per-socket-op.