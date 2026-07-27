# DeepSeek-V4 (Flash/Pro) engine design — CSA/HCA attention + mHC residual streams

*Written 2026-07-27 (overnight session), from the reference implementation
`deepseek-ai/DeepSeek-V4-Pro-DSpark` `inference/model.py` (961 lines, read in
full; cached at the analysis session). This is the design-doc gate
MULTIMODEL_PLAN prescribes before any V4 attention/residual code lands.
Prereqs already in the engine: `sqrtsoftplus` router arm (7a33077), FP4 e2m1
kernel + fmt=7 container (357bc1f, 508d945), hash-routing groundwork
(269c59b), swiglu-limit-style clamps proven in the GPT-OSS family (802ba51),
sinks + sliding window proven in the GPT-OSS family (802ba51), YaRN table
(802ba51), MTP framework, DSA indexer lineage.*

## Why V4 is a new engine core, not a port

Every family the engine now validates (GLM MLA, DSv3 MLA, Qwen3/OSS GQA)
keeps the two invariants the whole engine is built on:

1. **One KV row per token** — `Lc[t]`/`Rc[t]` indexed by absolute position;
2. **One residual stream** — `x += attention; x += moe`.

V4 breaks both. Its attention (call it CSA/HCA: compressed sparse attention
with hyper-connections) keeps a *sliding window* of full-resolution KV rows
**plus** a *compressed* KV sequence at 1/ratio density, and its residual is
**four parallel streams** mixed by learned Sinkhorn-projected matrices around
every sublayer. Neither maps onto the current `KVState`/`layer_forward_rows`
shape. That is exactly why the plan gates this work behind a design doc.

## The V4 attention, precisely (from model.py)

Per layer, `compress_ratios[layer]` ∈ {0, 4, 128} selects one of three modes:

- **ratio=0** — pure sliding-window MQA. KV cache is `window_size=128` rows.
  No YaRN (base theta), no compressor, no indexer.
- **ratio=4** — window + *overlapping* gated-pool compression (windows of 4,
  doubled state for overlap) + a dedicated **Indexer** (its own compressor
  with Hadamard rotation + FP4-simulated scoring) choosing `index_topk`
  compressed positions.
- **ratio=128** — window + heavy non-overlap compression, positional
  compressed-index selection (`get_compress_topk_idxs`), no indexer.

Common structure (all modes):

```
q  = wq_b(q_norm(wq_a(x)))            # low-rank Q, 64 heads x head_dim 512
q *= rsqrt(mean(q^2))                 # PER-HEAD rms (no weight!) after wq_b
rope(q[..., -64:])                    # rope only on the last rope_head_dim
kv = kv_norm(wkv(x))                  # ONE kv "head" of head_dim 512 (MQA)
rope(kv[..., -64:])
attn = sparse_attn(q, kv_set, attn_sink, topk_idxs, scale)   # sinks like OSS
rope(attn[..., -64:], inverse=True)   # NOTE: inverse rope on the OUTPUT
o = einsum over o_groups of wo_a      # grouped low-rank O: 16 groups x o_lora 1024
x = wo_b(o)
```

Key deltas vs everything we have:

| Piece | Nearest engine analog | Gap |
|---|---|---|
| MQA 1-kv-head × 512 | GQA with KVH=1 | trivially expressible in `attention_gqa` shapes |
| per-head weightless q-RMS after wq_b | qwen3 q_norm (weighted, pre-rope) | one flag |
| partial rope (last 64 of 512) | MLA does exactly this (`qk_rope`) | reuse |
| **inverse rope on attention output** | none | small new helper (conjugate rotation) |
| attention sinks | GPT-OSS (802ba51) | done |
| sliding window | GPT-OSS (802ba51) | done, but V4 windows use *ring-buffer indexing* (`start_pos % win`) |
| **compressed KV sequence** | none | the real work item #1 |
| grouped low-rank O (wo_a/wo_b) | MLA's o-proj is dense | new but mechanical |
| topk_idxs sparse attention | DSA selection (`dsa_sel`) | same *shape* of problem: attend over an explicit index list |

### Work item 1: the Compressor

The compressor is a *streaming* module with per-slot state:

- prefill: pool every `ratio` tokens (overlapping doubles the state width for
  ratio=4), softmax-weighted by `wgate(x)+ape` scores, RMS-norm, rope at the
  compressed position, quantize (fp8-sim for main, fp4-sim+Hadamard for the
  indexer's private compressor);
- decode: accumulate into `kv_state`/`score_state` ring slots, emit one
  compressed row every `ratio` steps.

Engine mapping: this is a **third KV plane**. Proposal:

```
KVState += float **Cc;      /* compressed rows [max_t/ratio, head_dim] per layer */
          int   *cc_len;    /* rows emitted so far (decode incremental state) */
          float *cc_state;  /* [coff*ratio, coff*head_dim] pooling ring per layer */
```

The pooling state is tiny (`8×1024` floats at ratio=4) and lives beside
`kv_start[]`. KVSAVE/prefix-truncate must snapshot `cc_*` alongside — the
disk KV format grows a versioned section (kv_persist.h already versions).

### Work item 2: sparse_attn over (window ∪ compressed) index sets

`topk_idxs` = ring-window absolute indices ++ compressed indices (offset past
the window region). The engine's DSA path already does attend-over-index-list
(`dsa_sel[]`); the new part is that indices address **two planes with
different strides** (window rows in `Lc`, compressed rows in `Cc`). Proposal:
keep the reference's own trick — one *logical* cache `[win + max_t/ratio]`
per layer, window plane first — so `topk_idxs` stays a flat index and the
score loop stays one loop. `Lc` at fixed 128+N/ratio rows replaces the
current per-token growth; the ring indexing (`pos % win`) replaces
`kv_start`-based windows (OSS T12 clamps st0; V4 *overwrites* rows — the
cache is genuinely bounded, which is a *benefit* on 1M contexts).

### Work item 3: mHC hyper-connections

Per block, the hidden state is `[hc_mult=4, dim]`. Around each sublayer:

```
pre, post, comb = sinkhorn_split(F.linear(flatten(x4), hc_fn) * rsqrt(...))  # data-dependent!
x1 = Σ_h pre[h] · x4[h]              # 4 -> 1 (weighted sum)
y  = sublayer(norm(x1))
x4 = post⊗y + comb·x4                # 1 -> 4 (+4×4 mixing of the residual streams)
```

`hc_split_sinkhorn` = split the `(2+4)*4=24` mixing logits into pre(4),
post(4), comb(4×4); comb goes through `hc_sinkhorn_iters=20` Sinkhorn
normalization (row/col stochastic); scales/bases are learned scalars.

Engine impact: `layer_forward_rows` currently owns `x[S,D]`. mHC makes the
inter-layer state `x4[S,4,D]` while every sublayer still consumes `[S,D]`.
Proposal that keeps EVERY existing sublayer function signature intact:

- `layers_forward` allocates `x4` (4× activations: at S=1 decode this is
  16 KB·hc — irrelevant; at prefill 4× the activation buffer, still MBs);
- a `hc_pre_rows()`/`hc_post_rows()` pair brackets the existing
  attention/moe calls inside `layer_forward_rows` when `c->hc_mult>1`;
- embed writes `x4[h]=x` broadcast; the head reads `hc_head()` (sigmoid
  gates, no Sinkhorn) → `[S,D]`.
- Sinkhorn: 20 iterations on a 4×4 matrix per (position, sublayer) — ~cheap
  (640 flops), scalar code is fine, no kernel work.

**Numerics discipline:** the reference runs hc math in fp32 with bf16
carriers. The C engine is fp32 end-to-end — pin with the L0 ladder
(numpy Sinkhorn vs reference `hc_split_sinkhorn`, then per-layer activation
comparison) exactly like Qwen3/OSS (both landed 32/32 first-run off that
method).

### Work item 4: hash routing (already 80% built)

`Gate.forward` with `input_ids`: layers `< n_hash_layers` take indices from
`tid2eid[token]` (269c59b's `hash_route_select`), gate weights = scores at
those indices. Remaining: FASE A wiring (skip router matmul on hash layers)
+ tokenize-time prefetch (the streaming engine's unique win: expert sets for
hash layers are known before the forward starts). No blocker; land with the
container work.

## Container / converter notes

- Experts: FP4 (e2m1) → fmt=7 slabs (kernel + container tag already landed;
  oracle-validated 475 cases). `swiglu_limit=10` clamp: reuse the OSS
  `oss_expert_act` shape with V4 semantics (silu, not 1.702-sigmoid — check
  `Expert.forward` exactly at implementation time).
- Attention weights: FP8 → int8/int4 via the existing converter paths;
  `wo_a` is grouped — store per-group 2-D tensors like unfused experts.
- New resident tensors per layer: `attn_sink[64]`, compressor
  `ape/wkv/wgate/norm`, hc `{attn,ffn}×{fn,base,scale}`, indexer set (V4's
  indexer weights match the DSA loader's shape expectations closely).
- `tid2eid` per hash layer: `[vocab, topk]` i32 → ship as a sidecar tensor.

## Sizing and sequencing (unchanged from MULTIMODEL_PLAN, now with effort)

1. **V4-Flash first** (284B, ~150 GB at FP4: single-NVMe class).
2. Order of implementation *inside* V4 (each L0-pinned before C):
   a. mHC (work item 3) — self-contained, testable on a 2-layer tiny model
      with ratio=0 attention only;
   b. sliding-window MQA + inverse-rope + grouped-O (mode ratio=0) — gets a
      full tiny-V4 oracle running TF end-to-end while compression is stubbed
      (a tiny model with all `compress_ratios=0` is a VALID config);
   c. Compressor plane + positional compressed indices (ratio=128);
   d. Indexer + overlap compression (ratio=4);
   e. hash-layer FASE A + tokenize-time prefetch;
   f. DSpark (optional, spec-decode framework home).
   Each step has a tiny-model TF oracle gate: `compress_ratios` being
   per-layer means every intermediate stage is a runnable model.
3. KV persistence version bump lands with (c) — first stage that adds state.

## Risks

- **Ring-buffer KV vs prefix reuse:** V4's bounded window overwrites rows, so
  serve-mode prefix reuse semantics change for V4 (only window+compressed
  state matters — which is exactly why 1M context fits). The mux protocol
  needs no changes; `kv_disk` format does (versioned).
- **fp32-vs-QAT drift:** reference simulates fp8/fp4 activations *in the
  forward* (act_quant calls). The tiny-model oracle must run the reference
  WITH those quant sims on (they ship in model.py) so the C engine matches
  the real model's numerics, not an idealized one. The oracle generator must
  therefore vendor `kernel.py`'s act_quant/fp4_act_quant math.
- **Sinkhorn iteration count:** fixed 20; convergence is not data-independent
  — do NOT early-exit, replicate exactly.
- **DSpark** touches three target layers with a second attention over main
  hidden states; keep it strictly optional (absence = plain V4-Pro/Flash).

## What is NOT needed

- No new tokenizer work (V4 ships tokenizer.json, cl100k-family regex — the
  existing GLM path loads it; verified on the shipped file 2026-07-27).
- No router work beyond hash wiring (sqrtsoftplus arm landed + tested).
- No new quant kernel (fmt=7 landed; GPU kernels are the existing deferred
  bucket).
