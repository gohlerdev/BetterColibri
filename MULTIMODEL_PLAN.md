# Multi-Model Gap Analysis — toward a universal MoE streaming runtime

Real HF configs (fetched 2026-07-26) diffed against every assumption the engine
makes in `load_cfg`/`model_init`/`attention_rows`/`moe()`. This is the ladder
from "GLM engine" to "run ANY giant MoE from disk".

## The four targets, ranked by effort

### 1. Kimi K2 (1T params) — NEARLY FREE ★ headline unlock
`DeepseekV3ForCausalLM`, and critically **`n_group=1`** — it passes the engine's
existing router check as-is. Same MLA shape the engine already computes:

| param | GLM-5.2 | Kimi K2 | engine impact |
|---|---|---|---|
| kv_lora_rank / qk_rope | 512 / 64 | 512 / 64 | **identical** — absorb path unchanged |
| q_lora | 2048 | 1536 | dim variable, already parametrized |
| qk_nope / v_head | 192 / 256 | 128 / 128 | within CKR bounds, no code change |
| experts / topk | 256 / 8 | **384** / 8 | within bounds (≤4096) |
| router | sigmoid + noaux_tc, n_group=1 | sigmoid + noaux_tc, n_group=1 | **identical** |
| shared experts / first dense | 1 / 3 | 1 / 1 | parametrized |
| layers / hidden | 93 / 6144 | 61 / 7168 | parametrized |
| DSA indexer | yes | no | already optional (`has_dsa` probe) |

**Real work:** converter coverage (K2 ships fp8 → our int4 container; weight
names are deepseek-style `model.layers.N.mlp.experts.M.*` — same scheme),
tokenizer (K2 uses a tiktoken variant; `tok.h` already has an o200k path to
extend), gateway chat template. Expert size ~22 MB at int4, ~515 GB total —
the same disk class as GLM. The Metal/CUDA fast paths gate on exact GLM dims
and fall back gracefully, so K2 runs the portable paths on day one.

### 2. DeepSeek-V3/R1 (671B) — SMALL: one routing feature
Identical MLA/expert shape to K2 except **`n_group=8, topk_group=4`**
(group-limited routing). The engine currently refuses `n_group!=1` at
colibri.c:934. Adding grouped top-k selection to FASE A is ~30 lines
(select top `topk_group` groups by max-score, then top-k within them —
the reference impl is public). Everything else identical to K2.

### 3. Qwen3-235B-A22B — MEDIUM: the GQA unlock
`qwen3_moe`: **GQA** (64 q-heads / 4 kv-heads, head_dim 128), softmax router
(no scoring_func field = softmax + norm_topk), **no shared expert**, **no dense
layers** (all 94 layers MoE), 128 experts / top-8. Needs:
- A GQA attention path with a standard paged KV cache (kv-cache per token:
  2·4·128 = 1024 floats — bigger than MLA's 576, still small; quantize to int8
  for parity)
- Softmax router arm in FASE A (simpler than the sigmoid+bias path)
- n_shared=0 handling in FASE E (skip shared-expert phase)

### 4. GPT-OSS-120B — MEDIUM+: GQA plus attention extras
`gpt_oss`: GQA (64/8, head_dim 64), **alternating sliding-window (128) /
full-attention layers** (`layer_types[]`), **attention bias + sinks**,
128 experts / top-4, all layers MoE, MXFP4-native experts. Rides the same
GQA path as Qwen3 plus: window masking per layer type, sink token handling,
and an MXFP4→int4/E8 conversion arm. Smallest model of the set (120B) but
the hottest name; its 5.1B-active / 128-expert shape is *ideal* for
streaming.

## The architecture descriptor this implies

```
ModelArch {
  attn:    MLA{q_lora,kv_lora,nope,rope} | GQA{n_kv,head_dim,window[],sinks,bias}
  router:  {score: sigmoid+bias | softmax, topk, n_group/topk_group,
            norm_topk, scale, n_shared}
  layers:  {n, dense_prefix | moe_step, dims}
  vocab/tokenizer: hf-json | tiktoken(o200k|kimi)
  speculation: mtp? (probe) | ngram | grammar   ← all engine-side already
}
```

The tier engine (slabs, PIPE, PILOT, LRU+pin+heat, dual-SSD, .coli_usage,
.coli_kv) needs **zero changes** for any of these — routing predictability is
what PILOT needs, and every one of these models has a router to look ahead
with.

## Build sequence

1. **K2 converter + dim generalization + tokenizer** → "1T on a desktop"
   (the headline; oracle-validate vs transformers like GLM was)
2. **DeepSeek-V3 group routing** (+30 lines) → two families, credibility
   with the model everyone knows
3. **ModelArch descriptor refactor** — mechanical once 1-2 show where the
   seams are
4. **GQA + paged KV** → Qwen3 + GPT-OSS land together → four families,
   "universal" is earned

## DeepSeek-V4 series + DSpark (added 2026-07-27, from real config + inference/model.py)

**What it is.** DeepSeek-V4: **Pro (1.6T params / 49B active)** and **Flash
(284B / 13B active)**, both 1M context, experts in **FP4** with FP8 elsewhere.
`DeepSeek-V4-Pro-DSpark` is NOT a new model: it is V4-Pro plus **DSpark**, a
speculative-decoding module (Markov head, `dspark_markov_rank=512`, special
attention on layers 58-60, block size 5 — see github.com/deepseek-ai/DeepSpec).

**Architecture delta (from `inference/model.py`, 961 lines, read directly):**

| Component | V4 reality | Engine gap |
|---|---|---|
| MoE | 384 experts, top-6, SwiGLU (`swiglu_limit=10` clamp), FP4 weights | streaming machinery applies as-is; needs an FP4 (e2m1) dequant kernel — same nibble-packing family as the existing int4 path |
| Router | `sqrtsoftplus` scoring (`sqrt(softplus(x))`) + bias for selection only, weights renormalized, `route_scale=2.5` | one new scoring arm in FASE A (structure identical to sigmoid+noaux_tc) |
| **Hash routing** | first `num_hash_layers=3` layers route by a `tid2eid[vocab, topk]` table — expert choice known from the TOKEN ID alone | **an advantage unique to a streaming engine**: those layers' experts can be prefetched before the forward even starts; no router matmul at all |
| Attention | **hybrid CSA/HCA, not MLA**: MQA-ish 1 KV head × `head_dim=512`, learned KV compression (`Compressor`: gated pooling over 4-token windows with overlap, or heavy 128× ratio per `compress_ratios[]`), `o_groups=16`/`o_lora_rank=1024` output projection, YaRN to 1M | **new forward pass** — this is the bulk of the work |
| Residual | **mHC hyper-connections**: `hc_mult=4` parallel residual streams mixed by Sinkhorn-projected matrices per layer | changes the layer skeleton; second-largest work item |
| Indexer | still present (`index_topk=1024`, 64 heads × 128 dim) | engine's DSA indexer is this exact lineage |
| MTP | `num_nextn_predict_layers=1` | engine's MTP framework applies |
| DSpark | draft via Markov head + `DSparkAttention` on 3 target layers | engine's spec_decode framework is the right home; new draft math; optional |

**Sizing.** V4-Flash at FP4 experts ≈ **~150 GB-class container** — smaller than
GLM's 372 GB; streams from a single NVMe. V4-Pro ≈ 1.6T → ~800 GB-class at FP4:
the "1.6-TRILLION params on consumer hardware" headline, strictly bigger than
K2's.

**Ladder position.** V4 is the biggest target on the board because CSA/HCA + mHC
are a new engine core, not a port. Sequence stays: V3/R1 group routing → K2 →
**V4-Flash** (small download, full V4 architecture) → V4-Pro → DSpark module.
The FP4 kernel and `sqrtsoftplus` arm can land early (locally testable); hash
routing should land with them (it is pure engine-side and the prefetch win is
free). CSA/HCA + mHC get their own design doc before any code.



## Positioning

> Run a 1-trillion-parameter model on the PC you already own.
> One engine. Any MoE. Any disk. It learns your workload and gets faster.

Differentiators no one else ships: learned tiering (.coli_usage), router
lookahead prefetch (71.6% recall), dual-SSD striping, grammar-accelerated
JSON mode, token-exact oracle validation, and the Brain/Atlas live
visualization of every expert firing.

---

## Status (2026-07-27, implementation session)

| Item | Commit | State |
|------|--------|-------|
| DeepSeek-V3 group-limited routing (n_group/topk_group) | 160dbfa | **done** — route_group_mask in FASE A + PILOT, 2,203-case brute-force-verified, GLM path provably no-op |
| Multi-model router scoring (sigmoid/softmax/sqrtsoftplus) | 7a33077 | **done** — route_score at both router sites, reference-math-verified incl. ±88 stability |
| FP4 (e2m1) pack + matmul kernel | 357bc1f | **done** — OCP MX table semantics, AVX2 PSHUFB path, 475-case oracle; engine wiring as fmt=7 awaits the V4 container format |
| Multi-model config acceptance test | 24e75a8 | **done** — real K2/DSv3/GLM/V4 dims through load_cfg in TEST_BINS; audit: zero hardcoded GLM dims on portable paths (Metal fast-path gates fall back by design) |
| DSv4 hash-routing groundwork | 269c59b | **done** — num_hash_layers config + hardened tid2eid lookup; FASE A wiring + tokenize-time prefetch land with the container work |

**Engine-side result:** DeepSeek-V3/R1 routing is fully implemented; Kimi K2
needs no engine changes (confirmed by the acceptance test) — both now wait on
converter + tokenizer + oracle validation, which need weights and disk.
V4's router/kernel groundwork is in; CSA/HCA attention + mHC remain the
V4-Flash design-doc milestone.

**Still needed for end-to-end new-model runs (weights/disk-bound):**
converter coverage (K2 fp8→int4, V4 fp4 passthrough into fmt=7 slabs +
tid2eid shipping), K2 tiktoken tokenizer, teacher-forcing oracle validation
per model — the GLM discipline applied to each.
