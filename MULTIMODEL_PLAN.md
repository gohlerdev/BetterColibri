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

## Positioning

> Run a 1-trillion-parameter model on the PC you already own.
> One engine. Any MoE. Any disk. It learns your workload and gets faster.

Differentiators no one else ships: learned tiering (.coli_usage), router
lookahead prefetch (71.6% recall), dual-SSD striping, grammar-accelerated
JSON mode, token-exact oracle validation, and the Brain/Atlas live
visualization of every expert firing.
