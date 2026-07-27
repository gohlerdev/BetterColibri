"""Oracolo GPT-OSS: modello MINUSCOLO GptOssForCausalLM a pesi random.
Le novita' vs Qwen3 (stessa base GQA):
  - attention SINKS: un logit extra per testa nella softmax, scartato dopo;
  - sliding window sui layer 'sliding_attention' (alternati);
  - YaRN RoPE (inv_freq rampato + attention_scaling su cos/sin);
  - bias lineari su q/k/v/o e sul router;
  - expert: gate_up INTERLEAVED [E,D,2I] con bias, clamp(gate<=7,|up|<=7),
    glu=gate*sigmoid(1.702*gate), act=(up+1)*glu, down con bias;
  - router: top-k sui logit grezzi, POI softmax sui k selezionati.

Salva pesi+config in c/gptoss_tiny/ e ref_gptoss.json (formato ref_glm.json):

    SNAP=./gptoss_tiny REF=ref_gptoss.json TF=1 ./colibri 64 16 16

--l0check: replay numpy dell'attention del layer 0 (sliding) e dell'expert
MLP contro transformers PRIMA di salvare (attribuzione float-level).

NOTA CONTAINER: transformers tiene gli expert FUSI [E,D,2I]/[E,I,D]; per il
motore C li spacchiamo in per-expert gate/up/down 2-D [I,D]/[D,I] row-major
(orientazione matmul_qt out=x@W.T), de-interleavando gate/up. E' la stessa
forma che un converter reale produrra' dai checkpoint MXFP4.

Eseguire da c/:  python3 tools/make_gptoss_oracle.py --l0check
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from transformers import GptOssConfig, GptOssForCausalLM  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--l0check", action="store_true")
args = ap.parse_args()

torch.manual_seed(1212)

cfg = GptOssConfig(
    vocab_size=256,
    hidden_size=96,
    intermediate_size=32,          # anche expert-I (gpt_oss non ha moe_inter)
    num_hidden_layers=4,           # sliding/full/sliding/full
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_local_experts=8,
    num_experts_per_tok=2,
    sliding_window=8,              # < seq per esercitare davvero la finestra
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = GptOssForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.06)

prompt = [9, 33, 180, 51, 6, 141, 77, 12, 220, 98, 30, 164]
ids = torch.tensor([prompt])

if args.l0check:
    sd = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    H, KVH, hd, D = 4, 2, 16, 96
    x = sd["model.embed_tokens.weight"][prompt]
    S = len(prompt)

    def rms(v, w, eps=cfg.rms_norm_eps):
        return v / np.sqrt((v * v).mean(-1, keepdims=True) + eps) * w

    h0 = rms(x, sd["model.layers.0.input_layernorm.weight"])

    def proj(name, h):
        return (h @ sd[f"model.layers.0.self_attn.{name}.weight"].T
                + sd[f"model.layers.0.self_attn.{name}.bias"])

    q = proj("q_proj", h0).reshape(S, H, hd)
    k = proj("k_proj", h0).reshape(S, KVH, hd)
    v = proj("v_proj", h0).reshape(S, KVH, hd)
    inv, scal = ROPE_INIT_FUNCTIONS["yarn"](cfg, None)
    inv = inv.numpy().astype(np.float64)
    t = np.arange(S)[:, None] * inv[None, :]
    cos = np.cos(t) * scal; sin = np.sin(t) * scal
    c2 = np.concatenate([cos, cos], -1)[:, None, :]
    s2 = np.concatenate([sin, sin], -1)[:, None, :]

    def rope(z):
        zl, zr = z[..., :hd//2], z[..., hd//2:]
        return z * c2 + np.concatenate([-zr, zl], -1) * s2

    q, k = rope(q), rope(k)
    sinks = sd["model.layers.0.self_attn.sinks"]
    win = cfg.sliding_window
    out = np.zeros((S, H, hd))
    for s in range(S):
        lo = max(0, s - win + 1)
        for h in range(H):
            kv = h // (H // KVH)
            sc = np.array([q[s, h] @ k[t2, kv] / math.sqrt(hd)
                           for t2 in range(lo, s + 1)])
            both = np.concatenate([sc, [sinks[h]]]); both -= both.max()
            p = np.exp(both); p /= p.sum()
            out[s, h] = sum(p[i] * v[lo + i, kv] for i in range(len(sc)))
    attn = (out.reshape(S, H * hd) @ sd["model.layers.0.self_attn.o_proj.weight"].T
            + sd["model.layers.0.self_attn.o_proj.bias"])
    got = x + attn
    ref = {}
    hnd = model.model.layers[0].self_attn.register_forward_hook(
        lambda m_, i_, o_: ref.__setitem__("a", o_[0].detach().numpy()[0]))
    with torch.no_grad():
        model.model(ids)
    hnd.remove()
    diff = np.abs(got - (x + ref["a"])).max()
    print(f"L0 attention (sinks+sliding+yarn): max|diff| = {diff:.3e}")

    hs = rms(got, sd["model.layers.0.post_attention_layernorm.weight"])
    W = sd["model.layers.0.mlp.router.weight"]; B = sd["model.layers.0.mlp.router.bias"]
    lg = hs @ W.T + B
    E, K = cfg.num_local_experts, cfg.num_experts_per_tok
    mo = np.zeros_like(hs)
    for s in range(S):
        idx = np.argsort(-lg[s])[:K]
        tv = lg[s][idx]; p = np.exp(tv - tv.max()); p /= p.sum()
        for kk, e in enumerate(idx):
            gu = (hs[s] @ sd["model.layers.0.mlp.experts.gate_up_proj"][e]
                  + sd["model.layers.0.mlp.experts.gate_up_proj_bias"][e])
            gate, up = gu[::2], gu[1::2]
            gate = np.minimum(gate, 7.0); up = np.clip(up, -7.0, 7.0)
            act = (up + 1) * (gate / (1 + np.exp(-gate * 1.702)))
            o = (act @ sd["model.layers.0.mlp.experts.down_proj"][e]
                 + sd["model.layers.0.mlp.experts.down_proj_bias"][e])
            mo[s] += p[kk] * o
    ref2 = {}
    h2 = model.model.layers[0].mlp.register_forward_hook(
        lambda m_, i_, o_: ref2.__setitem__("m", o_[0].detach().numpy()))
    with torch.no_grad():
        model.model(ids)
    h2.remove()
    diff2 = np.abs(mo - ref2["m"].reshape(S, -1)).max()
    print(f"L0 expert MLP (clamp-glu + bias, topk-softmax): max|diff| = {diff2:.3e}")
    if diff > 1e-4 or diff2 > 1e-4:
        sys.exit("L0 CHECK FAILED - do not write C against an unpinned reference")
    print("L0 math pinned\n")

with torch.no_grad():
    outg = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = outg[0].tolist()
print("prompt:", prompt)
print("full  :", full)
with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

# ---- unfuse per il container del motore: per-expert 2-D row-major ----------
sd = model.state_dict()
new_sd = {}
for name, ten in sd.items():
    if name.endswith(".mlp.experts.gate_up_proj"):
        prefix = name[:-len(".mlp.experts.gate_up_proj")]
        E_, D_, twoI = ten.shape; I_ = twoI // 2
        for e in range(E_):
            # colonne pari=gate, dispari=up; il motore vuole [I,D] (out=x@W.T)
            new_sd[f"{prefix}.mlp.experts.{e}.gate_proj.weight"] = ten[e, :, 0::2].T.contiguous()
            new_sd[f"{prefix}.mlp.experts.{e}.up_proj.weight"]   = ten[e, :, 1::2].T.contiguous()
    elif name.endswith(".mlp.experts.down_proj"):
        prefix = name[:-len(".mlp.experts.down_proj")]
        E_, I_, D_ = ten.shape
        for e in range(E_):
            new_sd[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = ten[e].T.contiguous()
    else:
        new_sd[name] = ten

print("\n=== state_dict tensors (names used by the C loader) ===")
for n, p in new_sd.items():
    if ".layers.0." in n or ".layers." not in n:
        print(f"  {n:60s} {tuple(p.shape)}")

Path("gptoss_tiny").mkdir(parents=True, exist_ok=True)
from safetensors.torch import save_file                # noqa: E402
save_file({k: v.contiguous() for k, v in new_sd.items()}, "gptoss_tiny/model.safetensors")
json.dump(cfg.to_dict(), open("gptoss_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
          open("ref_gptoss.json", "w"))
print("saved: gptoss_tiny/ (weights + config) and ref_gptoss.json")
