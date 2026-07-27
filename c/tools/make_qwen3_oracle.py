"""Oracolo Qwen3-MoE: modello MINUSCOLO Qwen3MoeForCausalLM a pesi random.
La famiglia porta le TRE novita' che mancano al motore (MULTIMODEL_PLAN §3):
  - GQA classica (num_key_value_heads < num_attention_heads, K/V per head)
    al posto della MLA compressa;
  - RoPE split-half (Llama-style rotate_half, NON interleaved DeepSeek);
  - q_norm/k_norm RMS per-head PRIMA della RoPE, router softmax senza bias,
    nessun shared expert, nessun layer denso (decoder_sparse_step=1).

Salva pesi+config in c/qwen3_tiny/ e ref_qwen3.json (formato ref_glm.json):

    SNAP=./qwen3_tiny REF=ref_qwen3.json TF=1 ./colibri 64 16 16

--l0check: prima di salvare, rifa' il forward del primo blocco attention in
numpy puro (la matematica che il C dovra' replicare: proiezioni, per-head
RMSNorm, rotate_half RoPE, mappa GQA kv_head = h // (H/KVH), softmax causale)
e confronta con gli hidden states di transformers — pinna l'ordine esatto
delle operazioni con attribuzione float-level, cosi' ogni divergenza C si
diagnostica contro un riferimento a vettori, non a token.

Eseguire da c/:  python3 tools/make_qwen3_oracle.py --l0check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_fp8_emit import unfuse_experts               # noqa: E402

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--l0check", action="store_true",
                help="verifica numpy vs transformers del primo blocco attention")
args = ap.parse_args()

torch.manual_seed(7777)

cfg = Qwen3MoeConfig(
    vocab_size=256,
    hidden_size=96,
    intermediate_size=64,          # inutilizzato con mlp_only_layers=[] e sparse_step=1
    moe_intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,         # <- GQA: 2 gruppi di 2 teste
    head_dim=16,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    rms_norm_eps=1e-6,
    attention_bias=False,
    tie_word_embeddings=False,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = Qwen3MoeForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)

prompt = [11, 45, 202, 7, 133, 88, 61, 240, 3, 179, 25, 96]
ids = torch.tensor([prompt])

if args.l0check:
    # ---- L0: numpy replay of layer-0 attention, float-level ----------------
    sd = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    H, KVH, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    D = cfg.hidden_size
    x = sd["model.embed_tokens.weight"][prompt]                 # [S, D]
    S = x.shape[0]

    def rms(v, w, eps=cfg.rms_norm_eps):
        return v / np.sqrt((v * v).mean(-1, keepdims=True) + eps) * w

    h0 = rms(x, sd["model.layers.0.input_layernorm.weight"])
    q = h0 @ sd["model.layers.0.self_attn.q_proj.weight"].T     # [S, H*hd]
    k = h0 @ sd["model.layers.0.self_attn.k_proj.weight"].T     # [S, KVH*hd]
    v = h0 @ sd["model.layers.0.self_attn.v_proj.weight"].T
    q = q.reshape(S, H, hd); k = k.reshape(S, KVH, hd); v = v.reshape(S, KVH, hd)
    # per-head RMSNorm PRIMA della RoPE (Qwen3: q_norm/k_norm su head_dim)
    q = rms(q, sd["model.layers.0.self_attn.q_norm.weight"])
    k = rms(k, sd["model.layers.0.self_attn.k_norm.weight"])
    # RoPE split-half (rotate_half): freq su hd/2, cos/sin duplicati
    inv = 1.0 / (10000.0 ** (np.arange(0, hd, 2) / hd))         # [hd/2]
    t = np.arange(S)[:, None] * inv[None, :]                    # [S, hd/2]
    cos = np.cos(t); sin = np.sin(t)
    cos2 = np.concatenate([cos, cos], -1)[:, None, :]           # [S,1,hd]
    sin2 = np.concatenate([sin, sin], -1)[:, None, :]

    def rope(z):
        zl, zr = z[..., :hd//2], z[..., hd//2:]
        rot = np.concatenate([-zr, zl], -1)
        return z * cos2 + rot * sin2

    q, k = rope(q), rope(k)
    out = np.zeros((S, H, hd))
    for h in range(H):
        kv = h // (H // KVH)                                     # mappa GQA
        sc = q[:, h, :] @ k[:, kv, :].T / np.sqrt(hd)            # [S,S]
        sc += np.triu(np.full((S, S), -np.inf), 1)               # causale
        w = np.exp(sc - sc.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        out[:, h, :] = w @ v[:, kv, :]
    attn = out.reshape(S, H * hd) @ sd["model.layers.0.self_attn.o_proj.weight"].T
    got = x + attn                                               # residuo

    with torch.no_grad():
        hs = model.model(ids, output_hidden_states=True).hidden_states
    # hidden_states[1] = dopo il blocco 0 COMPLETO (attn + moe); confrontare
    # solo l'attn richiede l'output pre-MoE: rifallo con un hook.
    ref = {}
    def hook(mod, inp, outp):
        ref["attn"] = outp[0].detach().numpy()[0]
    hnd = model.model.layers[0].self_attn.register_forward_hook(hook)
    with torch.no_grad():
        model.model(ids)
    hnd.remove()
    ref_res = x + ref["attn"]
    diff = np.abs(got - ref_res).max()
    print(f"L0 numpy vs transformers (layer-0 attention + residual): max|diff| = {diff:.3e}")
    if diff > 1e-4:
        sys.exit("L0 CHECK FAILED: the numpy math does not match transformers - "
                 "do not write C against an unpinned reference")
    print("L0 math pinned: split-half RoPE, per-head RMS q/k norm pre-RoPE, "
          "GQA map kv=h//(H/KVH), causal softmax\n")

print("=== state_dict tensors (names used by the C loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
print("\nprompt:", prompt)
print("full  :", full)

with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

sd = model.state_dict()
unfuse_experts(sd)

Path("qwen3_tiny").mkdir(parents=True, exist_ok=True)
from safetensors.torch import save_file                # noqa: E402
save_file({k: v.contiguous() for k, v in sd.items()}, "qwen3_tiny/model.safetensors")
json.dump(cfg.to_dict(), open("qwen3_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
          open("ref_qwen3.json", "w"))
print("saved: qwen3_tiny/ (weights + config) and ref_qwen3.json")
