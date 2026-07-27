"""Oracolo DeepSeek-V4 (stadi b+c+e del V4_DESIGN: layer 1 e'
heavily_compressed_attention (rate 4, seq 32 -> 8 righe compresse vive);
il primo MoE e' hash_moe con tid2eid random non banale).
Novita' vs le famiglie gia' validate:
  - mHC hyper-connections: residuo a hc_mult=4 stream paralleli, mixing
    pre/post/comb con proiezione Sinkhorn (20 iter) per sublayer + hyper head;
  - attention MQA V4: 1 KV head (K==V) head_dim 512-class, q_a->q_norm->q_b->
    RMS per-head SENZA peso, rope PARZIALE (ultimi rope_head_dim) con formula
    interleaved-duplicated (cos/sin repeat_interleave su rotate_half), rope
    INVERSO sull'output, sinks stile OSS, proiezione O raggruppata (o_a per
    gruppo + o_b), finestra scorrevole;
  - expert: clamp(swiglu_limit) + silu (niente bias, niente +1 — diverso da OSS);
  - router: sqrtsoftplus score, selezione score+bias, pesi normalizzati — il
    percorso FASE A esistente (score_func=2, noaux) e' GIA' questo.

    SNAP=./dsv4_tiny REF=ref_dsv4.json TF=1 ./colibri 64 16 16

--l0check: pinna in numpy (a) il modulo HyperConnection (Sinkhorn incluso)
e (b) l'attention del layer 0 chiamata DIRETTAMENTE, prima di scrivere C.

Eseguire da c/:  python3 tools/make_dsv4_oracle.py --l0check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from transformers import DeepseekV4Config, DeepseekV4ForCausalLM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--l0check", action="store_true")
args = ap.parse_args()

torch.manual_seed(3131)

HC = 4
cfg = DeepseekV4Config(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=48,          # expert-I E shared-I in V4
    moe_intermediate_size=48,      # inutilizzato da HF ma tenuto coerente
    num_hidden_layers=3,
    num_attention_heads=4,
    head_dim=32,
    rope_head_dim=8,
    q_lora_rank=32,
    o_groups=2,
    o_lora_rank=16,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    sliding_window=8,
    layer_types=["sliding_attention", "heavily_compressed_attention", "sliding_attention"],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
    mlp_layer_types=["hash_moe", "moe", "moe"],   # stadio e: primo layer hash
    hc_mult=HC,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    swiglu_limit=10.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    max_position_embeddings=4096,
    rope_parameters={
        "main":     {"rope_type": "default", "rope_theta": 10000.0,  "partial_rotary_factor": 0.25},
        "compress": {"rope_type": "default", "rope_theta": 160000.0, "partial_rotary_factor": 0.25},
    },
)
cfg._attn_implementation = "eager"

model = DeepseekV4ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.06)
    # hc scale/base a default-init empty: dare valori sensati e non banali
    for n, p in model.named_parameters():
        if ".attn_hc." in n or ".ffn_hc." in n or "hc_head" in n:
            if p.dim() == 1 and p.numel() <= 24:
                p.copy_(torch.linspace(-0.5, 0.5, p.numel()))
            elif p.dim() == 2:
                p.normal_(0, 0.02)
    for li, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp.gate, "e_score_correction_bias"):
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.3, 0.3, cfg.n_routed_experts))
        if hasattr(layer.mlp.gate, "tid2eid"):
            g = torch.Generator().manual_seed(li)   # tabella hash NON banale
            layer.mlp.gate.tid2eid.copy_(
                torch.randint(0, cfg.n_routed_experts, layer.mlp.gate.tid2eid.shape,
                              generator=g))

prompt = [5, 40, 210, 66, 9, 150, 88, 17, 231, 104, 44, 172]
ids = torch.tensor([prompt])

if args.l0check:
    sd = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    eps = cfg.rms_norm_eps
    hceps = cfg.hc_eps

    # ---- L0a: HyperConnection (pre/post/comb + Sinkhorn) ----
    hcmod = model.model.layers[0].attn_hc
    x4 = torch.randn(1, 3, HC, cfg.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        post_t, comb_t, coll_t = hcmod(x4)
    xf = x4.numpy().astype(np.float64).reshape(1, 3, HC * cfg.hidden_size)
    flat = xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + eps)
    mixes = flat @ sd["model.layers.0.attn_hc.fn"].T
    pre_w, post_w, comb_w = mixes[..., :HC], mixes[..., HC:2*HC], mixes[..., 2*HC:]
    base = sd["model.layers.0.attn_hc.base"]; scale = sd["model.layers.0.attn_hc.scale"]
    sig = lambda z: 1/(1+np.exp(-z))
    pre = sig(pre_w*scale[0]+base[:HC])+hceps
    post = 2*sig(post_w*scale[1]+base[HC:2*HC])
    cl = comb_w.reshape(1,3,HC,HC)*scale[2]+base[2*HC:].reshape(HC,HC)
    e = np.exp(cl-cl.max(-1,keepdims=True)); comb = e/e.sum(-1,keepdims=True)+hceps
    comb = comb/(comb.sum(-2,keepdims=True)+hceps)
    for _ in range(cfg.hc_sinkhorn_iters-1):
        comb = comb/(comb.sum(-1,keepdims=True)+hceps)
        comb = comb/(comb.sum(-2,keepdims=True)+hceps)
    coll = (pre[...,None]*x4.numpy()).sum(2)
    d1 = max(np.abs(post-post_t.numpy()).max(), np.abs(comb-comb_t.numpy()).max(),
             np.abs(coll-coll_t.numpy()).max())
    print(f"L0a HyperConnection (sigmoid gates + Sinkhorn 20): max|diff| = {d1:.3e}")

    # ---- L0b: attention layer 0, hook durante il forward REALE (la maschera
    # sliding+causale la costruisce il modello; chiamare il modulo con mask=None
    # spegnerebbe anche la causalita') ----
    S = len(prompt)
    cap = {}
    hnd = model.model.layers[0].self_attn.register_forward_hook(
        lambda m_, i_, o_: cap.__setitem__("o", o_[0].detach().numpy()[0]))
    hin = model.model.layers[0].self_attn.register_forward_pre_hook(
        lambda m_, a_, k_: cap.__setitem__("x", (a_[0] if a_ else k_["hidden_states"]).detach().numpy()[0]),
        with_kwargs=True)
    with torch.no_grad():
        model.model(ids)
    hnd.remove(); hin.remove()
    ref_o = cap["o"]; xn = cap["x"].astype(np.float64)
    H, hd, rd = cfg.num_attention_heads, cfg.head_dim, 8   # rd = head_dim*0.25
    L0 = "model.layers.0.self_attn."
    def rmsw(v, w): return v/np.sqrt((v*v).mean(-1,keepdims=True)+eps)*w
    def rmsu(v):    return v/np.sqrt((v*v).mean(-1,keepdims=True)+eps)
    qr = rmsw(xn @ sd[L0+"q_a_proj.weight"].T, sd[L0+"q_a_norm.weight"])
    q = (qr @ sd[L0+"q_b_proj.weight"].T).reshape(S, H, hd)
    q = rmsu(q)
    kv = rmsw(xn @ sd[L0+"kv_proj.weight"].T, sd[L0+"kv_norm.weight"]).reshape(S, 1, hd)
    inv = 1.0/(10000.0**(np.arange(0, rd, 2)/rd))          # main theta, rd dims
    ang = np.arange(S)[:,None]*inv[None,:]                 # [S, rd/2]
    cos = np.repeat(np.cos(ang), 2, axis=-1)[:,None,:]     # repeat_interleave
    sin = np.repeat(np.sin(ang), 2, axis=-1)[:,None,:]
    def rope_part(z, sgn=1.0):
        # V4 rotate_half e' la variante a COPPIE ADIACENTI (stack(-x2,x1).flatten):
        # y[2j] = x[2j]*c_j - x[2j+1]*s_j ; y[2j+1] = x[2j+1]*c_j + x[2j]*s_j
        nope, rope = z[..., :-rd], z[..., -rd:]
        rot = np.empty_like(rope)
        rot[..., 0::2] = -rope[..., 1::2]
        rot[..., 1::2] =  rope[..., 0::2]
        return np.concatenate([nope, rope*cos + rot*(sgn*sin)], -1)
    q = rope_part(q); kv = rope_part(kv)
    sinks = sd[L0+"sinks"]
    win = cfg.sliding_window
    out = np.zeros((S, H, hd))
    for s in range(S):
        lo = max(0, s-win+1)
        for h in range(H):
            sc = np.array([q[s,h] @ kv[t,0]/np.sqrt(hd) for t in range(lo, s+1)])
            both = np.concatenate([sc,[sinks[h]]]); both -= both.max()
            p = np.exp(both); p /= p.sum()
            out[s,h] = sum(p[i]*kv[lo+i,0] for i in range(len(sc)))
    out = rope_part(out, sgn=-1.0)                         # rope INVERSO sull'output
    g = cfg.o_groups; olr = cfg.o_lora_rank
    grouped = out.reshape(S, g, H*hd//g)
    wo_a = sd[L0+"o_a_proj.weight"].reshape(g, olr, H*hd//g)
    oa = np.einsum("sgd,grd->sgr", grouped, wo_a).reshape(S, g*olr)
    o = oa @ sd[L0+"o_b_proj.weight"].T
    d2 = np.abs(o - ref_o).max()
    print(f"L0b V4 attention (MQA K=V, partial rope, inverse-out-rope, grouped O, sinks, sliding): max|diff| = {d2:.3e}")
    if d1 > 1e-4 or d2 > 1e-4:
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

# ---- unfuse experts nel layout per-expert 2-D del motore ----
sd = model.state_dict()
new_sd = {}
for name, ten in sd.items():
    if name.endswith(".mlp.experts.gate_up_proj"):
        prefix = name[:-len(".mlp.experts.gate_up_proj")]
        E_, twoI, D_ = ten.shape; I_ = twoI//2
        for e in range(E_):
            new_sd[f"{prefix}.mlp.experts.{e}.gate_proj.weight"] = ten[e, :I_, :].contiguous()
            new_sd[f"{prefix}.mlp.experts.{e}.up_proj.weight"]   = ten[e, I_:, :].contiguous()
    elif name.endswith(".mlp.experts.down_proj"):
        prefix = name[:-len(".mlp.experts.down_proj")]
        E_, D_, I_ = ten.shape
        for e in range(E_):
            new_sd[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = ten[e].contiguous()
    else:
        new_sd[name] = ten

Path("dsv4_tiny").mkdir(parents=True, exist_ok=True)
from safetensors.torch import save_file                # noqa: E402
save_file({k: v.contiguous() for k, v in new_sd.items()}, "dsv4_tiny/model.safetensors")
json.dump(cfg.to_dict(), open("dsv4_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
          open("ref_dsv4.json", "w"))
print("saved: dsv4_tiny/ (weights + config) and ref_dsv4.json")
