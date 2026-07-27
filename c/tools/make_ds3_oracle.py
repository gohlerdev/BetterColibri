"""Oracolo DeepSeek-V3: modello MINUSCOLO DeepseekV3ForCausalLM a pesi random,
con n_group>1 (group-limited routing) — la feature che distingue DSv3 da GLM
nel motore. Salva pesi+config in c/ds3_tiny/ e il riferimento greedy +
teacher-forcing in c/ref_ds3.json, stesso formato di ref_glm.json:

    SNAP=./ds3_tiny REF=ref_ds3.json TF=1 ./colibri 64 16 16   (atteso N/N)

E' la stessa disciplina di make_glm_oracle.py applicata alla famiglia
DeepSeek: valida il forward C (MLA + router sigmoid/noaux_tc + route_group_mask
+ shared expert) contro transformers, layer DSA assente (has_dsa=0 per assenza
dei pesi indexer — il probe del motore deve gestirlo).

Eseguire da c/:  python3 tools/make_ds3_oracle.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_fp8_emit import unfuse_experts               # noqa: E402

from transformers import DeepseekV3Config, DeepseekV3ForCausalLM  # noqa: E402

torch.manual_seed(4321)

cfg = DeepseekV3Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=64,           # MLP densa (primo layer)
    moe_intermediate_size=32,
    num_hidden_layers=4,            # 1 denso + 3 MoE
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=16,
    num_experts_per_tok=4,
    n_shared_experts=1,
    n_group=4,                      # <- la novita' vs GLM: group-limited routing
    topk_group=2,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=24,
    qk_rope_head_dim=8,
    v_head_dim=32,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    tie_word_embeddings=False,
    rms_norm_eps=1e-5,
    attention_bias=False,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = DeepseekV3ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    # bias di correzione del router: distinti e abbastanza grandi da rendere il
    # group-mask decisivo (gruppi con bias alto vincono la selezione top-2-sum)
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.4, 0.4, cfg.n_routed_experts))

print("=== state_dict tensors (names used by the C loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [7, 21, 133, 45, 8, 201, 90, 4, 250, 66, 17, 111]
ids = torch.tensor([prompt])
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

Path("ds3_tiny").mkdir(parents=True, exist_ok=True)
from safetensors.torch import save_file                # noqa: E402
save_file({k: v.contiguous() for k, v in sd.items()}, "ds3_tiny/model.safetensors")
json.dump(cfg.to_dict(), open("ds3_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
          open("ref_ds3.json", "w"))
print("saved: ds3_tiny/ (weights + config) and ref_ds3.json")
