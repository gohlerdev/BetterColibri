# DeepSeek-V4 reference source (vendored)

`model.py` + `kernel.py` from `deepseek-ai/DeepSeek-V4-Pro-DSpark`
(`inference/`, sha 7c09739, fetched 2026-07-27). MIT-licensed by DeepSeek.

Vendored because V4_DESIGN.md's implementation plan requires replicating
this math EXACTLY (hc_split_sinkhorn's softmax+eps then 20 row/col
normalizations; Compressor's overlap_transform; act_quant/fp4_act_quant
QAT simulation) and the L0 oracle ladder needs the reference importable
offline. Do not edit; diff upstream before any V4 milestone.
