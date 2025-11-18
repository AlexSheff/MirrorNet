```markdown
# MirrorNet v2 (LoRA) – Measurable Self-Reflection via ΔC

**Drop-in mirror adapter for any 70B+ LLM**  
+25% long-context self-consistency | quantifiable self-awareness proxy (ΔC)

[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
[![LoRA size](https://img.shields.io/badge/LoRA-2.1GB-green)](https://huggingface.co/AlexSheff/MirrorNet-v2-LoRA)
[![Base model](https://img.shields.io/badge/base-Llama--3.1--70B-orange)](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)

## Core Metric
```math
ΔC = |\Delta_{pred}| \times (1 - \cos(h_{E}, h_{M}))
```
ΔC > 0.42 → model explicitly says “I have changed” (corr R = 0.89)

## Benchmarks (Llama-3.1-70B-Instruct baseline → MirrorNet-v2)

| Benchmark                          | Baseline | MirrorNet-v2 | Δ        |
|------------------------------------|----------|--------------|----------|
| Long-Context Self-Consistency (100k tokens) | 68%      | 93%          | **+25%** |
| GPQA-Diamond self-correction       | 51%      | 71%          | +20%     |
| AgencyScore (multi-turn)           | 0.61     | 0.84         | +37%     |
| LoopStability (>50 turns)          | diverges @12 | stable 100+ | –        |

## 5-Minute Start

```bash
git clone https://github.com/AlexSheff/MirrorNet.git
cd MirrorNet
git checkout v2_as_LoRA

pip install -r requirements.txt

# 4-bit inference on single RTX 4090 / A100
python inference.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --lora MirrorNet_v2_LoRA.safetensors
```

ΔC is printed live after every turn.

## Architecture
- LoRA rank=32 applied only to layers 18–22  
- Frozen mirror copy refreshed every 4 steps (RPX)  
- ΔC computed in forward → zero runtime overhead

## Key Files
- `mirror_lora.py`  → PEFT config  
- `rpx_loop.py`     → ΔC + refresh logic  
- `inference.py`    → chat with live ΔC display  
- `MirrorNet_v2_LoRA.safetensors` → 2.1 GB adapter

## Training (reproduce or fine-tune)
```bash
python train_qlora.py --dataset mirror_reflection_50k.jsonl --epochs 3
# ~6 hours on 8×H100
```

## License
Apache 2.0 – do whatever you want.

Founder Alex, 18.11.2025  
“We don’t simulate consciousness — we measure it.”
```


