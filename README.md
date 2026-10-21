# Custom Weight-Only Quantizer for Qwen3-0.6B

Building a from-scratch weight quantizer in raw PyTorch to shrink Qwen3-0.6B and choosing the best configuration by measured size, accuracy and speed.

**Quantized model:** [Hugging Face link](https://huggingface.co/<your-username>/<your-model-name>)

## What this project does

Post-training, weight-only quantization of [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B). All quantization math is written with plain PyTorch tensor operations: min/max, quantization range, scale, zero point, rounding, clamping, dequantization and INT4 bit-packing.

**No quantization libraries are used** (no `bitsandbytes`, `torch.ao.quantization`, `optimum`, GPTQ or AWQ). `torch` does the tensor math, `transformers` loads the model and tokenizer, `safetensors` stores the weights, and `datasets` downloads the evaluation text.

## Constraint

- **Memory:** run on a 16 GB laptop (about 8 GB usable)
- **Size:** at least 60% smaller than FP32 (2,384 MB)
- **Quality:** perplexity at most 10% above the 16-bit baseline

## Configurations tested

| ID | Bits | Scales | Mode |
|:--:|:----:|:------:|:----:|
| A | 8 | per tensor | symmetric |
| B | 8 | per channel | symmetric |
| C | 4 | per channel | asymmetric |

Only the 196 projection matrices (q, k, v, o, gate, up, down in 28 blocks) are quantized. The tied embedding/output matrix and the norm layers stay in 16-bit.

## Results

| Configuration | Size (MB) | Reduction vs FP32 | Reduction vs BF16 | Weight MAE | Perplexity (change) | Speed (tok/s) | Constraints met |
|---|---:|---:|---:|---:|---:|---:|:--:|
| Baseline (BF16) | 1,192.1 | 50.0% | 0.0% | 0.00000 | 2.687 | 19.2 | - |
| **A: INT8 per-tensor** | **751.7** | **68.5%** | **36.9%** | 0.00086 | 2.681 (-0.2%) | 15.7 | **Yes** |
| B: INT8 per-channel | 752.4 | 68.4% | 36.9% | 0.00021 | 2.689 (+0.1%) | 8.1 | Yes |
| C: INT4 per-channel | 532.5 | 77.7% | 55.3% | 0.00339 | 3.588 (+33.6%) | 6.5 | No (quality) |

**Selected: A (INT8, per-tensor, symmetric).** It meets the size and quality requirements, and its size is within 1% of B, so the tie goes to the lower perplexity. INT4 (config C) fails the quality bound. See the report for the full discussion.

## Notes and limitations

- Perplexity in this run was measured on a short built-in text because WikiText-2 could not be loaded. Absolute values are not comparable with published figures; relative changes are indicative only.
- Quantized models are slower than the baseline, because weights are rebuilt on every forward pass. The saving is in memory and storage, not speed.
- No configuration reaches 60% against the BF16 file, because the embedding/output matrix stays 16-bit. The 60% target is measured against FP32.

## Repository contents

| File | Description |
|---|---|
| `qwen3_custom_quantizer_final.ipynb` | Notebook with the quantizer, evaluation, and results |
| `quantizer_core.py` | Quantizer module (written by the notebook; needed to load the model) |
| `results.csv` | Full results table |
| `report.pdf` | 1-2 page report |

## How to run

1. Open the notebook in Google Colab and select a GPU runtime.
2. Run all cells from top to bottom. The notebook writes `quantizer_core.py`, evaluates the three configurations, picks the winner, and saves it to `final_quantized_model/`.

## How to load the quantized model

`quantizer_core.py` must be in the same folder as the weights (it is included in the model folder).

```python
from quantizer_core import load_quantized_model
from transformers import AutoTokenizer

model = load_quantized_model("final_quantized_model", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("final_quantized_model")

prompt = "Explain why the sky is blue in three simple sentences."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## References

- Course material: iCog Labs, "Shrinking AI models" training
- Model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
