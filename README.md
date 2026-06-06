<h1 align="center">
   <p>Asymmetric DenseLoRA</p>
</h1>

This fork extends DenseLoRA with a **non-square dense matrix M (asymmetric, r1×r2)** and three activation options (**GeLU / SiLU / GoLU**).
## What this fork adds
- **Asymmetric DenseLoRA**: the encoder maps `hidden → r1`, the dense matrix `M: r1 → r2`, the decoder `r2 → out`. Square when `r1 == r2`, expanding when `r2 > r1`.
- **Selectable dense activation**: `gelu` (default, as in the paper), `silu`, or `golu` (Gompertz Linear Unit, pure autograd — no CUDA kernel build required).
- **Data subsampling**: `--data_fraction 0.4` uses 40% of the data (shuffle `seed=42`, reproducible).
- **Robust bitsandbytes detection**: `is_bnb_available()` actually performs an `import` (not just `find_spec`), so a broken bnb does not break the entire `import peft` chain.

`scaling = lora_alpha / max(r1, r2)`. With the rule `alpha = 2 × largest rank`, scaling = 2.0.

## Experiment table (matching the Colab notebook)
| Exp | Activation | r1×r2 | Shape of M |
|--|--|--|--|
| 1 | GeLU | 32×32 | square (baseline) |
| 2 | GeLU | 16×32 | expansion |
| 3 | GeLU | 8×16  | aggressive expansion |
| 4 | SiLU | 32×32 | square |
| 5 | SiLU | 16×32 | expansion |
| 6 | SiLU | 8×16  | aggressive expansion |
| 7 | GoLU | 32×32 | square |
| 8 | GoLU | 16×32 | expansion |
| 9 | GoLU | 8×16  | aggressive expansion |

`--adapter_name lora` runs DenseLoRA. The encoder/decoder are shared across layers, while M is unique per layer.

## Quick Start — Colab (recommended path)
Open `notebook/DenseLoRA_TinyLlama_Colab.ipynb` in Google Colab (`Runtime → Change runtime type → T4 GPU`), then follow the cells in order:

1. **Check GPU** — `!nvidia-smi`.
2. **Mount Drive** — working folder at `MyDrive/DenseLoRA`.
3. **Point to the repo folder** — place `commonsense_reasoning/` (containing `commonsense_170k.json`, `finetune.py`, and this fork's `peft/`) in Drive, e.g. `MyDrive/DenseLoRA/commonsense_reasoning`. Set `COPY_TO_LOCAL=True` if you want to copy it to Colab's local disk for faster file reads.
4. **Install dependencies** — pin `numpy<2`, `transformers==4.36.0`, `accelerate==0.25.0`, `datasets==2.15.0`, etc. **Uninstall the pip version of `peft`** so that `finetune.py` uses the LOCAL `peft` (`peft/src/`) that contains DenseLoRA, then reinstall the latest `bitsandbytes`.
5. **Choose an experiment** — change `EXP` (1–9), `DATA_PATH` (your 20%/40% dataset file), `DATA_TAG`, and `MICRO_BATCH`. The notebook automatically computes `r1`, `r2`, `lora_r = max(r1,r2)`, and `alpha = 2×max(r1,r2)`, then creates a unique `OUTPUT_DIR` per experiment.
6. **Fine-tuning** — runs `finetune.py` with `--auto_resume True`. If the runtime disconnects, re-run the mount/setup cells then run the training cell again; it resumes from the last checkpoint.
7. **Evaluation** (optional) — `commonsense_evaluate.py` reads `r1/r2/activation` from `adapter_config.json`, so the shape of M matches automatically.
8. **Trade-off summary** — the last cell scans all `denselora_tinyllama_exp*` folders, computes trainable params + per-task accuracy, and saves `ringkasan_eksperimen.csv` to Drive.

## Quick Start — local CLI
Install dependencies:
```bash
pip install -r requirements.txt
```

### Training (finetune.py)
Example equivalent to the training cell in the notebook (Exp 6: SiLU, M = 8×16):
```bash
python finetune.py \
    --base_model 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
    --data_path './commonsense_170k.json' \
    --output_dir './denselora_tinyllama_exp6_silu_8x16_d40/' \
    --batch_size 64 --micro_batch_size 32 --num_epochs 2 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80 \
    --adapter_name lora \
    --target_modules '["q_proj","k_proj","v_proj","up_proj","down_proj"]' \
    --lora_r 16 --lora_r1 8 --lora_r2 16 \
    --lora_alpha 32 --dense_activation silu \
    --use_gradient_checkpointing \
    --auto_resume True
```

New arguments in `finetune.py`:
- `--lora_r1` / `--lora_r2`: input/output rank of matrix M (default = `--lora_r`). Square shape when `r1 == r2`.
- `--dense_activation`: `gelu` | `silu` | `golu`.
- `--auto_resume`: detect & resume the latest checkpoint in `--output_dir`.
- `--data_fraction`: use a fraction of the data, e.g. `0.4` for 40% (shuffle `seed=42`).

The repo's built-in helper scripts (`llama3_8B_denselora.sh`, etc.) remain available for the original paper's configuration.

### Inference / Evaluation
```bash
python commonsense_evaluate.py \
    --model LLaMA3-8B --adapter LoRA --dataset boolq \
    --base_model 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
    --batch_size 1 --lora_weights './denselora_tinyllama_exp6_silu_8x16_d40/'
```
`r1/r2/activation` are read automatically from `adapter_config.json` so the shape of M matches when loading the weights.



## Citation
If you find DenseLoRA useful, please consider giving a star and citation:
```bibtex
@Inproceedings{,
  title={},
  author={},
  booktitle = {},
  year={}
}
```

## Acknowledgement
This repo benefits from Original Dense-Lora, LLM-Adapters, DoRA. Thanks for their wonderful works.
