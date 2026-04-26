# FontSR-SwinIR LR32→HR128 Comparison

Unconditional SwinIR grayscale x4 baseline for LR32→HR128 super-resolution.
No `semantic_id` conditioning. Data logic mirrors FontSR `OnlineRenderDataset`.

## Quick Start

Run these commands from the controlled comparison root:

```bash
cd /Users/butterflies/Project/FontSR_Controlled
export FONTSR_ROOT=/Users/butterflies/Project/FontSR
```

`experiments/...` is intentionally rooted at this directory, so it can be a
single symlink to a larger disk shared by all comparison baselines.

### Conda Environment

On remote machines without `uv`, use the FontSR conda environment directly.
It already covers the FontSR data stack; SwinIR adds `timm` and `cv2`.

```bash
conda activate <fontsr-env>
export FONTSR_ROOT=/path/to/FontSR
python -c "import torch, torchvision, freetype, yaml, PIL, skimage, tqdm; print('FONTSR_ENV_OK')"
python -c "import timm, cv2; print('SWINIR_EXTRA_OK')"
```

If the second check fails, install the small SwinIR extra set:

```bash
pip install -r SwinIR/requirements-fontsr-conda.txt
```

All commands below are shown with plain `python` for conda. If you are on the
local `uv` setup, replace `python` with `uv run --project SwinIR python`.

### 1. Config check

```bash
python SwinIR/train_fontsr_swinir.py \
    --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --check-config
```

Exit code 0 + `CHECK_CONFIG_OK` in output confirms dataset shapes.

### 2. Smoke train (1 epoch)

```bash
python SwinIR/train_fontsr_swinir.py \
    --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --epochs 1 \
    --batch-size 1 \
    --save-every 10 \
    --max-train-batches 1 \
    --max-val-batches 1 \
    --device cpu \
    --output-dir experiments/smoke_swinir_fontsr
```

Outputs:
- `experiments/smoke_swinir_fontsr/checkpoints/epoch_001.pt`
- `experiments/smoke_swinir_fontsr/checkpoints/best.pt`
- `experiments/smoke_swinir_fontsr/checkpoints/last.pt`
- `experiments/smoke_swinir_fontsr/used_config.yaml`

### 3. Export predictions (2 validation samples)

```bash
python SwinIR/export_fontsr_predictions.py \
    --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --checkpoint experiments/smoke_swinir_fontsr/checkpoints/epoch_001.pt \
    --split validation \
    --batch-size 1 \
    --max-samples 2 \
    --device cpu \
    --output-dir experiments/smoke_swinir_fontsr/export_epoch_001 \
    --save-lr-hr
```

Outputs:
- `experiments/smoke_swinir_fontsr/export_epoch_001/sr/` — 2 PNG files
- `experiments/smoke_swinir_fontsr/export_epoch_001/predictions_manifest.csv`
- `experiments/smoke_swinir_fontsr/export_epoch_001/lr/` and `hr/` (if `--save-lr-hr`)

### 4. Retrieval evaluation

```bash
python SwinIR/export_fontsr_predictions.py \
    --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --checkpoint experiments/smoke_swinir_fontsr/checkpoints/epoch_001.pt \
    --split validation \
    --batch-size 1 \
    --max-samples 2 \
    --device cpu \
    --output-dir experiments/smoke_swinir_fontsr/export_epoch_001 \
    --run-retrieval-eval \
    --retrieval-device cpu \
    --retrieval-metric hybrid \
    --top-k 3
```

Or run the eval script directly:

```bash
python "$FONTSR_ROOT/scripts/eval_variant_retrieval.py" \
    --manifest experiments/smoke_swinir_fontsr/export_epoch_001/predictions_manifest.csv \
    --label-mode manifest \
    --font-path "$FONTSR_ROOT/resources/shufei.ttf" \
    --char-map-path "$FONTSR_ROOT/resources/meta/shufei.txt" \
    --output-dir experiments/smoke_swinir_fontsr/export_epoch_001/retrieval_eval \
    --device cpu \
    --top-k 3
```

Output: `experiments/smoke_swinir_fontsr/export_epoch_001/retrieval_eval/summary.json`

## Formal Runs

Run the aligned level1 / level2 baselines with the same training budget:

```bash
CUDA_VISIBLE_DEVICES=0 \
python SwinIR/train_fontsr_swinir.py \
    --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --difficulty-profile level1 \
    --epochs 200 \
    --batch-size 8 \
    --save-every 10 \
    --device cuda \
    --output-dir experiments/swinir_lr32_level1

CUDA_VISIBLE_DEVICES=1,2 \
python SwinIR/train_fontsr_swinir.py \
    --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --difficulty-profile level2 \
    --epochs 200 \
    --batch-size 8 \
    --save-every 10 \
    --device cuda \
    --output-dir experiments/swinir_lr32_level2
```

Use `--batch-size 16` only if both runs fit comfortably in memory. Keep the
same batch size for level1 and level2.
When multiple CUDA devices are visible, the trainer uses PyTorch DataParallel
and still saves plain SwinIR state dicts for strict export loading.

Checkpoint policy:

- `checkpoints/best.pt`: overwritten whenever `val_l1` improves.
- `checkpoints/last.pt`: overwritten every epoch for resume/final-state export.
- `checkpoints/epoch_010.pt`, `epoch_020.pt`, ...: milestones controlled by `--save-every`.
- The final epoch is always saved as `epoch_XXX.pt`, even if it is not on a save boundary.

Export both `best.pt` and the final checkpoint, using each run's resolved
`used_config.yaml`:

```bash
python SwinIR/export_fontsr_predictions.py \
    --config experiments/swinir_lr32_level1/used_config.yaml \
    --checkpoint experiments/swinir_lr32_level1/checkpoints/best.pt \
    --split validation \
    --batch-size 8 \
    --device auto \
    --output-dir experiments/swinir_lr32_level1/export_best \
    --save-lr-hr \
    --run-retrieval-eval \
    --retrieval-device auto \
    --retrieval-metric hybrid \
    --top-k 3

python SwinIR/export_fontsr_predictions.py \
    --config experiments/swinir_lr32_level1/used_config.yaml \
    --checkpoint experiments/swinir_lr32_level1/checkpoints/epoch_200.pt \
    --split validation \
    --batch-size 8 \
    --device auto \
    --output-dir experiments/swinir_lr32_level1/export_epoch_200 \
    --save-lr-hr \
    --run-retrieval-eval \
    --retrieval-device auto \
    --retrieval-metric hybrid \
    --top-k 3
```

Repeat the two export commands for `experiments/swinir_lr32_level2/...`.

## Difficulty Profiles

Override the default `level1` profile with `--difficulty-profile`:

```bash
# level1: renderer-domain variation (random hinting + subpixel)
python SwinIR/train_fontsr_swinir.py --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --difficulty-profile level2 ...  # adds LR font-size variation

python SwinIR/train_fontsr_swinir.py --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
    --difficulty-profile level3 ...  # level2 + geometric perturbation
```

level4 adds information loss (blur, JPEG, noise). level5 is reserved.

## Files

- `fontsr_swinir_utils.py` — dataset adapter, model builder, utility functions
- `train_fontsr_swinir.py` — training entry point
- `export_fontsr_predictions.py` — export + retrieval eval entry point
- `configs/fontsr_swinir_lr32_level1.yaml` — default LR32 level1 config
- `requirements-fontsr-conda.txt` — extra packages to install on top of FontSR conda env

## Notes

- This is an **unconditional** baseline: no `semantic_id` conditioning is used.
- SwinIR operates in grayscale (in_chans=1), channel adapter applied at data boundary.
- The adapter keeps tensors in `[0,1]` for SwinIR `img_range=1.0`; FontSR sample
  enumeration, rendering, split, and difficulty profile logic still come from
  `OnlineRenderDataset`.
- Validation samples are deterministic per sample seed — same indices produce identical LR renders across runs.
- Pretrained SwinIR weights are not used — this is a from-scratch training baseline.
