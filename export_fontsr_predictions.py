"""
export_fontsr_predictions.py — Export SwinIR SR predictions + optional retrieval eval.

Usage:
    uv run python export_fontsr_predictions.py \
        --config configs/fontsr_swinir_lr32_level1.yaml \
        --checkpoint experiments/smoke_swinir_fontsr/checkpoints/epoch_001.pt \
        --split validation \
        --batch-size 1 \
        --max-samples 2 \
        --device cpu \
        --output-dir experiments/smoke_swinir_fontsr/export_epoch_001 \
        --save-lr-hr \
        --run-retrieval-eval \
        --retrieval-device cpu \
        --retrieval-metric hybrid \
        --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from fontsr_swinir_utils import (
    FONTSR_ROOT,
    FontSRSwinIRDataset,
    build_export_filename,
    build_swinir_model,
    load_config,
    resolve_device,
    resolve_fontsr_path,
    save_grayscale_tensor,
    write_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split", type=str, default="validation", choices=["validation", "val", "train"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-lr-hr", action="store_true")
    parser.add_argument("--run-retrieval-eval", action="store_true")
    parser.add_argument("--retrieval-device", type=str, default="cpu")
    parser.add_argument("--retrieval-metric", type=str, default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    # Normalize split name
    split = "val" if args.split in ("val", "validation") else "train"

    # Build dataset
    ds = FontSRSwinIRDataset(config, split=split)
    logger.info("Export split '%s': %d samples", split, len(ds))

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Load model from checkpoint
    model_cfg = config["model"]
    model = build_swinir_model(model_cfg)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # Create output directories
    sr_dir = output_dir / "sr"
    sr_dir.mkdir(parents=True, exist_ok=True)
    if args.save_lr_hr:
        lr_dir = output_dir / "lr"
        hr_dir = output_dir / "hr"
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)

    # Export loop
    manifest_rows = []
    global_sample_idx = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Exporting")
        for batch in pbar:
            batch_lr = batch["lr"].to(device)
            batch_hr = batch["hr"]
            batch_sem = batch["semantic_hex_code"]
            batch_hex = batch["hex_code"]
            batch_base = batch["base_char"]
            batch_split = batch["split"]
            batch_sample_idx = batch["sample_index"]

            batch_sr = model(batch_lr)

            for i in range(batch_sr.size(0)):
                sem_hex = batch_sem[i]
                var_hex = batch_hex[i]
                base_char = batch_base[i]
                split_val = batch_split[i]
                sample_idx = int(batch_sample_idx[i].item()) if isinstance(batch_sample_idx[i], torch.Tensor) else int(batch_sample_idx[i])

                filename = build_export_filename(global_sample_idx, sem_hex, var_hex)

                # Save SR
                save_grayscale_tensor(sr_dir / filename, batch_sr[i].cpu())

                # Save LR/HR if requested
                if args.save_lr_hr:
                    save_grayscale_tensor(lr_dir / filename, batch_lr[i].cpu())
                    save_grayscale_tensor(hr_dir / filename, batch_hr[i].cpu())

                manifest_rows.append({
                    "image_path": f"sr/{filename}",
                    "gt_hex_code": var_hex,
                    "gt_semantic_hex_code": sem_hex,
                    "gt_base_char": base_char,
                    "split": split_val,
                    "sample_index": sample_idx,
                    "checkpoint": args.checkpoint,
                })

                global_sample_idx += 1

                if args.max_samples and global_sample_idx >= args.max_samples:
                    break

            if args.max_samples and global_sample_idx >= args.max_samples:
                break

    # Write manifest
    manifest_path = output_dir / "predictions_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    logger.info("Wrote manifest: %s (%d rows)", manifest_path, len(manifest_rows))

    # Write export summary
    summary = {
        "split": split,
        "num_samples": global_sample_idx,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "tensor_range": "[0,1]",
        "adapter_note": "FontSR rendering/split/profile logic is reused; tensors are converted with to_tensor and kept in [0,1] for SwinIR img_range=1.0.",
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save used config
    with open(output_dir / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    logger.info("Export complete: %d samples", global_sample_idx)

    # Run retrieval eval if requested
    if args.run_retrieval_eval:
        eval_dir = output_dir / "retrieval_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        retrieval_cmd = [
            sys.executable,
            str(FONTSR_ROOT / "scripts/eval_variant_retrieval.py"),
            "--manifest", str(manifest_path),
            "--label-mode", "manifest",
            "--font-path", resolve_fontsr_path(config["data"].get("font_path", "resources/shufei.ttf")),
            "--char-map-path", resolve_fontsr_path(config["data"].get("char_map_path", "resources/meta/shufei.txt")),
            "--output-dir", str(eval_dir),
            "--metric", args.retrieval_metric,
            "--device", args.retrieval_device,
            "--top-k", str(args.top_k),
        ]
        logger.info("Running retrieval eval: %s", " ".join(retrieval_cmd))
        result = subprocess.run(
            retrieval_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Retrieval eval failed: %s", result.stderr)
            sys.exit(1)
        logger.info("Retrieval eval output:\n%s", result.stdout)


if __name__ == "__main__":
    main()
