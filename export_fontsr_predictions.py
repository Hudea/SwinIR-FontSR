"""
export_fontsr_predictions.py — Export SwinIR SR predictions + optional recog eval.

Usage:
    CUDA_VISIBLE_DEVICES=0 python SwinIR/export_fontsr_predictions.py \
        --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
        --checkpoint experiments/swinir_lr32_level3/checkpoints/best.pt \
        --difficulty-profile level3 \
        --split validation \
        --batch-size 8 \
        --device cuda \
        --output-dir experiments/swinir_lr32_level3/export_best \
        --save-lr-hr \
        --run-recog-eval \
        --retrieval-device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from fontsr_swinir_utils import (
    FONTSR_ROOT,
    FontSRSwinIRDataset,
    apply_difficulty_profile,
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
    parser.add_argument("--difficulty-profile", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="validation", choices=["validation", "val", "train"]
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-lr-hr", action="store_true")
    parser.add_argument("--run-recog-eval", action="store_true")
    parser.add_argument("--run-retrieval-eval", dest="run_recog_eval", action="store_true")
    parser.add_argument("--retrieval-device", type=str, default="cpu")
    parser.add_argument(
        "--methods", type=str, default="retrieval_hybrid,retrieval_structural"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.difficulty_profile:
        apply_difficulty_profile(config, args.difficulty_profile)

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    # Normalize split name
    split = "val" if args.split in ("val", "validation") else "train"

    # Build dataset
    ds = FontSRSwinIRDataset(config, split=split)
    logger.info("Export split '%s': %d samples", split, len(ds))

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Load model from checkpoint
    model_cfg = config["model"]
    model = build_swinir_model(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
                sem_hex = batch_sem[i] if isinstance(batch_sem[i], str) else batch_sem[i].item()
                var_hex = batch_hex[i] if isinstance(batch_hex[i], str) else batch_hex[i].item()
                base_char = batch_base[i] if isinstance(batch_base[i], str) else batch_base[i].item()
                split_val = batch_split[i] if isinstance(batch_split[i], str) else batch_split[i].item()
                sample_idx = int(batch_sample_idx[i].item()) if isinstance(batch_sample_idx[i], torch.Tensor) else int(batch_sample_idx[i])

                filename = build_export_filename(global_sample_idx, str(sem_hex), str(var_hex))

                save_grayscale_tensor(sr_dir / filename, batch_sr[i].cpu())
                if args.save_lr_hr:
                    save_grayscale_tensor(lr_dir / filename, batch_lr[i].cpu())
                    save_grayscale_tensor(hr_dir / filename, batch_hr[i].cpu())

                manifest_rows.append({
                    "image_path": f"sr/{filename}",
                    "gt_hex_code": str(var_hex),
                    "gt_semantic_hex_code": str(sem_hex),
                    "gt_base_char": str(base_char),
                    "split": str(split_val),
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
        "adapter_note": (
            "FontSR build_dataset public API; tensors denormalized from [-1,1] to [0,1] "
            "for SwinIR img_range=1.0. LR=32×32, HR=128×128."
        ),
    }
    with open(output_dir / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    logger.info("Export complete: %d samples", global_sample_idx)

    # Run recog eval if requested
    if args.run_recog_eval:
        eval_dir = output_dir / "retrieval_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        font_path = resolve_fontsr_path(
            config["data"].get("spec", {}).get("font", {}).get("font_path", "resources/shufei.ttf")
        )
        char_map_path = resolve_fontsr_path(
            config["data"].get("spec", {}).get("charset", {}).get(
                "char_map_path", "resources/meta/shufei.txt"
            )
        )
        recog_cmd = [
            sys.executable,
            str(FONTSR_ROOT / "recog" / "evaluate_manifest.py"),
            "--manifest", str(manifest_path),
            "--output-dir", str(eval_dir),
            "--methods", args.methods,
            "--device", args.retrieval_device,
            "--top-k", str(args.top_k),
            "--font-path", font_path,
            "--char-map-path", char_map_path,
        ]
        logger.info("Running recog eval: %s", " ".join(recog_cmd))
        result = subprocess.run(recog_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Recog eval failed: %s", result.stderr)
            sys.exit(1)
        logger.info("Recog eval output:\n%s", result.stdout)


if __name__ == "__main__":
    main()
