"""
Train SwinIR-Light on the FontSR LR32 -> HR128 data flow.

Single-GPU:
    CUDA_VISIBLE_DEVICES=0 python SwinIR/train_fontsr_swinir.py \
      --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
      --difficulty-profile level3 \
      --output-dir experiments/swinir_lr32_level3 \
      --device cuda --epochs 200 --batch-size 16 --num-workers 4

Multi-GPU DDP (recommended):
    CUDA_VISIBLE_DEVICES=2,1,0 torchrun --nproc_per_node=3 \
      SwinIR/train_fontsr_swinir.py \
      --config SwinIR/configs/fontsr_swinir_lr32_level1.yaml \
      --difficulty-profile level4 \
      --output-dir experiments/swinir_lr32_level4 \
      --device cuda --epochs 200 --batch-size 16 --num-workers 4
      (batch-size is per-GPU; effective batch = batch-size × nproc_per_node)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

from fontsr_swinir_utils import (
    FontSRSwinIRDataset,
    apply_difficulty_profile,
    build_swinir_model,
    load_config,
    resolve_device,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _ddp_enabled() -> bool:
    return "LOCAL_RANK" in os.environ


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_main() -> bool:
    return _local_rank() == 0


def init_ddp() -> torch.device:
    """Initialise NCCL process group; returns this rank's device."""
    dist.init_process_group(backend="nccl")
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks (in-place)."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (DDP, torch.nn.DataParallel)):
        return model.module
    return model


def maybe_wrap_data_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info("Using DataParallel across %d visible CUDA devices", torch.cuda.device_count())
        return torch.nn.DataParallel(model)
    return model


def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(train_cfg.get("lr", 2e-4))
    optim_type = str(train_cfg.get("optim", "adam")).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if optim_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99))
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--difficulty-profile", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # DDP init (torchrun sets LOCAL_RANK) or fall back to DataParallel
    # ------------------------------------------------------------------
    use_ddp = _ddp_enabled()
    if use_ddp:
        device = init_ddp()
        if _is_main():
            logger.info("DDP mode: %d GPUs, this rank=%d", dist.get_world_size(), _local_rank())
    else:
        device = resolve_device(args.device)

    config = load_config(args.config)
    if args.difficulty_profile:
        apply_difficulty_profile(config, args.difficulty_profile)

    # ------------------------------------------------------------------
    # --check-config (rank 0 only)
    # ------------------------------------------------------------------
    if args.check_config:
        if use_ddp:
            cleanup_ddp()
        logger.info("Running --check-config (dataset validation only)")
        train_ds = FontSRSwinIRDataset(config, split="train")
        val_ds = FontSRSwinIRDataset(config, split="val")
        logger.info("  train samples: %d", len(train_ds))
        logger.info("  val samples: %d", len(val_ds))
        s = train_ds[0]
        assert s["lr"].shape == (1, 32, 32), s["lr"].shape
        assert s["hr"].shape == (1, 128, 128), s["hr"].shape
        logger.info("CHECK_CONFIG_OK")
        print("CHECK_CONFIG_OK")
        sys.exit(0)

    if not args.output_dir:
        parser.error("--output-dir is required for training (omit for --check-config)")

    output_dir = Path(args.output_dir)
    if not use_ddp or _is_main():
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()
    checkpoint_dir = output_dir / "checkpoints"

    train_cfg = config.get("train", {})
    set_seed(int(train_cfg.get("seed", 42)) + _local_rank())

    train_ds = FontSRSwinIRDataset(config, split="train")
    val_ds = FontSRSwinIRDataset(config, split="val")
    if not use_ddp or _is_main():
        logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    pin_memory = device.type == "cuda"

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = build_swinir_model(config["model"]).to(device)

    start_epoch = 1
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        if not use_ddp or _is_main():
            logger.info("Loaded model weights from %s", args.resume_from)

    if use_ddp:
        model = DDP(model, device_ids=[_local_rank()])
    else:
        model = maybe_wrap_data_parallel(model, device)

    optimizer = build_optimizer(unwrap_model(model), train_cfg)

    scheduler_cfg = train_cfg.get("scheduler", {})
    milestones = scheduler_cfg.get("milestones", [100, 150, 175])
    gamma = float(scheduler_cfg.get("gamma", 0.5))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(m) for m in milestones], gamma=gamma,
    )
    criterion = torch.nn.L1Loss()

    if resume_checkpoint:
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])

    if not use_ddp or _is_main():
        with open(output_dir / "used_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train", disable=not _is_main())
        for batch in pbar:
            lr_batch = batch["lr"].to(device, non_blocking=pin_memory)
            hr_batch = batch["hr"].to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            if _is_main():
                pbar.set_postfix({"l1": f"{loss.item():.4f}"})
            if args.max_train_batches and train_batches >= args.max_train_batches:
                break

        avg_train_loss = train_loss / train_batches if train_batches else 0.0
        scheduler.step()

        # ---- Val ----
        model.eval()
        val_loss_t = torch.tensor(0.0, device=device)
        val_psnr_t = torch.tensor(0.0, device=device)
        val_batches = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} val", disable=not _is_main())
            for batch in pbar_val:
                lr_batch = batch["lr"].to(device, non_blocking=pin_memory)
                hr_batch = batch["hr"].to(device, non_blocking=pin_memory)
                sr_batch = model(lr_batch)
                loss = criterion(sr_batch, hr_batch)
                val_loss_t += loss
                val_psnr_t += psnr_metric(sr_batch.clamp(0.0, 1.0), hr_batch)
                val_batches += 1
                if args.max_val_batches and val_batches >= args.max_val_batches:
                    break

        val_loss_t /= max(val_batches, 1)
        val_psnr_t /= max(val_batches, 1)
        if use_ddp:
            all_reduce_mean(val_loss_t)
            all_reduce_mean(val_psnr_t)

        avg_val_loss = val_loss_t.item()
        avg_val_psnr = val_psnr_t.item()

        if not use_ddp or _is_main():
            logger.info(
                "Epoch %d: train_l1=%.4f val_l1=%.4f val_psnr=%.2f lr=%.2e",
                epoch, avg_train_loss, avg_val_loss, avg_val_psnr,
                scheduler.get_last_lr()[0],
            )

        # ---- Checkpoint (rank 0 only) ----
        if not use_ddp or _is_main():
            checkpoint_payload = {
                "epoch": epoch,
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
            }
            torch.save(checkpoint_payload, checkpoint_dir / "last.pt")
            save_milestone = epoch == args.epochs or (
                args.save_every > 0 and epoch % args.save_every == 0
            )
            if save_milestone:
                torch.save(checkpoint_payload, checkpoint_dir / f"epoch_{epoch:03d}.pt")
                logger.info("Saved milestone: epoch_%03d.pt", epoch)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
                logger.info("New best (val_l1=%.4f)", best_val_loss)

        if use_ddp:
            dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()
