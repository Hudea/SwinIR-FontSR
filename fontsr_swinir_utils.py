"""
fontsr_swinir_utils — FontSR data adapters and SwinIR model helpers.

Uses FontSR's build_dataset(data_cfg, split) public API.
Produces SwinIR-compatible [0,1] grayscale tensors:
  lr : [1, 32, 32]   (native LR, no pre-upscaling)
  hr : [1, 128, 128]
"""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# FontSR root resolution
# ----------------------------------------------------------------------


def _resolve_fontsr_root() -> Path:
    repo_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if os.environ.get("FONTSR_ROOT"):
        candidates.append(Path(os.environ["FONTSR_ROOT"]))
    candidates.extend(
        [
            repo_dir.parent / "FontSR",
            repo_dir.parent.parent / "FontSR",
            Path("/Users/butterflies/Project/FontSR"),
        ]
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "datasets" / "dataset.py").exists():
            return root
    searched = ", ".join(str(p) for p in candidates)
    raise RuntimeError(f"Cannot locate FontSR root. Set FONTSR_ROOT. Searched: {searched}")


FONTSR_ROOT = _resolve_fontsr_root()
if str(FONTSR_ROOT) not in sys.path:
    sys.path.insert(0, str(FONTSR_ROOT))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_fontsr_path(path: str | Path) -> str:
    """Resolve a path that may be relative to FONTSR_ROOT."""
    candidate = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        parts = candidate.parts
        if "FontSR" in parts:
            rel = Path(*parts[parts.index("FontSR") + 1:])
            relocated = FONTSR_ROOT / rel
            if relocated.exists():
                return str(relocated)
        return str(candidate)
    return str(FONTSR_ROOT / candidate)


def apply_difficulty_profile(config: dict[str, Any], profile: str) -> None:
    config["data"]["level"] = profile


def resolve_device(requested: str | None) -> torch.device:
    if not requested or requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda, but CUDA is not available.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Requested mps, but MPS is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {requested}")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """FontSR normalizes to [-1,1] (mean=0.5, std=0.5); convert back to [0,1]."""
    return (t + 1.0) / 2.0


def save_grayscale_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    """Save a [1, H, W] or [H, W] tensor as grayscale PNG."""
    import cv2

    if tensor.dim() == 3:
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), arr)


def build_export_filename(index: int, semantic_hex: str, variant_hex: str) -> str:
    return f"{index:06d}_sem-{semantic_hex}_var-{variant_hex}.png"


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_path",
        "gt_hex_code",
        "gt_semantic_hex_code",
        "gt_base_char",
        "split",
        "sample_index",
        "checkpoint",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# FontSRSwinIRDataset — wraps build_dataset; denormalizes [-1,1] → [0,1]
# ----------------------------------------------------------------------


class FontSRSwinIRDataset(Dataset):
    """
    Wraps FontSR's build_dataset() public API.

    Returns SwinIR-compatible dict:
        lr : [1, 32, 32], float32, range [0, 1]   (native LR, no pre-upscaling)
        hr : [1, 128, 128], float32, range [0, 1]
        semantic_hex_code, hex_code, base_char, split, sample_index

    FontSR's OnlineRenderDataset._lr_transform by default upscales LR to HR
    size (128) via bicubic for diffusion models. We override it to skip that
    resize, keeping native 32×32 for pixelshuffledirect ×4 upsampling.
    """

    def __init__(self, config: dict[str, Any], split: str):
        super().__init__()
        self._config = config
        self._split = split

        # Resolve font/charset paths to absolute before build_dataset sees them
        data_cfg = config["data"]
        spec = data_cfg.get("spec", {})
        if "font" in spec:
            spec["font"] = dict(spec["font"])
            spec["font"]["font_path"] = resolve_fontsr_path(spec["font"]["font_path"])
        if "charset" in spec:
            spec["charset"] = dict(spec["charset"])
            spec["charset"]["char_map_path"] = resolve_fontsr_path(
                spec["charset"]["char_map_path"]
            )

        from datasets import build_dataset
        from torchvision import transforms

        self._dataset = build_dataset(data_cfg, split)

        # Override _lr_transform: skip bicubic resize, keep native 32×32
        self._dataset._lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self._dataset[index]

        hr_tensor = denormalize(raw["hr_image"])   # [1, 128, 128] in [0,1]
        lr_tensor = denormalize(raw["lr_image"])   # [1, 32, 32] in [0,1]

        if lr_tensor.dim() == 2:
            lr_tensor = lr_tensor.unsqueeze(0)
        if hr_tensor.dim() == 2:
            hr_tensor = hr_tensor.unsqueeze(0)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "semantic_hex_code": raw["semantic_hex_code"],
            "hex_code": raw["hex_code"],
            "base_char": raw["base_char"],
            "split": self._split,
            "sample_index": index,
        }


# ----------------------------------------------------------------------
# SwinIR model builder
# ----------------------------------------------------------------------


def build_swinir_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """Build SwinIR model from config dict. Lazy import inside function."""
    from models.network_swinir import SwinIR

    model = SwinIR(
        img_size=model_cfg["img_size"],
        patch_size=1,
        in_chans=model_cfg["in_chans"],
        embed_dim=model_cfg["embed_dim"],
        depths=model_cfg["depths"],
        num_heads=model_cfg["num_heads"],
        window_size=model_cfg["window_size"],
        mlp_ratio=model_cfg["mlp_ratio"],
        upscale=model_cfg["upscale"],
        upsampler=model_cfg["upsampler"],
        resi_connection=model_cfg["resi_connection"],
    )
    model.eval()
    return model
