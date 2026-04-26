"""
fontsr_swinir_utils — FontSR data adapters and SwinIR model helpers.

Reuses OnlineRenderDataset from FontSR for data logic but produces
SwinIR-compatible [0,1] grayscale tensors: LR at native 32×32, HR at 128×128.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image

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
    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"Cannot locate FontSR root. Set FONTSR_ROOT. Searched: {searched}")


# FontSR root — env override first, then common sibling layouts.
FONTSR_ROOT = _resolve_fontsr_root()
if str(FONTSR_ROOT) not in sys.path:
    sys.path.insert(0, str(FONTSR_ROOT))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from datasets.char_map import CharMap, CharVariant
from datasets.renderer import FontRenderer
from datasets.dataset import OnlineRenderDataset

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_fontsr_path(path: str | Path) -> str:
    candidate = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate)
        parts = candidate.parts
        if "FontSR" in parts:
            rel = Path(*parts[parts.index("FontSR") + 1 :])
            relocated = FONTSR_ROOT / rel
            if relocated.exists():
                return str(relocated)
        return str(candidate)
    return str(FONTSR_ROOT / candidate)


def apply_difficulty_profile(config: dict[str, Any], profile: str) -> None:
    config["data"]["difficulty_profile"] = profile


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


def denormalize_fontsr_tensor(t: torch.Tensor) -> torch.Tensor:
    """FontSR normalizes to [-1,1]; convert back to [0,1]."""
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
# Replication of OnlineRenderDataset private seed logic
# ----------------------------------------------------------------------


def _sample_seed(split: str, split_seed: int, namespace: str, index: int, hex_code: str) -> int:
    """Stateless hash-based seed — mirrors OnlineRenderDataset._sample_seed.

    Keep the full 8-byte integer for Python's random module. NumPy receives a
    32-bit modulo inside ``_seed_random_sources``, matching FontSR.
    """
    raw = f"{split}:{split_seed}:{namespace}:{index}:{hex_code}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


@contextmanager
def _seed_random_sources(seed: int):
    """Context manager that saves/restores random state and seeds with given seed."""
    import random
    import numpy as np

    # Python random
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed % (2 ** 63))

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)


# ----------------------------------------------------------------------
# FontSRSwinIRDataset — replicates OnlineRenderDataset.__getitem__ flow
# ----------------------------------------------------------------------


class FontSRSwinIRDataset(torch.utils.data.Dataset):
    """
    Wraps OnlineRenderDataset but returns SwinIR-compatible tensors:

    - lr : [1, 32, 32], float32, range [0, 1]
    - hr : [1, 128, 128], float32, range [0, 1]
    - semantic_hex_code, hex_code, base_char, split, sample_index

    Replicates the rendering/pipeline flow of OnlineRenderDataset but
    replaces transforms with a plain functional.to_tensor (no resize, no normalize).
    """

    def __init__(self, config: dict[str, Any], split: str):
        super().__init__()
        self._config = config
        self._data_cfg = config["data"]
        self._split = split
        self._is_train = split in ("train", "train_split")

        # Instantiate parent dataset to get access to its internals
        params = self._data_cfg[self._split]["params"]
        difficulty_profile = self._data_cfg.get("difficulty_profile", "level1")

        self._parent = OnlineRenderDataset(
            font_path=resolve_fontsr_path(self._data_cfg["font_path"]),
            char_map_path=resolve_fontsr_path(self._data_cfg["char_map_path"]),
            hr_size=params["hr_size"],
            lr_size=params["lr_size"],
            hr_font_size=params["hr_font_size"],
            lr_font_size=params["lr_font_size"],
            hr_hinting=params.get("hr_hinting", "mac"),
            pad_mode=params.get("pad_mode", "center"),
            split=split,
            train_ratio=params.get("train_ratio", 0.9),
            split_seed=params.get("split_seed", 42),
            split_mode=params.get("split_mode", "variant"),
            difficulty_profile=difficulty_profile,
            forced_val_chars=params.get("forced_val_chars"),
            return_stroke_vector=False,
        )

        # Extract internals we need to replicate rendering/pipeline flow
        self._samples: list[CharVariant] = self._parent.samples
        self._renderer: FontRenderer = self._parent._renderer
        self._char_map: CharMap = self._parent._char_map
        self._hr_hinting = self._parent._hr_hinting
        self._lr_hinting_probs = self._parent._lr_hinting_probs
        self._lr_subpixel_range = self._parent._lr_subpixel_range
        self._val_lr_hinting = self._parent._val_lr_hinting
        self._val_lr_subpixel = self._parent._val_lr_subpixel
        self._val_use_renderer_randomization = self._parent._val_use_renderer_randomization
        self._pipeline = self._parent._pipeline
        self._phase_b_enabled = self._parent._phase_b_enabled
        self._phase_b_lr_font_scale_factors = self._parent._phase_b_lr_font_scale_factors
        self._split_seed = self._parent._split_seed
        self._hr_size = params["hr_size"]
        self._lr_size = params["lr_size"]
        self._hr_font_size = params["hr_font_size"]
        self._lr_font_size = params["lr_font_size"]
        self._pad_mode = params.get("pad_mode", "center")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        variant = self._samples[index]
        char = variant.unicode_char

        # Replicate parent logic: font size sampling
        legacy_font_scale = self._sample_legacy_font_scale(index, variant)
        phase_b_lr_scale = self._sample_phase_b_lr_font_scale(index, variant)
        hr_font_size = max(1, round(self._hr_font_size * legacy_font_scale))
        lr_font_size = max(
            1, round(self._lr_font_size * legacy_font_scale * phase_b_lr_scale)
        )

        # HR render
        hr_img = self._renderer.render_to_square(
            char,
            font_size=hr_font_size,
            target_size=self._hr_size,
            hinting=self._hr_hinting,
            subpixel_shift=0.0,
            pad_mode=self._pad_mode,
        )

        # LR render — training uses randomization, validation uses seeded deterministic path
        if self._is_train:
            lr_img, _ = self._renderer.render_with_randomization(
                char,
                font_size=lr_font_size,
                target_size=self._lr_size,
                hinting_probs=self._lr_hinting_probs,
                subpixel_range=self._lr_subpixel_range,
                pad_mode=self._pad_mode,
            )
        else:
            if self._val_use_renderer_randomization:
                with _seed_random_sources(
                    _sample_seed(
                        self._split, self._split_seed, "val_lr_render", index, variant.hex_code
                    )
                ):
                    lr_img, _ = self._renderer.render_with_randomization(
                        char,
                        font_size=lr_font_size,
                        target_size=self._lr_size,
                        hinting_probs=self._lr_hinting_probs,
                        subpixel_range=self._lr_subpixel_range,
                        pad_mode=self._pad_mode,
                    )
            else:
                lr_img = self._renderer.render_to_square(
                    char,
                    font_size=lr_font_size,
                    target_size=self._lr_size,
                    hinting=self._val_lr_hinting,
                    subpixel_shift=self._val_lr_subpixel,
                    pad_mode=self._pad_mode,
                )

        hr_pil = Image.fromarray(hr_img, mode="L")
        lr_pil = Image.fromarray(lr_img, mode="L")

        # Pipeline (Phase C/D geometric + degradation) — disabled for level1 validation
        if self._pipeline:
            if self._is_train:
                hr_pil, lr_pil = self._pipeline(hr_pil, lr_pil)
            else:
                with _seed_random_sources(
                    _sample_seed(
                        self._split, self._split_seed, "val_pipeline", index, variant.hex_code
                    )
                ):
                    hr_pil, lr_pil = self._pipeline(hr_pil, lr_pil)

        # Our transform: plain to_tensor only (no resize, no normalize)
        # [0,255] uint8 PIL → [0,1] float32 tensor
        hr_tensor = F.to_tensor(hr_pil)  # [1, 128, 128] or [1, H, W]
        lr_tensor = F.to_tensor(lr_pil)  # [1, 32, 32]

        # Ensure channel dimension exists for grayscale (to_tensor may give [H,W] for 1-channel)
        if lr_tensor.dim() == 2:
            lr_tensor = lr_tensor.unsqueeze(0)
        if hr_tensor.dim() == 2:
            hr_tensor = hr_tensor.unsqueeze(0)

        # Slice to exact expected sizes (to_tensor preserves actual size)
        lr_tensor = lr_tensor[:, : self._lr_size, : self._lr_size]
        hr_tensor = hr_tensor[:, : self._hr_size, : self._hr_size]

        return {
            "lr": lr_tensor,  # [1, 32, 32], [0, 1]
            "hr": hr_tensor,  # [1, 128, 128], [0, 1]
            "semantic_hex_code": variant.semantic_hex_code,
            "hex_code": variant.hex_code,
            "base_char": variant.base_char,
            "split": self._split,
            "sample_index": index,
        }

    # ------------------------------------------------------------------
    # Font size sampling — mirrors OnlineRenderDataset private methods
    # ------------------------------------------------------------------

    def _sample_legacy_font_scale(self, index: int, variant: CharVariant) -> float:
        """Returns 1.0 for level1 (disabled)."""
        if self._is_train and self._parent._font_scale_factors:
            return random.choice(self._parent._font_scale_factors)
        if (
            not self._is_train
            and self._parent._font_scale_factors
            and self._val_use_renderer_randomization
        ):
            with _seed_random_sources(
                _sample_seed(
                    self._split, self._split_seed, "val_font_scale", index, variant.hex_code
                )
            ):
                return random.choice(self._parent._font_scale_factors)
        return 1.0

    def _sample_phase_b_lr_font_scale(self, index: int, variant: CharVariant) -> float:
        """Returns 1.0 when phase_b is disabled (level1)."""
        if not self._phase_b_enabled:
            return 1.0
        factors = self._phase_b_lr_font_scale_factors
        if self._is_train:
            return random.choice(factors)
        with _seed_random_sources(
            _sample_seed(
                self._split, self._split_seed, "phase_b_font_size", index, variant.hex_code
            )
        ):
            return random.choice(factors)


# ----------------------------------------------------------------------
# SwinIR model builder (lazy import to avoid timm dependency at --check-config)
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
