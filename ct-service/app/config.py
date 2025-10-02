from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None else default


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None else default


@dataclass(frozen=True)
class AppConfig:
    k_slices: int = _get_int("K_SLICES", 5)
    img_size: int = _get_int("IMG_SIZE", 320)
    uncert_low: float = _get_float("UNCERT_LOW", 0.45)
    uncert_high: float = _get_float("UNCERT_HIGH", 0.55)
    device: str = os.getenv("DEVICE", "cuda" if os.getenv("CUDA", "1") == "1" else "cpu")
    models_dir: str = os.getenv("MODELS_DIR", "models")
    reports_dir: str = os.getenv("REPORTS_DIR", "reports")
    tmp_dir: str = os.getenv("TMP_DIR", "tmp")


CFG = AppConfig()
