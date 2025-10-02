from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Configuration for the CT web demo."""

    device: str = os.getenv("CT_DEVICE", "cpu")
    img_size: int = int(os.getenv("CT_IMG_SIZE", 320))
    k_slices: int = int(os.getenv("CT_K_SLICES", 5))

    @property
    def model_path(self) -> Path:
        raw = os.getenv("CT_MODEL_PATH")
        if raw:
            return Path(raw)
        # Default to the model shipped with the previous service if it exists.
        root = Path(__file__).resolve().parents[2]
        fallback = root / "ct-service" / "models" / "resnet2p5d.pt"
        return fallback


SETTINGS = Settings()
