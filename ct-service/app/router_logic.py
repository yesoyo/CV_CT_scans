from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import CFG
from .model3d import predict3d


def route_and_ensemble(score_2d: float, volume_hu: np.ndarray) -> Tuple[float, bool]:
    """
    Если score_2d в зоне неопределенности -> считаем 3D и усредняем.
    """
    routed = CFG.uncert_low < score_2d < CFG.uncert_high
    if routed:
        s3d = predict3d(volume_hu)
        final = 0.5 * (score_2d + s3d)
    else:
        final = score_2d
    return float(final), routed


def to_label(score: float) -> str:
    return "pathology" if score >= 0.5 else "normal"
