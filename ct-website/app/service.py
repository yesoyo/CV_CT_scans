from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .config import SETTINGS
from .model2p5d import load_model, predict_score_2d
from .preprocess import build_25d_stack, to_tensor_25d


@dataclass
class SeriesResult:
    series_uid: str
    num_slices: int
    score: float
    label: str


class ClassificationService:
    def __init__(self) -> None:
        self._model, self._has_checkpoint = load_model(SETTINGS.model_path, SETTINGS.device)
        if not self._has_checkpoint:
            # Model weights are optional for local demo usage. We still keep the
            # module instantiated so that the UI works with a neutral score.
            self._model = self._model.to(SETTINGS.device)

    @property
    def has_checkpoint(self) -> bool:
        return self._has_checkpoint

    def _predict_single(self, volume: np.ndarray) -> float:
        stack_np = build_25d_stack(volume, SETTINGS.img_size, SETTINGS.k_slices)
        if not self._has_checkpoint:
            return 0.5
        stack_t = to_tensor_25d(stack_np, SETTINGS.device)
        with torch.inference_mode():
            score = predict_score_2d(self._model, stack_t)
        return float(score)

    def classify(self, volumes: Dict[str, np.ndarray]) -> List[SeriesResult]:
        results: List[SeriesResult] = []
        for series_uid, volume in volumes.items():
            score = self._predict_single(volume)
            label = "pathology" if score >= 0.5 else "normal"
            results.append(
                SeriesResult(
                    series_uid=series_uid,
                    num_slices=int(volume.shape[0]),
                    score=round(float(score), 4),
                    label=label,
                )
            )
        return results


classifier = ClassificationService()
