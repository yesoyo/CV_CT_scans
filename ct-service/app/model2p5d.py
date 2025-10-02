from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class Simple2p5DNet(nn.Module):
    """
    Простой CNN для per-slice предсказаний. Вход [B,3,320,320] -> [B,1]
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x  # logits


def load_model(checkpoint_path: Path, device: str) -> Tuple[nn.Module, bool]:
    """
    Возвращает (model, has_checkpoint)
    """
    model = Simple2p5DNet().to(device)
    model.eval()
    has_ckpt = False
    if checkpoint_path.exists():
        try:
            state = torch.load(str(checkpoint_path), map_location=device)
            # поддержка как чистого state_dict, так и {"state_dict": ...}
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            has_ckpt = True
        except Exception:
            has_ckpt = False
    return model, has_ckpt


@torch.inference_mode()
def predict_score_2d(model: nn.Module, stack: torch.Tensor) -> float:
    """
    stack: [Z,3,H,W] на устройстве. Возвращает средний сигмоид по Z.
    """
    if stack.ndim != 4:
        raise ValueError("stack must be [Z,3,H,W]")
    logits = model(stack)           # [Z,1]
    probs = torch.sigmoid(logits).squeeze(-1)  # [Z]
    score = float(probs.mean().item())
    return score
