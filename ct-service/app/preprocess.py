from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

WINDOWS = {
    "base": (-1000.0, 2000.0),
    "lung": (-600.0, 1500.0),
    "medi": (40.0, 400.0),
}


def window_clip(x: np.ndarray, wl: float, ww: float) -> np.ndarray:
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    x = np.clip(x, low, high)
    x = (x - low) / (high - low + 1e-6)  # 0..1
    return x


def apply_windows(slice_hu: np.ndarray) -> np.ndarray:
    # base(-1000,2000), lung(-600,1500), medi(40,400)
    c_base = window_clip(slice_hu, wl=500.0, ww=3000.0)   # ~(-1000..2000)
    c_lung = window_clip(slice_hu, wl=225.0, ww=2100.0)   # ~(-600..1500)
    c_medi = window_clip(slice_hu, wl=220.0, ww=360.0)    # ~(40..400)
    rgb = np.stack([c_base, c_lung, c_medi], axis=-1)     # [H,W,3]
    return rgb.astype(np.float32)


def resize_img(x: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)


def build_25d_stack(vol_hu: np.ndarray, img_size: int, k: int) -> np.ndarray:
    """
    На каждый индекс z усредняем k соседей по Z, получаем [Z,3,H,W] -> потом resize.
    Паддинг отражением.
    """
    z, h, w = vol_hu.shape
    pad = k // 2
    padded = np.pad(vol_hu, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    outs = []
    for i in range(z):
        slab = padded[i:i + k].mean(axis=0)  # [H,W]
        rgb = apply_windows(slab)            # [H,W,3]
        rgb = resize_img(rgb, img_size)
        rgb = np.transpose(rgb, (2, 0, 1))   # [3,H,W]
        outs.append(rgb)
    arr = np.stack(outs, axis=0).astype(np.float32)  # [Z,3,H,W]
    return arr


def to_tensor_25d(arr: np.ndarray, device: str) -> torch.Tensor:
    # [Z,3,H,W] -> torch [Z,3,H,W]
    t = torch.from_numpy(arr)
    return t.to(device)
