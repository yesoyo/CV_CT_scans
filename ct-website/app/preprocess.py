from __future__ import annotations

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
    x = (x - low) / (high - low + 1e-6)
    return x


def apply_windows(slice_hu: np.ndarray) -> np.ndarray:
    base_wl, base_ww = 500.0, 3000.0
    lung_wl, lung_ww = 225.0, 2100.0
    medi_wl, medi_ww = 220.0, 360.0

    c_base = window_clip(slice_hu, wl=base_wl, ww=base_ww)
    c_lung = window_clip(slice_hu, wl=lung_wl, ww=lung_ww)
    c_medi = window_clip(slice_hu, wl=medi_wl, ww=medi_ww)

    rgb = np.stack([c_base, c_lung, c_medi], axis=-1)
    return rgb.astype(np.float32)


def resize_img(x: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)


def build_25d_stack(vol_hu: np.ndarray, img_size: int, k: int) -> np.ndarray:
    z, _, _ = vol_hu.shape
    pad = k // 2
    padded = np.pad(vol_hu, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    outs = []
    for idx in range(z):
        slab = padded[idx : idx + k].mean(axis=0)
        rgb = apply_windows(slab)
        rgb = resize_img(rgb, img_size)
        rgb = np.transpose(rgb, (2, 0, 1))
        outs.append(rgb)
    arr = np.stack(outs, axis=0).astype(np.float32)
    return arr


def to_tensor_25d(arr: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(arr)
    return tensor.to(device)
