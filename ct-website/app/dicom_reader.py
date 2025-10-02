from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pydicom


def _safe_get(ds: pydicom.Dataset, name: str, default):
    try:
        return getattr(ds, name)
    except Exception:  # noqa: BLE001
        return default


def _iter_dicom_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def read_all_series(root: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Read every DICOM series inside ``root``.

    Parameters
    ----------
    root:
        Directory with extracted study files.

    Returns
    -------
    volumes, warnings:
        ``volumes`` maps SeriesInstanceUID to the corresponding HU volume as
        ``np.ndarray`` with shape ``[Z, H, W]``. ``warnings`` collects all
        parsing issues for display in the UI.
    """

    warnings: List[str] = []
    series_map: Dict[str, List[Path]] = defaultdict(list)

    for path in _iter_dicom_files(root):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            series_uid = _safe_get(ds, "SeriesInstanceUID", None)
            if series_uid:
                series_map[str(series_uid)].append(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{path.name}: cannot read header ({exc})")

    if not series_map:
        raise RuntimeError("No DICOM series found in the archive.")

    volumes: Dict[str, np.ndarray] = {}

    def _instance_number(file_path: Path) -> int:
        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
            return int(_safe_get(ds, "InstanceNumber", 0))
        except Exception:  # noqa: BLE001
            return 0

    for series_uid, files in series_map.items():
        ordered = sorted(files, key=_instance_number)
        slices: List[np.ndarray] = []
        for fpath in ordered:
            try:
                ds = pydicom.dcmread(str(fpath), force=True)
                arr = ds.pixel_array.astype(np.int16)
                slope = float(_safe_get(ds, "RescaleSlope", 1.0))
                intercept = float(_safe_get(ds, "RescaleIntercept", 0.0))
                hu = arr * slope + intercept
                slices.append(hu)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{series_uid}: skip slice {fpath.name} ({exc})")
        if not slices:
            warnings.append(f"{series_uid}: no readable slices")
            continue
        volume = np.stack(slices, axis=0)
        volumes[series_uid] = volume

    if not volumes:
        raise RuntimeError("Found series but none contained readable pixel data.")

    return volumes, warnings
