from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pydicom


def _safe_get(ds: pydicom.Dataset, name: str, default):
    try:
        return getattr(ds, name)
    except Exception:
        return default


def read_series(root: Path) -> Tuple[str, np.ndarray, list[str]]:
    """
    Ищет все DICOM, группирует по SeriesInstanceUID, берет самую большую серию.
    Возвращает (series_uid, volume[Z,H,W], warnings)
    """
    warnings: list[str] = []
    dicom_files = [p for p in root.rglob("*") if p.is_file()]
    series_map: Dict[str, List[Path]] = defaultdict(list)

    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            uid = _safe_get(ds, "SeriesInstanceUID", None)
            if uid:
                series_map[uid].append(fp)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"bad dicom header {fp.name}: {e}")

    if not series_map:
        raise RuntimeError("no dicom series found")

    # берем серию с максимальным количеством файлов
    series_uid, files = max(series_map.items(), key=lambda kv: len(kv[1]))

    # сортировка по InstanceNumber
    def _ins_num(p: Path) -> int:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            return int(_safe_get(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    files_sorted = sorted(files, key=_ins_num)

    slices = []
    for p in files_sorted:
        try:
            ds = pydicom.dcmread(str(p), force=True)
            arr = ds.pixel_array.astype(np.int16)
            slope = float(_safe_get(ds, "RescaleSlope", 1.0))
            intercept = float(_safe_get(ds, "RescaleIntercept", 0.0))
            hu = arr * slope + intercept
            slices.append(hu)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"skip pixel {p.name}: {e}")

    if not slices:
        raise RuntimeError("series has no valid slices")

    vol = np.stack(slices, axis=0)  # [Z,H,W]
    return str(series_uid), vol, warnings
