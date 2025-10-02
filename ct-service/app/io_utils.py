from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Tuple


def new_job(tmp_dir: str) -> Tuple[str, Path]:
    job_id = uuid.uuid4().hex
    job_root = Path(tmp_dir) / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    return job_id, job_root


def extract_zip(zip_path: Path, to_dir: Path) -> list[str]:
    warnings: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for m in zf.infolist():
            try:
                # защита пути
                out_path = to_dir / Path(m.filename).name if ".." in m.filename else to_dir / m.filename
                if m.is_dir():
                    out_path.mkdir(parents=True, exist_ok=True)
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m, "r") as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"skip {m.filename}: {e}")
    return warnings
