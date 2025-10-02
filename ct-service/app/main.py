from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import CFG
from .dicom_reader import read_series
from .io_utils import extract_zip, new_job
from .model2p5d import load_model, predict_score_2d
from .preprocess import build_25d_stack, to_tensor_25d
from .report import save_report_xlsx
from .router_logic import route_and_ensemble, to_label
from .schemas import AnalyzeResp, HealthResp

app = FastAPI(title="CT Analyze Service")


@app.get("/health", response_model=HealthResp)
def health() -> HealthResp:
    return HealthResp()


@app.post("/analyze", response_model=AnalyzeResp)
def analyze(file: UploadFile = File(...)) -> AnalyzeResp:
    warnings: List[str] = []

    filename = file.filename or "upload.zip"
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="only .zip is supported")

    job_id, job_root = new_job(CFG.tmp_dir)
    zip_path = job_root / "upload.zip"
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        warnings += extract_zip(zip_path, job_root)
        series_uid, volume_hu, w1 = read_series(job_root)
        warnings += w1

        stack_np = build_25d_stack(volume_hu, img_size=CFG.img_size, k=CFG.k_slices)

        ckpt = Path(CFG.models_dir) / "resnet2p5d.pt"
        model, has_ckpt = load_model(ckpt, CFG.device)
        if not has_ckpt:
            warnings.append("models/resnet2p5d.pt not found, using stub=0.5")

        if has_ckpt:
            with torch.inference_mode():
                stack_t = to_tensor_25d(stack_np, CFG.device)
                score_2d = predict_score_2d(model, stack_t)
        else:
            score_2d = 0.5

        final_score, routed = route_and_ensemble(score_2d, volume_hu)
        label = to_label(final_score)

        report_path = save_report_xlsx(
            out_dir=Path(CFG.reports_dir),
            job_id=job_id,
            series_uid=series_uid,
            score=final_score,
            label=label,
            routed_to_3d=routed,
        )

        return AnalyzeResp(
            job_id=job_id,
            series_uid=series_uid,
            score=round(final_score, 6),
            label=label,
            routed_to_3d=routed,
            report_path=f"/reports/{report_path.name}",
            warnings=warnings,
        )
    except Exception as e:  # noqa: BLE001
        warnings.append(str(e))
        raise HTTPException(status_code=500, detail={"error": "analysis failed", "warnings": warnings})
    finally:
        shutil.rmtree(job_root, ignore_errors=True)


@app.get("/reports/{report_name}")
def get_report(report_name: str):
    path = Path(CFG.reports_dir) / report_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------- статика ----------
_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", include_in_schema=False)
def index():

    idx = _STATIC / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html missing")
        alt_idx = _STATIC / "assets" / "index.html"
        if alt_idx.exists():
            idx = alt_idx
        else:
            raise HTTPException(status_code=404, detail="index.html missing")
    return FileResponse(str(idx), media_type="text/html")
