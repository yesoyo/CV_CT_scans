from __future__ import annotations
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .dicom_reader import read_all_series
from .service import SeriesResult, classifier

app = FastAPI(title="CT Scan Triage")

_templates_dir = Path(__file__).parent / "templates"
_static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(_templates_dir))
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "warnings": [],
            "has_model": classifier.has_checkpoint,
        },
    )


def _extract_zip(upload: UploadFile, dst: Path) -> list[str]:
    warnings: list[str] = []
    with zipfile.ZipFile(upload.file) as zf:
        for info in zf.infolist():
            try:
                # Prevent directory traversal.
                member_path = Path(info.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("Unsafe path inside archive.")
                target = dst / member_path
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, "r") as src, open(target, "wb") as dst_file:
                        shutil.copyfileobj(src, dst_file)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"skip {info.filename}: {exc}")
    return warnings


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip archive with DICOM studies.")

    warnings: list[str] = []
    try:
        with TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            file.file.seek(0)
            warnings.extend(_extract_zip(file, tmp_root))
            volumes, dicom_warnings = read_all_series(tmp_root)
            warnings.extend(dicom_warnings)

            results: list[SeriesResult] = classifier.classify(volumes)

    except HTTPException:
        raise
    except zipfile.BadZipFile as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot read ZIP archive: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.seek(0)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "warnings": warnings,
            "has_model": classifier.has_checkpoint,
        },
    )
