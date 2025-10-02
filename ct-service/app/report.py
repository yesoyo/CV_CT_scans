from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook


def save_report_xlsx(
    out_dir: Path,
    job_id: str,
    series_uid: str,
    score: float,
    label: str,
    routed_to_3d: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{job_id}.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "report"

    ws.append(["job_id", "series_uid", "score_pathology", "pred_label", "routed_to_3d"])
    ws.append([job_id, series_uid, round(score, 6), label, routed_to_3d])

    wb.save(str(path))
    return path
