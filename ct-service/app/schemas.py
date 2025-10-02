from __future__ import annotations

from pydantic import BaseModel


class HealthResp(BaseModel):
    ok: bool = True


class AnalyzeResp(BaseModel):
    job_id: str
    series_uid: str
    score: float
    label: str  # "normal" | "pathology"
    routed_to_3d: bool
    report_path: str
    warnings: list[str]
