"""
Web front end for the Data Quality Toolkit.

The analytical engine in data_quality_toolkit.py is intentionally unchanged.
This application adds the AIKosh Structured Dataset readiness UI, metadata
collection/reporting, result visualisation and downloads around that engine.
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

import data_quality_toolkit as dq

BASE_DIR = Path(__file__).parent
WORK_ROOT = Path(tempfile.gettempdir()) / "dq_toolkit_jobs"
MAX_UPLOAD_MB = int(os.environ.get("DQ_MAX_UPLOAD_MB", "500"))
JOB_TTL_SECONDS = 2 * 60 * 60

app = FastAPI(title="AIKosh AI/ML Readiness Check for Structured Dataset")
JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Lock()

METADATA_FIELDS = [
    ("Dataset Title", "dataset_title", True),
    ("Short Description", "short_description", True),
    ("Long Description / About Dataset", "long_description", True),
    ("Purpose", "purpose", True),
    ("Author", "author", True),
    ("Dataset Type", "dataset_type", True),
    ("Format and structure of dataset provided", "format", True),
    ("Year Range", "year_range", False),
    ("Primary Key / Indicator", "primary_key", False),
    ("Hosted / Redirected", "hosted_redirected", False),
    ("Sector", "sector", True),
    ("License", "license", True),
    ("Source organization", "source_organization", True),
    ("Frequency", "frequency", False),
    ("Date & Time", "date_time", False),
    ("Category", "category", False),
    ("If redirection which source", "redirect_source", False),
    ("Geographical Coverage", "geographical_coverage", False),
    ("Uploaded by", "uploaded_by", False),
    ("Time Granularity", "time_granularity", False),
    ("Visibility", "visibility", False),
    ("Data Collection Method", "data_collection_method", False),
]


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v if isinstance(v, str) else str(v)


def records(df: pd.DataFrame, cols: list[str] | None = None, limit: int | None = None) -> list[dict]:
    if df is None or df.empty:
        return []
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if limit:
        df = df.head(limit)
    return clean(df.to_dict("records"))


class JobLogHandler(logging.Handler):
    def __init__(self, job: dict):
        super().__init__(logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        self.job["log"].append(msg)
        m = re.match(r"\[(\d+)/(\d+)\] Processing", msg)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            self.job["progress"] = round(0.05 + 0.80 * (i - 1) / n, 3)
        elif "cross-dataset" in msg.lower() or "Cross comparison" in msg:
            self.job["progress"] = max(self.job["progress"], 0.88)
        elif "Master report saved" in msg:
            self.job["progress"] = 0.97


def metadata_defaults(master_df: pd.DataFrame, result_tables: list[dict]) -> list[dict]:
    """Build metadata rows from the master summary and table-level analysis."""
    rows = []
    now = datetime.now().strftime("%d %b %Y")
    for t in result_tables:
        file_name = t["file"]
        sheet = t.get("sheet") or ""
        title = Path(file_name).stem if not sheet else f"{Path(file_name).stem} - {sheet}"
        fmt = Path(file_name).suffix.lower().replace(".", "").upper()
        quality = t.get("score")
        quality_5 = round(float(quality) / 12.0, 1) if quality is not None else None
        rows.append({
            "dataset_key": t["key"],
            "file": file_name,
            "sheet": sheet,
            "metadata": {
                "dataset_title": title,
                "short_description": "",
                "long_description": "",
                "purpose": "",
                "author": "",
                "dataset_type": "Structured",
                "format": fmt,
                "year_range": "N.A.",
                "primary_key": "N.A.",
                "hosted_redirected": "Hosted",
                "sector": "",
                "license": "",
                "source_organization": "",
                "frequency": "N.A.",
                "date_time": now,
                "category": "",
                "redirect_source": "N.A.",
                "geographical_coverage": "",
                "uploaded_by": "",
                "time_granularity": "N.A.",
                "visibility": "Open",
                "data_collection_method": "N.A.",
                "rows": t.get("rows"),
                "columns": t.get("columns"),
                "missing_values": t.get("null_total"),
                "missing_percentage": t.get("null_pct"),
                "duplicate_rows": t.get("dup_rows"),
                "pii_findings": len(t.get("pii", [])),
                "small_groups": t.get("privacy", {}).get("violating_groups", 0),
                "quality_score_60": quality,
                "data_quality_score_5": quality_5,
                "fitness": t.get("fitness"),
                "quality_decision": t.get("decision"),
                "privacy_status": t.get("privacy", {}).get("status"),
            },
        })
    return rows


def build_payload(master_df: pd.DataFrame, collect: dict, k_threshold: int) -> dict:
    tables = []
    for res in collect["results"]:
        summaries = {r["Sheet Name"]: r for r in res["summary_rows"]}
        for table_key, score in res["aikosh_scores"].items():
            sheet = score["Sheet Name"]
            summ = summaries.get(sheet, {})
            components = []
            for name in score.index:
                m = re.match(r"(.+?) \((\d+)\)$", str(name))
                if m:
                    components.append({"name": m.group(1), "score": clean(score[name]), "max": int(m.group(2))})
            nulls = pd.DataFrame(res["null_details"])
            if not nulls.empty:
                nulls = nulls[nulls["Sheet Name"] == sheet].sort_values("Null Count", ascending=False)
            groups = res["kanon_groups"].get(table_key, pd.DataFrame())
            viol = res["kanon_violations"].get(table_key, pd.DataFrame())
            viol_sorted = viol.sort_values("Group_Size") if not viol.empty else viol
            profile = res["column_profiles"].get(table_key, pd.DataFrame()).copy()
            if not profile.empty:
                profile["Unique Values (sample)"] = profile["Unique Values (sample)"].astype(str).str.slice(0, 110)
            report_path = res.get("report_path")
            chart_paths = []
            for chart_path in res.get("charts", {}).get(table_key, []):
                try:
                    rel = Path(chart_path).resolve().relative_to((Path(collect.get("output_path", "")) .resolve()))
                    chart_paths.append(str(rel).replace("\\", "/"))
                except Exception:
                    try:
                        chart_paths.append(Path(chart_path).name)
                    except Exception:
                        pass
            tables.append(clean({
                "file": res["file_name"], "sheet": None if sheet == "NA" else sheet,
                "key": table_key, "rows": score["Rows"], "columns": score["Columns"],
                "fitness": score["DATASET FITNESS"], "reason": score["FITNESS REASON"],
                "decision": score["FINAL DECISION"], "score": score["TOTAL SCORE (Max 60)"],
                "components": components, "null_pct": summ.get("Null Percentage (%)"),
                "null_total": summ.get("Total Null Values"), "dup_rows": summ.get("Duplicate Rows"),
                "dup_pct": summ.get("Duplicate Percentage (%)"),
                "privacy": {
                    "status": score["K-Anonymity Status"],
                    "k": None if score["Min K-Value"] == "Not Applicable" else score["Min K-Value"],
                    "threshold": score["K Threshold"],
                    "qis": [] if score["Quasi Identifiers"] == "None detected" else str(score["Quasi Identifiers"]).split(", "),
                    "groups": len(groups), "violating_groups": len(viol),
                    "violations": {"columns": list(viol_sorted.columns), "rows": viol_sorted.head(30).values.tolist()} if not viol_sorted.empty else None,
                },
                "nulls": records(nulls, ["Column Name", "Null Count", "Null Percentage (%)"], 200),
                "profile": records(profile, None, 500), "pii": records(res["pii_reports"].get(table_key, pd.DataFrame())),
                "report": Path(report_path).name if report_path else None,
                "charts": chart_paths,
                "errors": res["errors"],
            }))

    cross_cols = ["Similarity Status", "Dataset A", "Dataset B", "Comparison Scope", "Schema Jaccard (%)",
                  "Comparison Grain", "Comparison Confidence", "Common Keys", "Key Overlap vs A (%)",
                  "Key Overlap vs B (%)", "Duplicate Keys A", "Duplicate Keys B", "Exact Row Hash Matches", "Interpretation"]
    counts = {k: int(master_df["DATASET FITNESS"].eq(k).sum()) for k in ["FIT", "CONDITIONAL", "NOT FIT"]} if not master_df.empty else {"FIT": 0, "CONDITIONAL": 0, "NOT FIT": 0}
    batch_chart = None
    if collect.get("output_path"):
        candidate = Path(collect["output_path"]) / "charts" / "fitness_distribution.png"
        if candidate.exists():
            batch_chart = "charts/fitness_distribution.png"
    return clean({
        "tables": tables, "counts": counts, "failed": collect["failed"],
        "cross": records(collect["cross_summary"], cross_cols), "k_threshold": k_threshold,
        "has_master": bool(collect.get("master_path")),
        "metadata": metadata_defaults(master_df, tables),
        "batch_chart": batch_chart,
    })


def run_job(job_id: str, opts: dict) -> None:
    job = JOBS[job_id]
    with RUN_LOCK:
        job.update(status="running", progress=0.03)
        handler = JobLogHandler(job)
        dq.LOGGER.addHandler(handler)
        dq.LOGGER.setLevel(logging.INFO)
        try:
            collect: dict = {}
            collect["output_path"] = str(job["dir"] / "output")
            master_df = dq.run_pipeline(
                input_path=str(job["dir"] / "input"), output_path=str(job["dir"] / "output"),
                k_threshold=opts["k_threshold"], recursive=False, sort_by_decision=True,
                sample_size=50, max_rows=opts["max_rows"], max_qi=opts["max_qi"],
                regex_sample_size=opts["pii_sample_size"], enable_charts=opts["charts"],
                cross_dataset=opts["cross_dataset"], cross_keys=opts["cross_keys"],
                cross_max_rows=None, collect=collect,
            )
            if master_df.empty:
                raise RuntimeError("No readable tables were found in the uploaded files.")
            job["result"] = build_payload(master_df, collect, opts["k_threshold"])
            job["master_df"] = master_df
            job["collect"] = collect
            # Create a metadata workbook immediately so the download is available
            # even before the user fills optional or missing metadata fields.
            metadata_report(job, {})
            out = job["dir"] / "output"
            zip_path = job["dir"] / "DQ_Output.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in out.rglob("*"):
                    if p.is_file():
                        zf.write(p, p.relative_to(out))
            job.update(status="done", progress=1.0)
        except Exception as exc:
            dq.LOGGER.error("Run failed: %s", exc)
            job.update(status="error", error=str(exc))
        finally:
            dq.LOGGER.removeHandler(handler)


def purge_old_jobs():
    now = time.time()
    for jid in [j for j, v in JOBS.items() if now - v["created"] > JOB_TTL_SECONDS]:
        shutil.rmtree(JOBS[jid]["dir"], ignore_errors=True)
        JOBS.pop(jid, None)


def parse_int(value: str, default: int | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def metadata_report(job: dict, submitted: dict | None = None) -> Path:
    out = job["dir"] / "output"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    supplied = submitted or {}
    for item in job["result"]["metadata"]:
        md = dict(item["metadata"])
        key = item["dataset_key"]
        md.update(supplied.get(key, {}))
        rows.append({"Dataset Key": key, "File": item["file"], "Sheet": item["sheet"], **md})

    path = out / "AIKosh_Metadata_Report.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Metadata", index=False)
        wb = writer.book
        ws = wb["Metadata"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True, color="FFFFFF")
            cell.fill = cell.fill.copy(fgColor="17232B", fill_type="solid")
        for col in ws.columns:
            width = min(max(max(len(str(c.value or "")) for c in col) + 2, 14), 42)
            ws.column_dimensions[col[0].column_letter].width = width

        guide = pd.DataFrame([
            {"Field": label, "Required": "Yes" if required else "No", "Guidance": "Enter or confirm the dataset metadata. N.A. may be used where the field genuinely does not apply."}
            for label, _, required in METADATA_FIELDS
        ])
        guide.to_excel(writer, sheet_name="Metadata_Guide", index=False)

        assessment = pd.DataFrame([
            {"Metric": "Fit", "Count": job["result"]["counts"].get("FIT", 0)},
            {"Metric": "Conditional", "Count": job["result"]["counts"].get("CONDITIONAL", 0)},
            {"Metric": "Not fit", "Count": job["result"]["counts"].get("NOT FIT", 0)},
        ])
        assessment.to_excel(writer, sheet_name="Assessment_Summary", index=False)
    return path


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), k_threshold: str = Form("5"), cross_dataset: str = Form("true"),
                  cross_keys: str = Form(""), charts: str = Form("true"), max_rows: str = Form(""),
                  max_qi: str = Form("10"), pii_sample_size: str = Form("5000")):
    purge_old_jobs()
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_ROOT / job_id
    in_dir = job_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    saved, used, total = [], set(), 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    for f in files:
        name = Path(f.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in dq.SUPPORTED_EXTENSIONS:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, f"'{name}' isn't supported. Use CSV, XLS, XLSX or Parquet files.")
        stem, n = Path(name).stem, 1
        while name.lower() in used:
            n += 1
            name = f"{stem}_{n}{ext}"
        used.add(name.lower())
        with open(in_dir / name, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                total += len(chunk)
                if total > limit:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(413, f"Upload is larger than {MAX_UPLOAD_MB} MB. Set DQ_MAX_UPLOAD_MB to raise the limit.")
                out.write(chunk)
        saved.append(name)
    k = parse_int(k_threshold, 5); mq = parse_int(max_qi, 10); ps = parse_int(pii_sample_size, 5000); mr = parse_int(max_rows, None)
    opts = {"k_threshold": k if k and k > 0 else 5, "max_qi": None if mq is None or mq <= 0 else mq,
            "pii_sample_size": None if ps is None or ps <= 0 else ps, "max_rows": mr if mr and mr > 0 else None,
            "cross_dataset": cross_dataset.lower() == "true", "cross_keys": [x.strip() for x in cross_keys.split(",") if x.strip()] or None,
            "charts": charts.lower() == "true"}
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "log": [], "result": None, "error": None,
                    "dir": job_dir, "created": time.time(), "files": saved, "metadata_submitted": {}}
    threading.Thread(target=run_job, args=(job_id, opts), daemon=True).start()
    return {"job_id": job_id, "files": saved}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "This run has expired. Upload your files again.")
    body = {"status": job["status"], "progress": job["progress"], "log": job["log"][-6:], "error": job["error"]}
    if job["status"] == "done":
        body["result"] = job["result"]
    return JSONResponse(body)


@app.post("/api/jobs/{job_id}/metadata")
async def save_metadata(job_id: str, payload: dict):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Assessment not available.")
    job["metadata_submitted"] = payload.get("datasets", {}) if isinstance(payload, dict) else {}
    path = metadata_report(job, job["metadata_submitted"])
    return {"ok": True, "file": path.name}


@app.get("/api/jobs/{job_id}/chart")
def chart(job_id: str, name: str):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Assessment not available.")
    out = (job["dir"] / "output").resolve()
    try:
        path = (out / name).resolve()
    except Exception:
        raise HTTPException(400, "Invalid chart path.")
    if out not in path.parents or not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(404, "Chart not found.")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str, name: str | None = None):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Nothing to download for this run.")
    out = job["dir"] / "output"
    if kind == "master":
        path = out / "Master_Data_Quality_Report.xlsx"
    elif kind == "metadata":
        path = out / "AIKosh_Metadata_Report.xlsx"
    elif kind == "zip":
        path = job["dir"] / "DQ_Output.zip"
    elif kind == "report" and name:
        path = out / "reports" / Path(name).name
    else:
        raise HTTPException(400, "Unknown download.")
    if not path.exists():
        raise HTTPException(404, "That file was not generated for this run.")
    return FileResponse(path, filename=path.name)


if __name__ == "__main__":
    host, port = os.environ.get("DQ_HOST", "127.0.0.1"), int(os.environ.get("DQ_PORT", "8000"))
    print(f"AIKosh AI/ML Readiness Check for Structured Dataset running at http://{host}:{port}")
    if host in ("127.0.0.1", "localhost"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
