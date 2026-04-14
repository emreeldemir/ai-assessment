"""
FastAPI application for the CV-to-job matchmaker service.

Endpoints:
  POST /match             — match a CV + preferences against the job corpus
  GET  /jobs              — list all loaded jobs
  POST /jobs/reload       — reload jobs from disk (or from request body)
  GET  /health            — liveness check
"""

import json
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.matcher import Preferences, load_jobs, match, get_jobs
import app.matcher as matcher

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="CV Matchmaker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
JOBS_PATH = os.environ.get("JOBS_PATH", "/app/data/jobs.json")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def startup():
    try:
        load_jobs(JOBS_PATH)
    except Exception as e:
        print(f"STARTUP ERROR: {e}")
        raise


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class MatchRequest(BaseModel):
    cv: str = Field(..., min_length=50, description="Full CV / resume text")
    desired_roles: list[str] = Field(default_factory=list, description="e.g. ['backend', 'ml engineer']")
    preferred_locations: list[str] = Field(default_factory=list, description="e.g. ['remote', 'london']")
    preferred_levels: list[str] = Field(default_factory=list, description="e.g. ['senior', 'staff']")
    preferred_tech: list[str] = Field(default_factory=list, description="e.g. ['python', 'rust']")
    open_to_startup: bool = True
    top_n: int = Field(default=5, ge=1, le=20)


class JobMatch(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    level: str
    tech: list[str]
    semantic_score: float
    final_score: float
    boost_reasons: list[str]
    explanation: str


class MatchResponse(BaseModel):
    total_jobs_searched: int
    matches: list[JobMatch]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "jobs_loaded": len(get_jobs())}


@app.get("/jobs")
async def list_jobs():
    jobs = get_jobs()
    return {"total": len(jobs), "jobs": jobs}


@app.post("/jobs/reload")
async def reload_jobs(file: UploadFile = File(None)):
    """
    Reload the job corpus.
    - If a JSON file is uploaded, use that.
    - Otherwise, reload from the default JOBS_PATH on disk.
    """
    if file:
        content = await file.read()
        try:
            new_jobs = json.loads(content)
            if not isinstance(new_jobs, list):
                raise ValueError("Expected a JSON array of job objects")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        # Write to a temp path and reload
        tmp_path = "/tmp/jobs_uploaded.json"
        with open(tmp_path, "w") as f:
            json.dump(new_jobs, f)
        loaded = load_jobs(tmp_path)
    else:
        loaded = load_jobs(JOBS_PATH)

    return {"ok": True, "jobs_loaded": len(loaded)}


@app.post("/match", response_model=MatchResponse)
async def match_cv(req: MatchRequest):
    if not get_jobs():
        raise HTTPException(status_code=503, detail="Job corpus not loaded")

    prefs = Preferences(
        desired_roles=req.desired_roles,
        preferred_locations=req.preferred_locations,
        preferred_levels=req.preferred_levels,
        preferred_tech=req.preferred_tech,
        open_to_startup=req.open_to_startup,
    )

    try:
        results = match(req.cv, prefs, top_n=req.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    matches = [
        JobMatch(
            job_id=r.job["id"],
            title=r.job["title"],
            company=r.job["company"],
            location=r.job.get("location", ""),
            level=r.job.get("level", ""),
            tech=r.job.get("tech", []),
            semantic_score=r.semantic_score,
            final_score=r.final_score,
            boost_reasons=r.boost_reasons,
            explanation=r.explanation,
        )
        for r in results
    ]

    return MatchResponse(total_jobs_searched=len(get_jobs()), matches=matches)
