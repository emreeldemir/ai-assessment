"""
Core matching logic.

Strategy:
  1. Embed each job description + the candidate's CV using sentence-transformers.
  2. Compute cosine similarity between CV embedding and each job embedding.
  3. Apply preference-based re-ranking boosts on top of semantic similarity.
  4. Return top-N jobs with scores and a structured rationale per job.

Explanation generation:
  - Uses Claude API (claude-haiku-4-5 for speed/cost) to produce a concise,
    human-readable rationale for each top match.
  - Falls back to a template-based explanation if the API key is absent.
"""

import json
import os
from dataclasses import dataclass, field

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
EXPLANATION_MODEL = "claude-haiku-4-5-20251001"

# Lazy-loaded globals
_embedder: SentenceTransformer | None = None
_job_embeddings: dict[str, np.ndarray] = {}
_jobs: list[dict] = []


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def get_jobs() -> list[dict]:
    """Safe accessor for the job list — avoids module-level reference issues."""
    return _jobs


def load_jobs(path: str) -> list[dict]:
    global _jobs, _job_embeddings
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Jobs file not found: {path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in jobs file: {e}")

    _jobs = data
    print(f"Loaded {len(_jobs)} jobs from {path}")

    try:
        embedder = _get_embedder()
        texts = [_job_text(j) for j in _jobs]
        vecs = embedder.encode(texts, batch_size=32, show_progress_bar=False)
        _job_embeddings = {j["id"]: vecs[i] for i, j in enumerate(_jobs)}
        print(f"Embedded {len(_jobs)} jobs successfully")
    except Exception as e:
        print(f"WARNING: Embedding failed ({e}). Embeddings will be computed on-demand.")

    return _jobs


def _job_text(job: dict) -> str:
    """Concatenate job fields into a single string for embedding."""
    tech = ", ".join(job.get("tech", []))
    return (
        f"{job['title']} at {job['company']}. "
        f"Level: {job.get('level', '')}. "
        f"Location: {job.get('location', '')}. "
        f"Tech: {tech}. "
        f"{job.get('description', '')}"
    )


@dataclass
class Preferences:
    desired_roles: list[str] = field(default_factory=list)   # e.g. ["backend", "ml"]
    preferred_locations: list[str] = field(default_factory=list)  # e.g. ["remote", "london"]
    preferred_levels: list[str] = field(default_factory=list)     # e.g. ["senior", "staff"]
    preferred_tech: list[str] = field(default_factory=list)       # e.g. ["python", "rust"]
    open_to_startup: bool = True


@dataclass
class MatchResult:
    job: dict
    semantic_score: float    # raw cosine similarity
    final_score: float       # after preference boosts
    boost_reasons: list[str]
    explanation: str         # LLM or template-generated rationale


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _apply_preference_boosts(
    job: dict,
    score: float,
    prefs: Preferences,
) -> tuple[float, list[str]]:
    """
    Apply preference boosts/penalties on top of semantic similarity.

    Boost magnitudes are intentionally strong so that explicit preferences
    dominate the final ranking — a user who asks for Estonia expects Estonia
    jobs near the top regardless of baseline semantic score.

    Conversely, a job that misses an explicitly requested location or level
    receives a penalty to push it below matching jobs.
    """
    reasons: list[str] = []
    boosted = score

    # Location: strong boost on match, penalty on miss when preference is set
    if prefs.preferred_locations:
        job_loc = job.get("location", "").lower()
        location_matched = any(loc.lower() in job_loc for loc in prefs.preferred_locations)
        if location_matched:
            boosted *= 1.45
            matched = next(loc for loc in prefs.preferred_locations if loc.lower() in job_loc)
            reasons.append(f"Location match: {job['location']}")
        else:
            boosted *= 0.70
            reasons.append(f"Location mismatch (preferred: {', '.join(prefs.preferred_locations)})")

    # Level: meaningful boost on match, penalty on miss when preference is set
    if prefs.preferred_levels:
        level_matched = job.get("level", "").lower() in [l.lower() for l in prefs.preferred_levels]
        if level_matched:
            boosted *= 1.25
            reasons.append(f"Seniority match: {job['level']}")
        else:
            boosted *= 0.80
            reasons.append(f"Seniority mismatch (preferred: {', '.join(prefs.preferred_levels)})")

    # Tech stack: cumulative boost per matching technology (capped at 5 matches)
    if prefs.preferred_tech:
        job_tech_lower = [t.lower() for t in job.get("tech", [])]
        matched_tech = [t for t in prefs.preferred_tech if t.lower() in job_tech_lower]
        if matched_tech:
            boost = 1 + 0.08 * min(len(matched_tech), 5)
            boosted *= boost
            reasons.append(f"Tech overlap: {', '.join(matched_tech)}")

    # Startup flag
    startup_signals = ["startup", "stealth", "founding", "early stage", "seed"]
    title_desc = (job.get("title", "") + job.get("description", "") + job.get("company", "")).lower()
    is_startup = any(s in title_desc for s in startup_signals)
    if prefs.open_to_startup and is_startup:
        boosted *= 1.12
        reasons.append("Startup environment match")
    elif not prefs.open_to_startup and is_startup:
        boosted *= 0.65
        reasons.append("Startup excluded by preference")

    return boosted, reasons


def _explain_with_llm(cv_text: str, job: dict, score: float, boost_reasons: list[str]) -> str:
    """Call Claude to generate a concise match explanation."""
    if not ANTHROPIC_API_KEY:
        return _fallback_explanation(job, score, boost_reasons)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tech = ", ".join(job.get("tech", []))
    boost_text = "; ".join(boost_reasons) if boost_reasons else "no specific preference boosts"

    prompt = f"""You are a recruitment assistant. Given a candidate's CV and a job listing, write a concise 2-3 sentence explanation of why this job is a strong match. Be specific about skills and experience alignment. Do not use filler phrases like "This role is a great fit." Just explain what aligns.

Job: {job['title']} at {job['company']}
Level: {job.get('level', 'N/A')} | Location: {job.get('location', 'N/A')}
Tech stack: {tech}
Description: {job['description']}

Candidate CV (excerpt):
{cv_text[:1500]}

Semantic similarity score: {score:.2f}/1.0
Preference boosts applied: {boost_text}

Write the explanation now:"""

    try:
        message = client.messages.create(
            model=EXPLANATION_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"LLM explanation failed: {e}")
        return _fallback_explanation(job, score, boost_reasons)


def _fallback_explanation(job: dict, score: float, boost_reasons: list[str]) -> str:
    tech = ", ".join(job.get("tech", [])[:4])
    parts = [
        f"Strong semantic alignment with your profile (score: {score:.2f}).",
        f"Key technologies match: {tech}." if tech else "",
    ]
    if boost_reasons:
        parts.append("Additional preference matches: " + "; ".join(boost_reasons) + ".")
    return " ".join(p for p in parts if p)


def match(
    cv_text: str,
    prefs: Preferences,
    top_n: int = 5,
    jobs_override: list[dict] | None = None,
) -> list[MatchResult]:
    """
    Match a CV against the loaded job corpus and return top_n results.
    jobs_override allows passing a custom corpus for testing.
    """
    jobs = jobs_override if jobs_override is not None else _jobs

    if not jobs:
        raise RuntimeError("No jobs loaded. Call load_jobs() first.")

    embedder = _get_embedder()
    cv_vec = embedder.encode(cv_text, show_progress_bar=False)

    scored: list[tuple[dict, float, list[str]]] = []
    for job in jobs:
        job_vec = _job_embeddings.get(job["id"])
        if job_vec is None:
            # Embed on the fly (e.g. for externally loaded jobs)
            job_vec = embedder.encode(_job_text(job), show_progress_bar=False)
            _job_embeddings[job["id"]] = job_vec

        sem_score = _cosine(cv_vec, job_vec)
        final_score, boost_reasons = _apply_preference_boosts(job, sem_score, prefs)
        scored.append((job, sem_score, final_score, boost_reasons))

    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:top_n]

    results: list[MatchResult] = []
    for job, sem_score, final_score, boost_reasons in top:
        explanation = _explain_with_llm(cv_text, job, sem_score, boost_reasons)
        results.append(MatchResult(
            job=job,
            semantic_score=round(sem_score, 4),
            final_score=round(final_score, 4),
            boost_reasons=boost_reasons,
            explanation=explanation,
        ))

    return results
