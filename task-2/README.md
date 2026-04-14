# Task 2 — CV Matchmaker

A service that takes CV/resume text, matches it against a job corpus using semantic embeddings, applies candidate preference boosts, and explains each match in plain English.

## Stack

| Layer | Choice |
|-------|--------|
| API | FastAPI (Python 3.11) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Explanations | Claude API (`claude-haiku-4-5`) — falls back to template if no key |
| Frontend | Vanilla JS |
| Container | Docker + Compose |

## Setup & Running

**Prerequisites:** Docker + Docker Compose (v2).

```bash
cd task-2
docker compose up --build
```

Open `http://localhost:8001` in your browser.

The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded during `docker build` so container startup is instant.

### Optional: Claude API explanations

For richer natural-language explanations, set your Anthropic API key:

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

Without it the service works fully — explanations are generated from a template instead.

## Usage

1. Paste your CV text into the left panel (or upload a `.txt` file).
2. Set preferences: location, seniority, tech stack, top-N, startup openness.
3. Click **Find Matches**.
4. Each result shows: title, company, match score, tech pills, preference boost tags, and a plain-English explanation.

## API Endpoints

### `POST /match`

```bash
curl -s -X POST http://localhost:8001/match \
  -H "Content-Type: application/json" \
  -d '{
    "cv": "Senior Python engineer with 6 years experience. Skilled in FastAPI, PostgreSQL, Kubernetes. Led backend teams at two startups.",
    "preferred_locations": ["remote"],
    "preferred_levels": ["senior"],
    "preferred_tech": ["python", "kubernetes"],
    "open_to_startup": true,
    "top_n": 3
  }' | jq
```

Response:
```json
{
  "total_jobs_searched": 20,
  "matches": [
    {
      "job_id": "job-007",
      "title": "AI Product Engineer",
      "company": "Anthropic",
      "location": "San Francisco, CA",
      "level": "senior",
      "tech": ["Python", "TypeScript", "React", "FastAPI", "Claude API"],
      "semantic_score": 0.6821,
      "final_score": 0.7234,
      "boost_reasons": ["Seniority match: senior", "Tech overlap: python"],
      "explanation": "Your FastAPI and Python background maps directly onto this role's core stack..."
    }
  ]
}
```

### `GET /jobs`

Returns the full loaded job corpus.

### `POST /jobs/reload`

Reload jobs from disk, or upload a new corpus:

```bash
# Reload from default path
curl -X POST http://localhost:8001/jobs/reload

# Upload a custom jobs file
curl -X POST http://localhost:8001/jobs/reload \
  -F "file=@/path/to/my_jobs.json"
```

The uploaded file must be a JSON array of job objects with at least: `id`, `title`, `company`, `description`.

### `GET /health`

```json
{ "status": "ok", "jobs_loaded": 20 }
```

## Loading Custom Jobs

Place a `jobs.json` file in the `data/` directory (mounted as a volume):

```json
[
  {
    "id": "my-job-001",
    "title": "Backend Engineer",
    "company": "Acme Corp",
    "location": "Remote",
    "type": "full-time",
    "level": "senior",
    "tech": ["Python", "PostgreSQL"],
    "description": "..."
  }
]
```

Then reload without rebuilding:
```bash
curl -X POST http://localhost:8001/jobs/reload
```
