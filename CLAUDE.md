# CLAUDE.md — Candidate Brief Project

## Project Overview

This repo contains a technical assessment consisting of two independent tasks:

- **task-1/**: MNIST digit classifier (ML + API + UI + feedback loop)
- **task-2/**: CV-to-job matchmaker (LLM/embedding-based + explainable + UI)

## Hard Constraints

- **Python backend/API** is required for both tasks.
- Each task must include its own `docker-compose.yml`; the full stack must start with `docker compose up`.
- Each task folder must contain `README.md` and `TECHNICAL_COMMENTARY.md`.
- Final submission: one GitHub repository with `task-1/` and `task-2/` folders.
- All code, documentation, comments, and API responses must be in **English**.

## Repo Structure

```
candidate-brief/          ← working directory
  CLAUDE.md
  README.md               ← assignment brief (do not modify)
  task-1-mnist-classifier.md
  task-2-cv-matchmaker.md
  submission-template/    ← empty scaffolds

task-1/                   ← to be created
  README.md
  TECHNICAL_COMMENTARY.md
  docker-compose.yml
  ...

task-2/                   ← to be created
  README.md
  TECHNICAL_COMMENTARY.md
  docker-compose.yml
  ...
```

## Task 1: MNIST Classifier — Requirements

### Functional
- Model trained on MNIST capable of inference
- Simple UI where user can draw a digit or upload an image
- Display model prediction in the UI
- **Feedback loop**: if model is wrong, user can submit the correct label
- Persist prediction events and feedback (DB or log)

### Technical
- Python API (FastAPI recommended)
- Model: PyTorch or TensorFlow/Keras (simple CNN is sufficient)
- Frontend: minimal (vanilla JS or Streamlit)
- Database: SQLite is sufficient
- Single-command startup via `docker-compose.yml`

### Deliverables
- Working code
- `README.md`: setup, run instructions, example requests
- `TECHNICAL_COMMENTARY.md`: approach, assumptions, scope cuts, architecture decisions, trade-offs, sanity checks, risks/limitations

## Task 2: CV Matchmaker — Requirements

### Functional
- Service that accepts CV/resume text (or structured input)
- Matches against a realistic job corpus (at least 10–20 listings)
- **Preference-aware**: candidate preferences (location, tech stack, seniority, etc.) influence ranking
- **Explainable output**: for each recommended role, explain *why* it ranked highly (not just a score)
- Simple frontend for demo/testing
- **External job/CV loading without code changes** (JSON/CSV or similar)

### Technical
- Python API (FastAPI recommended)
- Matching: embedding similarity (sentence-transformers) + LLM explanation (Claude API), or fully LLM-based
- Frontend: minimal (vanilla JS or Streamlit)
- Single-command startup via `docker-compose.yml`

### Deliverables
- Working code
- `README.md`: setup, run instructions, example requests/outputs
- `TECHNICAL_COMMENTARY.md`: approach, assumptions, scope cuts, architecture decisions, trade-offs, sanity checks, risks/limitations

## README Template (Per Task)

```markdown
## Setup

## Running

docker compose up

## Example Requests

## Expected Output
```

## TECHNICAL_COMMENTARY Template (Per Task)

```markdown
## Approach

## Assumptions

## Scope Cuts and Rationale

## Architecture and Stack Decisions

## Trade-offs and Alternatives Considered

## Sanity Check — How I Verified the Output

## Risks, Limitations, and Next Improvements
```

## Development Notes

- Depth over breadth: fewer things done well beats many things done partially
- Scope cuts are acceptable but must be documented in `TECHNICAL_COMMENTARY.md`
- Code readability and clear architectural decisions are part of the evaluation
- AI tooling is permitted
- Expected effort: ~4 hours total
