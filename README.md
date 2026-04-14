# AI Engineering Assessment

Two independent tasks completed as part of a technical assessment for an AI-oriented product team.

## Structure

```
ai-assessment/
  task-1/   — MNIST digit classifier with feedback loop
  task-2/   — CV-to-job matchmaker with explainable matching
```

## Task 1 — MNIST Digit Classifier

Draw a digit in the browser, get an instant prediction, and correct the model if it guesses wrong. All predictions and feedback are persisted to SQLite.

**Stack:** FastAPI · PyTorch CNN · Vanilla JS · SQLite · Docker

```bash
cd task-1
docker compose up --build
# Open http://localhost:8000
```

First boot trains the model (~3 min on CPU). Subsequent starts are instant.

→ See [`task-1/README.md`](task-1/README.md) for full setup and API docs.

## Task 2 — CV Matchmaker

Paste a CV, set preferences (location, seniority, tech stack), and get ranked job matches with a plain-English explanation for each result.

**Stack:** FastAPI · sentence-transformers · Claude API · Vanilla JS · Docker

```bash
cd task-2
docker compose up --build
# Open http://localhost:8001
```

Optionally set `ANTHROPIC_API_KEY` for LLM-generated explanations. Works without it using template-based fallback.

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build
```

→ See [`task-2/README.md`](task-2/README.md) for full setup and API docs.

## Running Both Tasks Simultaneously

Tasks run on different ports (8000 and 8001) and can be started independently at the same time.
