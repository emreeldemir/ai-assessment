# Task 1 — MNIST Digit Classifier

A containerised digit-recognition service: draw a digit in the browser, get an instant prediction, and optionally correct the model if it guesses wrong.

## Stack

| Layer | Choice |
|-------|--------|
| API | FastAPI (Python 3.11) |
| ML | PyTorch — 2-layer CNN |
| Frontend | Vanilla JS + HTML Canvas |
| Storage | SQLite |
| Container | Docker + Compose |

## Setup & Running

**Prerequisites:** Docker + Docker Compose (v2).

```bash
cd task-1
docker compose up --build
```

On first boot the container trains the model (~2–3 min on CPU, ~5 epochs, ~99% test accuracy), then starts the API.
Subsequent starts skip training because weights are cached in a named volume.

Open `http://localhost:8000` in your browser.

## Usage

1. Draw a digit (0–9) on the black canvas.
2. Click **Predict**.
3. The model returns a digit, confidence %, and a probability bar for each class.
4. If the prediction is wrong, click the correct digit in the **Feedback** grid and hit **Submit Correction**.
5. The **Recent Predictions** table at the bottom logs all events and shows feedback status.

## API Endpoints

### `POST /predict`

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-PNG>"}' | jq
```

Response:
```json
{
  "prediction_id": 42,
  "digit": 7,
  "confidence": 0.9821,
  "probabilities": [0.0, 0.0, 0.0001, 0.001, 0.0, 0.0, 0.0, 0.9821, 0.0, 0.0]
}
```

### `POST /feedback`

```bash
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": 42, "correct_label": 1}' | jq
```

Response:
```json
{ "ok": true, "message": "Feedback recorded. Thank you!" }
```

### `GET /predictions?limit=20`

Returns recent prediction + feedback rows:

```json
{
  "predictions": [
    {
      "id": 42,
      "created_at": "2026-04-14T10:30:00+00:00",
      "digit": 7,
      "confidence": 0.9821,
      "correct_label": 1
    }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

## Re-training

To force a re-train, remove the model volume and restart:

```bash
docker compose down -v
docker compose up --build
```
