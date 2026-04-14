"""
FastAPI application for MNIST digit classification.

Endpoints:
  POST /predict          — accept a base64 PNG, return prediction + confidence
  POST /feedback         — store a user correction for a prior prediction
  GET  /predictions      — list recent prediction + feedback events
  GET  /health           — liveness check
"""

import base64
import io
import os
import sys

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — model_def lives in /app/model/
# ---------------------------------------------------------------------------
sys.path.insert(0, "/app/model")
from model_def import MNISTNet  # noqa: E402

from app.database import init_db, log_prediction, log_feedback, get_recent_predictions  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="MNIST Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/mnist_cnn.pt")
device = torch.device("cpu")
model: MNISTNet | None = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}. Run train.py first.")
    m = MNISTNet()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    model = m
    print(f"Model loaded from {MODEL_PATH}")


@app.on_event("startup")
async def startup():
    init_db()
    load_model()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG/JPEG of the digit (any size)")


class PredictResponse(BaseModel):
    prediction_id: int
    digit: int
    confidence: float
    probabilities: list[float]


class FeedbackRequest(BaseModel):
    prediction_id: int
    correct_label: int = Field(..., ge=0, le=9)


class FeedbackResponse(BaseModel):
    ok: bool
    message: str


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def preprocess(b64_image: str) -> torch.Tensor:
    """
    Decode a base64 image, convert to 28x28 grayscale float tensor
    normalised to MNIST statistics.
    Handles both white-on-black and black-on-white input by inverting
    if the mean pixel value is > 128 (i.e. white background).
    """
    try:
        image_bytes = base64.b64decode(b64_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    # Invert if background is white (MNIST uses white digit on black bg)
    if arr.mean() > 128:
        arr = 255.0 - arr

    arr = arr / 255.0
    arr = (arr - 0.1307) / 0.3081

    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return tensor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tensor = preprocess(req.image)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    prediction_id = log_prediction(digit, confidence, req.image[:64])  # store prefix only

    return PredictResponse(
        prediction_id=prediction_id,
        digit=digit,
        confidence=confidence,
        probabilities=[round(p, 4) for p in probs],
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    log_feedback(req.prediction_id, req.correct_label)
    return FeedbackResponse(ok=True, message="Feedback recorded. Thank you!")


@app.get("/predictions")
async def predictions(limit: int = 20):
    rows = get_recent_predictions(limit)
    return {"predictions": rows}
