#!/bin/sh
set -e

MODEL_PATH="${MODEL_PATH:-/app/model/mnist_cnn.pt}"

# Train if weights are missing
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model weights not found. Training now (this takes ~2-3 min on CPU)..."
  cd /app && python model/train.py
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
