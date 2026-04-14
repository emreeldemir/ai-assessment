FROM python:3.11-slim

WORKDIR /app

# System deps for torch/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container starts fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app/     ./app/
COPY static/  ./static/
COPY data/    ./data/

ENV JOBS_PATH=/app/data/jobs.json
ENV ANTHROPIC_API_KEY=""
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
