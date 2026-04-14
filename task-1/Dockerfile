FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/       ./app/
COPY model/     ./model/
COPY static/    ./static/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_PATH=/app/model/mnist_cnn.pt
ENV DB_PATH=/app/data/mnist.db
ENV DATA_DIR=/app/data
ENV TRAIN_EPOCHS=5

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
