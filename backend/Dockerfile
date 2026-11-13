# Leafora FastAPI backend.
# CPU-only TensorFlow: inference does not need a GPU, and Render has none.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

# libgomp1 is required by TensorFlow; the rest keeps the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

# Defaults; Render overrides these with its own environment variables.
ENV MODEL_PATH=/app/models/exported/leafora_densenet121_field.keras \
    CLASS_NAMES_PATH=/app/app/ml/class_names.json \
    MODEL_VERSION=leafora-densenet121-field-v1

EXPOSE 8000

# Render provides $PORT. Never hard-code a production port.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
