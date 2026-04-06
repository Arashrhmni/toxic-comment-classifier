FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download tokenizer and DistilBERT weights
RUN python -c "from transformers import DistilBertTokenizerFast, DistilBertModel; \
    DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'); \
    DistilBertModel.from_pretrained('distilbert-base-uncased')"

COPY model/ ./model/
COPY app/ ./app/
COPY scripts/ ./scripts/

RUN mkdir -p /app/checkpoints /app/data

ENV CHECKPOINT_PATH=/app/checkpoints/best_model.pt
ENV PREDICTION_THRESHOLD=0.5

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
