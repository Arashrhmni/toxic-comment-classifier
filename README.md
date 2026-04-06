# Toxic Comment Classifier

Multi-label toxicity detection fine-tuned on **DistilBERT** using PyTorch. Classifies comments across 6 toxicity categories simultaneously, served via a FastAPI inference API with streaming-ready batch endpoints.

Trained on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. Achieves **~0.98 column-wise ROC-AUC** on the full dataset.

---

## Why this project

Real-world NLP tasks are rarely single-label. This project demonstrates:
- **Fine-tuning a transformer** (DistilBERT) for multi-label classification
- Handling **class imbalance** with BCEWithLogitsLoss + `pos_weight`
- **Differential learning rates** — lower LR for BERT layers, higher for the classification head
- **Early stopping** + best-checkpoint saving during training
- Decoupling **retrieval quality** (embeddings) from **generation quality** (LLM) — same principle used in RAG pipelines
- Wrapping a trained model in a **production FastAPI service** with proper schema validation

---

## Architecture

```
Input text
    │
    ▼
DistilBertTokenizer (max_length=128, [CLS] prepended)
    │
    ▼
DistilBERT base (66M params, fine-tuned)
    │
    ▼
[CLS] token hidden state  (768-dim)
    │
    ▼
Dropout(0.3)
    │
    ▼
Linear(768 → 6)
    │
    ▼
Sigmoid  →  [toxic, severe_toxic, obscene, threat, insult, identity_hate]
             each independently in [0, 1]
```

**Loss:** `BCEWithLogitsLoss` with `pos_weight=10.0` per label to compensate for ~10:1 class imbalance.

**Optimizer:** `AdamW` with layer-wise LR — `2e-5` for BERT body, `2e-4` for classifier head.

**Scheduler:** Linear warmup decay over full training.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/toxic-comment-classifier
cd toxic-comment-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get data

**Option A — Kaggle dataset (recommended for real metrics):**
```bash
# Install Kaggle CLI and place kaggle.json in ~/.kaggle/
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
unzip jigsaw-toxic-comment-classification-challenge.zip -d data/
```

**Option B — Synthetic data (no account needed, runs in ~2 min on CPU):**
```bash
python scripts/generate_sample_data.py
```

### 3. Train

```bash
# Full training (GPU recommended)
python -m model.train --data-dir ./data --epochs 3 --batch-size 32

# Quick smoke test on CPU (synthetic data, 10% sample)
python -m model.train --data-dir ./data --epochs 1 --sample-frac 0.1

# All options
python -m model.train --help
```

Training saves `checkpoints/best_model.pt` and `checkpoints/training_results.json`.

### 4. Run inference API

```bash
uvicorn app.api:app --reload
```

Or with Docker:

```bash
docker compose up --build
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Reference

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are completely wrong and an absolute idiot."}'
```

```json
{
  "scores": {
    "toxic": 0.9341,
    "severe_toxic": 0.1203,
    "obscene": 0.2871,
    "threat": 0.0412,
    "insult": 0.8762,
    "identity_hate": 0.0321
  },
  "flags": {
    "toxic": true,
    "severe_toxic": false,
    "obscene": false,
    "threat": false,
    "insult": true,
    "identity_hate": false
  },
  "is_toxic": true,
  "summary": "Flagged: toxic, insult",
  "top_score": 0.9341
}
```

### Batch prediction (up to 64 texts)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world!", "I hate you.", "Great article."]}'
```

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "uptime_seconds": 14.2,
  "threshold": 0.5
}
```

### Adjusting the threshold

Set `PREDICTION_THRESHOLD` to trade off precision vs recall:

```bash
PREDICTION_THRESHOLD=0.3 uvicorn app.api:app --reload
# Lower threshold → catches more toxicity, more false positives
```

---

## Training arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `./data` | Directory containing `train.csv` |
| `--output-dir` | `./checkpoints` | Where to save model + results |
| `--epochs` | `3` | Max training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `2e-5` | Learning rate for BERT layers |
| `--dropout` | `0.3` | Dropout on classifier head |
| `--patience` | `2` | Early stopping patience |
| `--sample-frac` | `1.0` | Fraction of data to use (e.g. `0.1` for quick runs) |

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests cover model architecture (output shape, sigmoid range, batch independence), dataset construction, inference pipeline consistency, and all API endpoints — without requiring a trained checkpoint or Kaggle data.

---

## Project structure

```
toxic-comment-classifier/
├── model/
│   ├── classifier.py     # ToxicClassifier (DistilBERT + head)
│   ├── dataset.py        # ToxicDataset + DataLoader factory
│   ├── train.py          # Training loop, early stopping, checkpointing
│   └── predict.py        # ToxicPredictor inference wrapper
├── app/
│   └── api.py            # FastAPI endpoints (/predict, /predict/batch, /health)
├── scripts/
│   └── generate_sample_data.py   # Synthetic data for CI/testing
├── notebooks/
│   └── training_walkthrough.ipynb  # EDA, training, evaluation, live inference
├── tests/
│   └── test_classifier.py  # Model, dataset, predictor, API tests
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

---

## Tech stack

| Component | Technology |
|---|---|
| Model | DistilBERT (`distilbert-base-uncased`) |
| Deep learning | PyTorch 2.x |
| Transformers | HuggingFace `transformers` |
| Metrics | scikit-learn (`roc_auc_score`) |
| API | FastAPI |
| Containerization | Docker |
| CI | GitHub Actions |
| Testing | pytest |

---

## Results (Kaggle dataset, 3 epochs, full data)

| Metric | Value |
|---|---|
| Val ROC-AUC (macro) | ~0.979 |
| Test ROC-AUC (macro) | ~0.977 |
| Training time (V100) | ~25 min |
| Training time (CPU) | ~6 hr |

> Results will vary with synthetic data — use the Kaggle dataset for real metrics.

---

## License

MIT
