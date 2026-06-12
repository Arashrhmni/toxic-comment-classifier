# Toxic Comment Classifier

A beginner-friendly NLP project that classifies comments into six toxicity categories using **DistilBERT**, PyTorch, and FastAPI.

The project is intentionally kept simple: one model, one training script, one prediction wrapper, and one API.

---

## What the project does

The model reads a comment and returns a score for each label:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Because this is a **multi-label** problem, one comment can belong to more than one category at the same time.

Example:

```json
{
  "toxic": 0.91,
  "insult": 0.84,
  "threat": 0.03
}
```

This means the model can say: “this comment is toxic and insulting, but probably not a threat.”

---

## Why this project is useful

This project demonstrates the most important parts of a small machine-learning application:

- Loading and preparing text data
- Tokenizing text with a transformer tokenizer
- Fine-tuning DistilBERT for classification
- Saving and loading a trained model
- Running predictions from Python
- Serving predictions through a FastAPI API
- Testing the model, dataset, predictor, and API
- Running the app with Docker

---

## Simple architecture

```text
Input comment
    ↓
DistilBERT tokenizer
    ↓
DistilBERT model
    ↓
Dropout layer
    ↓
Linear classifier
    ↓
Six toxicity scores
```

The model returns raw logits during training. During prediction, the logits are converted into probabilities using sigmoid.

---

## Project structure

```text
toxic-comment-classifier/
├── app/
│   └── api.py                  # FastAPI app
├── model/
│   ├── classifier.py           # DistilBERT model
│   ├── dataset.py              # Dataset and DataLoader code
│   ├── predict.py              # Prediction helper
│   └── train.py                # Training script
├── scripts/
│   └── generate_sample_data.py # Creates small synthetic data
├── tests/
│   └── test_classifier.py      # Unit and API tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For tests and linting:

```bash
pip install -r requirements-dev.txt
```

---

## Option A: Run with sample data

This is the easiest way to test the project without a Kaggle account.

```bash
python scripts/generate_sample_data.py
python -m model.train --data-dir ./data --epochs 1 --batch-size 8 --freeze-base
```

This creates:

```text
checkpoints/best_model.pt
checkpoints/training_results.json
```

The sample data is only for testing the code. It is not meant to produce a real production-quality model.

---

## Option B: Train with the real Kaggle dataset

Download the Jigsaw Toxic Comment Classification dataset from Kaggle and place `train.csv` inside the `data/` folder.

Then run:

```bash
python -m model.train --data-dir ./data --epochs 3 --batch-size 32
```

A GPU is recommended for real training.

---

## Run the API

After training, start the API:

```bash
uvicorn app.api:app --reload
```

Open the API docs in your browser:

```text
http://localhost:8000/docs
```

---

## Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are completely wrong and an idiot."}'
```

Example response:

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

---

## Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world!", "I hate you.", "Great article."]}'
```

---

## Health check

```bash
curl http://localhost:8000/health
```

Example:

```json
{
  "status": "ok",
  "model_loaded": true,
  "uptime_seconds": 14.2,
  "threshold": 0.5
}
```

---

## Change the prediction threshold

The default threshold is `0.5`.

```bash
PREDICTION_THRESHOLD=0.3 uvicorn app.api:app --reload
```

Lower threshold: catches more toxic comments, but may create more false positives.

Higher threshold: fewer false positives, but may miss some toxic comments.

---

## Run with Docker

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000/docs
```

Make sure `checkpoints/best_model.pt` exists before using the prediction endpoints.

---

## Run tests

```bash
pytest tests/ -v
```

Run the linter:

```bash
ruff check model/ app/ tests/ scripts/
```

---

## Training arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--data-dir` | `./data` | Folder containing `train.csv` |
| `--output-dir` | `./checkpoints` | Folder for saved model files |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `32` | Number of comments per batch |
| `--lr` | `2e-5` | Learning rate |
| `--dropout` | `0.3` | Dropout before the final classifier |
| `--patience` | `2` | Early stopping patience |
| `--sample-frac` | `1.0` | Use only part of the data |
| `--freeze-base` | `False` | Train only the final classifier head |

---

## Notes

This is a learning project, not a finished moderation system. A real moderation system would need stronger evaluation, bias checks, human review, monitoring, and clear rules for how predictions are used.
