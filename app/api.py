"""
FastAPI inference server for ToxicClassifier.

Endpoints:
    GET  /health          — liveness + readiness probe
    POST /predict         — single text prediction
    POST /predict/batch   — up to 64 texts
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./checkpoints/best_model.pt")
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
_start = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(CHECKPOINT_PATH)
    if not checkpoint.exists():
        logger.warning(
            f"No checkpoint at {CHECKPOINT_PATH}. "
            "Run `python -m model.train` first. "
            "Prediction endpoints will return 503 until a model is available."
        )
        app.state.predictor = None
    else:
        from model.predict import ToxicPredictor
        logger.info(f"Loading model from {CHECKPOINT_PATH} ...")
        app.state.predictor = ToxicPredictor(CHECKPOINT_PATH, threshold=THRESHOLD)
        logger.info("Model ready.")
    yield


app = FastAPI(
    title="Toxic Comment Classifier",
    description=(
        "Multi-label toxicity detection fine-tuned on DistilBERT. "
        "Predicts 6 toxicity categories with per-label confidence scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip() for t in v]
        if any(not t for t in cleaned):
            raise ValueError("Batch contains empty strings.")
        return cleaned


class ScoreMap(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float


class FlagMap(BaseModel):
    toxic: bool
    severe_toxic: bool
    obscene: bool
    threat: bool
    insult: bool
    identity_hate: bool


class PredictResponse(BaseModel):
    scores: ScoreMap
    flags: FlagMap
    is_toxic: bool
    summary: str
    top_score: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    threshold: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _predictor(request: Request):
    p = request.app.state.predictor
    if p is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first: python -m model.train",
        )
    return p


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.predictor is not None,
        uptime_seconds=round(time.time() - _start, 1),
        threshold=THRESHOLD,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    """Classify a single comment and return per-label scores + a human-readable summary."""
    predictor = _predictor(request)
    return predictor.predict_with_explanation(body.text)


@app.post("/predict/batch")
async def predict_batch(body: BatchPredictRequest, request: Request):
    """Classify up to 64 comments in a single request."""
    predictor = _predictor(request)
    results = predictor.predict_batch(body.texts)
    return {"results": results, "count": len(results)}
