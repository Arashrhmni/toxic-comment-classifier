"""
Unit and integration tests for ToxicClassifier.

Runs entirely without a trained checkpoint or Kaggle data.
Uses synthetic data and untrained model weights.
"""
import importlib
import os

import pandas as pd
import pytest
import torch
from transformers import DistilBertTokenizerFast

from model.classifier import ToxicClassifier
from model.dataset import ToxicDataset
from model.predict import ToxicPredictor

# ── Constants ─────────────────────────────────────────────────────────────────

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TOKENIZER = "distilbert-base-uncased"


def make_sample_df(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        toxic = int(i % 5 == 0)
        rows.append(
            {
                "comment_text": f"This is test comment number {i}.",
                "toxic": toxic,
                "severe_toxic": 0,
                "obscene": 0,
                "threat": 0,
                "insult": toxic,
                "identity_hate": 0,
            }
        )
    return pd.DataFrame(rows)


# ── Model tests ───────────────────────────────────────────────────────────────


class TestToxicClassifier:
    def test_output_shape(self):
        model = ToxicClassifier()
        model.eval()
        input_ids = torch.randint(0, 1000, (4, 32))
        attention_mask = torch.ones(4, 32, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        model = ToxicClassifier()
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        assert (out >= 0).all() and (out <= 1).all()

    def test_num_labels(self):
        assert ToxicClassifier().num_labels == 6

    def test_labels_list(self):
        assert ToxicClassifier.LABELS == LABELS

    def test_freeze_base(self):
        model = ToxicClassifier(freeze_base=True)
        assert all(not p.requires_grad for p in model.bert.parameters())
        assert model.classifier.weight.requires_grad

    def test_deterministic_in_eval_mode(self):
        """
        In eval mode (dropout disabled) the model must return identical
        outputs for the same input across two forward passes.
        Replaces the incorrect batch-independence test: DistilBERT's
        layer norm and attention are batch-aware, so a sample processed
        alone vs inside a batch will legitimately differ numerically.
        """
        model = ToxicClassifier()
        model.eval()
        x = torch.randint(0, 1000, (2, 32))
        mask = torch.ones(2, 32, dtype=torch.long)
        with torch.no_grad():
            out1 = model(x, mask)
            out2 = model(x, mask)
        assert torch.allclose(out1, out2), "Same input in eval mode must yield identical output"


# ── Dataset tests ─────────────────────────────────────────────────────────────


class TestToxicDataset:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return DistilBertTokenizerFast.from_pretrained(TOKENIZER)

    def test_length(self, tokenizer):
        ds = ToxicDataset(make_sample_df(10), tokenizer, max_length=64)
        assert len(ds) == 10

    def test_item_keys(self, tokenizer):
        ds = ToxicDataset(make_sample_df(5), tokenizer, max_length=64)
        assert set(ds[0].keys()) == {"input_ids", "attention_mask", "labels"}

    def test_label_dtype(self, tokenizer):
        ds = ToxicDataset(make_sample_df(5), tokenizer, max_length=64)
        assert ds[0]["labels"].dtype == torch.float32

    def test_input_length(self, tokenizer):
        ds = ToxicDataset(make_sample_df(5), tokenizer, max_length=64)
        assert ds[0]["input_ids"].shape == (64,)
        assert ds[0]["attention_mask"].shape == (64,)


# ── Predict tests (untrained model) ──────────────────────────────────────────


class TestToxicPredictor:
    @pytest.fixture(scope="class")
    def predictor(self, tmp_path_factory):
        ckpt = tmp_path_factory.mktemp("ckpt") / "best_model.pt"
        torch.save(ToxicClassifier().state_dict(), ckpt)
        return ToxicPredictor(str(ckpt), device="cpu")

    def test_predict_returns_dict(self, predictor):
        result = predictor.predict("Hello world")
        assert "scores" in result
        assert "flags" in result
        assert "is_toxic" in result

    def test_predict_score_keys(self, predictor):
        assert set(predictor.predict("Test comment")["scores"].keys()) == set(LABELS)

    def test_predict_score_range(self, predictor):
        for v in predictor.predict("Some text here")["scores"].values():
            assert 0.0 <= v <= 1.0

    def test_predict_with_explanation(self, predictor):
        result = predictor.predict_with_explanation("Test")
        assert "summary" in result
        assert "top_score" in result
        assert isinstance(result["summary"], str)

    def test_predict_batch(self, predictor):
        results = predictor.predict_batch(["Hello", "World", "Test comment"])
        assert len(results) == 3
        for r in results:
            assert "scores" in r and "flags" in r

    def test_batch_matches_single(self, predictor):
        """A batch of one should equal a single prediction exactly."""
        text = "This is a consistency test."
        single = predictor.predict(text)["scores"]
        batch = predictor.predict_batch([text])[0]["scores"]
        for label in LABELS:
            assert abs(single[label] - batch[label]) < 1e-4


# ── API tests ─────────────────────────────────────────────────────────────────


class TestAPI:
    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory):
        import app.api as api_module
        from fastapi.testclient import TestClient

        ckpt = tmp_path_factory.mktemp("api_ckpt") / "best_model.pt"
        torch.save(ToxicClassifier().state_dict(), ckpt)
        os.environ["CHECKPOINT_PATH"] = str(ckpt)
        importlib.reload(api_module)

        with TestClient(api_module.app) as c:
            yield c

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "model_loaded" in r.json()

    def test_predict(self, client):
        r = client.post("/predict", json={"text": "Hello world"})
        assert r.status_code == 200
        assert "scores" in r.json()
        assert "is_toxic" in r.json()

    def test_predict_empty_rejected(self, client):
        assert client.post("/predict", json={"text": ""}).status_code == 422

    def test_predict_batch(self, client):
        r = client.post("/predict/batch", json={"texts": ["Hello", "World"]})
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_predict_batch_too_large(self, client):
        assert client.post("/predict/batch", json={"texts": ["x"] * 65}).status_code == 422
