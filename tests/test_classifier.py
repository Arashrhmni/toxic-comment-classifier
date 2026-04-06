"""
Unit and integration tests for ToxicClassifier.

Runs entirely without a trained checkpoint or Kaggle data.
Uses synthetic data and untrained model weights.
"""
import pytest
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.classifier import ToxicClassifier
from model.dataset import ToxicDataset, make_loaders
from model.predict import ToxicPredictor
from transformers import DistilBertTokenizerFast


# ── Fixtures ──────────────────────────────────────────────────────────────────

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TOKENIZER = "distilbert-base-uncased"


def make_sample_df(n: int = 20) -> pd.DataFrame:
    import random
    random.seed(0)
    rows = []
    for i in range(n):
        toxic = int(i % 5 == 0)
        rows.append({
            "comment_text": f"This is test comment number {i}.",
            "toxic": toxic,
            "severe_toxic": 0,
            "obscene": 0,
            "threat": 0,
            "insult": toxic,
            "identity_hate": 0,
        })
    return pd.DataFrame(rows)


# ── Model tests ───────────────────────────────────────────────────────────────

class TestToxicClassifier:
    def test_output_shape(self):
        model = ToxicClassifier()
        input_ids = torch.randint(0, 1000, (4, 32))
        attention_mask = torch.ones(4, 32, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"

    def test_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        model = ToxicClassifier()
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        assert (out >= 0).all() and (out <= 1).all()

    def test_num_labels(self):
        model = ToxicClassifier()
        assert model.num_labels == 6

    def test_labels_list(self):
        assert ToxicClassifier.LABELS == LABELS

    def test_freeze_base(self):
        model = ToxicClassifier(freeze_base=True)
        bert_params = list(model.bert.parameters())
        assert all(not p.requires_grad for p in bert_params)
        # Classifier head should still be trainable
        assert model.classifier.weight.requires_grad

    def test_batch_independence(self):
        """Each sample in a batch should be independently processed."""
        model = ToxicClassifier()
        x = torch.randint(0, 1000, (3, 32))
        mask = torch.ones(3, 32, dtype=torch.long)
        with torch.no_grad():
            out_batch = model(x, mask)
            out_single = model(x[:1], mask[:1])
        assert torch.allclose(out_batch[0], out_single[0], atol=1e-5)


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestToxicDataset:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return DistilBertTokenizerFast.from_pretrained(TOKENIZER)

    def test_length(self, tokenizer):
        df = make_sample_df(10)
        ds = ToxicDataset(df, tokenizer, max_length=64)
        assert len(ds) == 10

    def test_item_keys(self, tokenizer):
        df = make_sample_df(5)
        ds = ToxicDataset(df, tokenizer, max_length=64)
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_label_dtype(self, tokenizer):
        df = make_sample_df(5)
        ds = ToxicDataset(df, tokenizer, max_length=64)
        assert ds[0]["labels"].dtype == torch.float32

    def test_input_length(self, tokenizer):
        df = make_sample_df(5)
        ds = ToxicDataset(df, tokenizer, max_length=64)
        assert ds[0]["input_ids"].shape == (64,)
        assert ds[0]["attention_mask"].shape == (64,)


# ── Predict tests (untrained model) ──────────────────────────────────────────

class TestToxicPredictor:
    @pytest.fixture(scope="class")
    def predictor(self, tmp_path_factory):
        """Save a random-weight checkpoint and load it."""
        tmp = tmp_path_factory.mktemp("ckpt")
        ckpt = tmp / "best_model.pt"
        model = ToxicClassifier()
        torch.save(model.state_dict(), ckpt)
        return ToxicPredictor(str(ckpt), device="cpu")

    def test_predict_returns_dict(self, predictor):
        result = predictor.predict("Hello world")
        assert "scores" in result
        assert "flags" in result
        assert "is_toxic" in result

    def test_predict_score_keys(self, predictor):
        result = predictor.predict("Test comment")
        assert set(result["scores"].keys()) == set(LABELS)

    def test_predict_score_range(self, predictor):
        result = predictor.predict("Some text here")
        for v in result["scores"].values():
            assert 0.0 <= v <= 1.0

    def test_predict_with_explanation(self, predictor):
        result = predictor.predict_with_explanation("Test")
        assert "summary" in result
        assert "top_score" in result
        assert isinstance(result["summary"], str)

    def test_predict_batch(self, predictor):
        texts = ["Hello", "World", "Test comment"]
        results = predictor.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "scores" in r and "flags" in r

    def test_batch_matches_single(self, predictor):
        text = "This is a consistency test."
        single = predictor.predict(text)["scores"]
        batch = predictor.predict_batch([text])[0]["scores"]
        for label in LABELS:
            assert abs(single[label] - batch[label]) < 1e-4


# ── API tests ─────────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory):
        from fastapi.testclient import TestClient
        import importlib
        import app.api as api_module

        # Patch checkpoint path to a real (random-weight) model
        tmp = tmp_path_factory.mktemp("api_ckpt")
        ckpt = tmp / "best_model.pt"
        torch.save(ToxicClassifier().state_dict(), ckpt)

        import os
        os.environ["CHECKPOINT_PATH"] = str(ckpt)

        # Reload module with patched env
        importlib.reload(api_module)

        with TestClient(api_module.app) as c:
            yield c

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert "model_loaded" in d

    def test_predict(self, client):
        r = client.post("/predict", json={"text": "Hello world"})
        assert r.status_code == 200
        d = r.json()
        assert "scores" in d
        assert "is_toxic" in d

    def test_predict_empty_rejected(self, client):
        r = client.post("/predict", json={"text": ""})
        assert r.status_code == 422

    def test_predict_batch(self, client):
        r = client.post("/predict/batch", json={"texts": ["Hello", "World"]})
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_predict_batch_too_large(self, client):
        r = client.post("/predict/batch", json={"texts": ["x"] * 65})
        assert r.status_code == 422
