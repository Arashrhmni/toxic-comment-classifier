"""
Inference utilities for ToxicClassifier.

Handles single texts, batches, and threshold tuning.
"""

import torch
from transformers import DistilBertTokenizerFast

from model.classifier import ToxicClassifier

TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
DEFAULT_THRESHOLD = 0.5


class ToxicPredictor:
    """
    Lightweight inference wrapper. Loads a checkpoint and exposes
    predict() and predict_batch() with configurable thresholds.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_NAME)
        self.model = ToxicClassifier()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def _tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict toxicity for a list of texts.

        Returns list of dicts with:
            - scores: {label: probability}
            - flags: {label: bool} (True if score >= threshold)
            - is_toxic: bool (any flag is True)
        """
        enc = self._tokenize(texts)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for prob_row in probs:
            scores = {
                label: round(float(p), 4) for label, p in zip(ToxicClassifier.LABELS, prob_row)
            }
            flags = {label: bool(p >= self.threshold) for label, p in scores.items()}
            results.append(
                {
                    "scores": scores,
                    "flags": flags,
                    "is_toxic": any(flags.values()),
                }
            )
        return results

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_with_explanation(self, text: str) -> dict:
        result = self.predict(text)
        active = [label for label, flagged in result["flags"].items() if flagged]
        top_score = max(result["scores"].values())

        summary = "Clean" if not result["is_toxic"] else f"Flagged: {', '.join(active)}"
        result["summary"] = summary
        result["top_score"] = round(top_score, 4)
        return result
