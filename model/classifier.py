import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class ToxicClassifier(nn.Module):
    """
    Multi-label toxic comment classifier built on DistilBERT.

    Predicts 6 toxicity categories simultaneously:
        toxic, severe_toxic, obscene, threat, insult, identity_hate

    Architecture:
        DistilBERT (frozen or fine-tuned) → [CLS] pooling → Dropout → Linear → Sigmoid
    """

    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(self, dropout: float = 0.3, freeze_base: bool = False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(self.LABELS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return torch.sigmoid(logits)  # multi-label: independent probabilities

    @property
    def num_labels(self) -> int:
        return len(self.LABELS)
