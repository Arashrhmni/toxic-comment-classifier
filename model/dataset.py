from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


class ToxicDataset(Dataset):
    """
    PyTorch Dataset for the Jigsaw Toxic Comment dataset.

    Expects a DataFrame with columns:
        comment_text  - raw comment string
        toxic, severe_toxic, obscene, threat, insult, identity_hate - binary labels

    Tokenization is done once at construction time (fast tokenizer, pre-truncated).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = MAX_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        texts = df["comment_text"].tolist()
        self.labels = torch.tensor(df[LABELS].values, dtype=torch.float32)

        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_dataframes(data_dir: str, sample_frac: float = 1.0):
    """
    Load train.csv and split it into train, validation, and test DataFrames.

    The split is intentionally simple: 80% train, 10% validation, 10% test.
    The rows are shuffled first so that the split is not affected by the original CSV order.
    """
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be greater than 0 and less than or equal to 1.")

    data_path = Path(data_dir)
    train_path = data_path / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found in {data_dir}.\n"
            "Download the Kaggle data and place train.csv in the data folder, or run:\n"
            "python scripts/generate_sample_data.py"
        )

    df = pd.read_csv(train_path)
    required_columns = ["comment_text", *LABELS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"train.csv is missing required columns: {missing_columns}")

    # Shuffle before splitting. This avoids a biased train/validation/test split
    # if the CSV is sorted by label or by time.
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_NAME)

    train_ds = ToxicDataset(train_df, tokenizer)
    val_ds = ToxicDataset(val_df, tokenizer)
    test_ds = ToxicDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
