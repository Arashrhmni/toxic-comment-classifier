import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
import pandas as pd
from pathlib import Path


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


class ToxicDataset(Dataset):
    """
    PyTorch Dataset for the Jigsaw Toxic Comment dataset.

    Expects a DataFrame with columns:
        comment_text  — raw comment string
        toxic, severe_toxic, obscene, threat, insult, identity_hate  — binary labels

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
    Load Jigsaw train/test CSVs from data_dir.

    Args:
        data_dir: path containing train.csv (and optionally test.csv, test_labels.csv)
        sample_frac: fraction of training data to use (useful for quick iteration)

    Returns:
        train_df, val_df, test_df
    """
    data_path = Path(data_dir)
    train_path = data_path / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found in {data_dir}.\n"
            "Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data\n"
            "Or run: python scripts/generate_sample_data.py  (creates synthetic data for testing)"
        )

    df = pd.read_csv(train_path)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # 80/10/10 split
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
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:  # noqa: UP006
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
