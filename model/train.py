"""
Training script for the ToxicClassifier.

Usage:
    python -m model.train --data-dir ./data --epochs 3 --batch-size 32 --lr 2e-5

For a quick smoke-test on synthetic data:
    python scripts/generate_sample_data.py
    python -m model.train --data-dir ./data --epochs 1 --sample-frac 0.1
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from model.classifier import ToxicClassifier
from model.dataset import load_dataframes, make_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_probs, average="macro")
    except ValueError:
        auc = float("nan")

    return total_loss / len(loader), auc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading data...")
    train_df, val_df, test_df = load_dataframes(args.data_dir, sample_frac=args.sample_frac)
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    model = ToxicClassifier(dropout=args.dropout, freeze_base=args.freeze_base).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Give rare labels a higher weight during training.
    # This is calculated from the training data instead of being hard-coded.
    positive_counts = torch.tensor(train_df[ToxicClassifier.LABELS].sum().values, dtype=torch.float32)
    negative_counts = len(train_df) - positive_counts
    pos_weight = (negative_counts / positive_counts.clamp(min=1)).clamp(max=20).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bert_params = [param for param in model.bert.parameters() if param.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer_groups = [{"params": head_params, "lr": args.lr * 10}]
    if bert_params:
        optimizer_groups.insert(0, {"params": bert_params, "lr": args.lr})

    optimizer = AdamW(optimizer_groups, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | {elapsed:.1f}s"
        )

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}
        )

        if np.isnan(val_auc):
            logger.info("Validation AUC is not available for this split; using validation loss instead.")
            improved = not history[:-1] or val_loss < min(item["val_loss"] for item in history[:-1])
        else:
            improved = val_auc > best_auc

        if improved:
            if not np.isnan(val_auc):
                best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"  ✓ Saved new best checkpoint. Best AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    logger.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test AUC: {test_auc:.4f}")

    metadata = {
        "best_val_auc": best_auc,
        "test_auc": test_auc,
        "epochs_trained": len(history),
        "history": history,
        "args": vars(args),
    }
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Artifacts saved to {output_dir}/")
    return metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data", help="Directory with train.csv")
    p.add_argument("--output-dir", default="./checkpoints", help="Where to save model + results")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    p.add_argument(
        "--sample-frac", type=float, default=1.0, help="Fraction of data to use (for quick tests)"
    )
    p.add_argument(
        "--freeze-base",
        action="store_true",
        help="Train only the final classifier layer. Useful for quick CPU demos.",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
