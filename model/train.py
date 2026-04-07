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

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import roc_auc_score
import numpy as np

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
        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)
        loss.backward()

        # Gradient clipping — important for BERT fine-tuning stability
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
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)
        total_loss += loss.item()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Mean column-wise ROC-AUC (the Kaggle competition metric)
    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except ValueError:
        auc = float("nan")  # can happen with very small batches if a class has 0 positives

    return total_loss / len(loader), auc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    logger.info("Loading data...")
    train_df, val_df, test_df = load_dataframes(args.data_dir, sample_frac=args.sample_frac)
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    # Model
    model = ToxicClassifier(dropout=args.dropout).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss: BCELoss with pos_weight to handle class imbalance
    # Toxic comments are ~10% of the dataset
    pos_weight = torch.tensor([10.0] * model.num_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer: lower LR for BERT layers, higher for classifier head
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW(
        [{"params": bert_params, "lr": args.lr}, {"params": head_params, "lr": args.lr * 10}],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    # Training loop
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

        # Checkpoint best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"  ✓ New best AUC: {best_auc:.4f} — saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Final evaluation on test set
    logger.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test AUC: {test_auc:.4f}")

    # Save training metadata
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
    p.add_argument("--sample-frac", type=float, default=1.0, help="Fraction of data to use (for quick tests)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
