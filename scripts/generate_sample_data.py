"""
Generates a synthetic train.csv for smoke-testing the pipeline
without downloading the full Kaggle dataset.

Usage:
    python scripts/generate_sample_data.py
    python -m model.train --data-dir ./data --epochs 1 --sample-frac 0.05
"""
import random
import pandas as pd
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

CLEAN_TEMPLATES = [
    "I really enjoyed reading this article, very informative.",
    "Can anyone help me understand how this works?",
    "Great point, I hadn't thought of it that way.",
    "This is a fascinating topic. I'd love to learn more.",
    "Thanks for sharing, this was really helpful.",
    "I disagree with this view but respect your opinion.",
    "The weather today is absolutely beautiful.",
    "Does anyone have recommendations for good books on this subject?",
    "I think we can all agree this is a complex issue.",
    "Looking forward to the next update on this project.",
]

TOXIC_TEMPLATES = [
    "You are completely stupid and wrong about everything.",
    "I hate people like you, you should just disappear.",
    "This is absolute garbage and so are you.",
    "What an idiot. Go back to school before commenting.",
    "You are the worst kind of person on this platform.",
]

OBSCENE_TEMPLATES = [
    "What the hell is wrong with you people.",
    "This is total bull and you know it.",
    "You're full of crap and your argument is garbage.",
]


def generate_row(idx: int) -> dict:
    roll = random.random()
    if roll < 0.85:
        text = random.choice(CLEAN_TEMPLATES) + f" (#{idx})"
        labels = [0, 0, 0, 0, 0, 0]
    elif roll < 0.93:
        text = random.choice(TOXIC_TEMPLATES) + f" (#{idx})"
        labels = [1, 0, 0, 0, 1, 0]
    elif roll < 0.97:
        text = random.choice(OBSCENE_TEMPLATES) + f" (#{idx})"
        labels = [1, 0, 1, 0, 0, 0]
    else:
        text = random.choice(TOXIC_TEMPLATES) + " " + random.choice(OBSCENE_TEMPLATES) + f" (#{idx})"
        labels = [1, 1, 1, 1, 1, 0]

    return {
        "id": f"synthetic_{idx:06d}",
        "comment_text": text,
        "toxic": labels[0],
        "severe_toxic": labels[1],
        "obscene": labels[2],
        "threat": labels[3],
        "insult": labels[4],
        "identity_hate": labels[5],
    }


def main(n: int = 2000, output_dir: str = "./data"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = [generate_row(i) for i in range(n)]
    df = pd.DataFrame(rows)
    out_path = Path(output_dir) / "train.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {n} synthetic rows → {out_path}")
    print(f"Label distribution:\n{df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].mean().round(3)}")


if __name__ == "__main__":
    main()
