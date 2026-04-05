"""
Downloads the political bias dataset from Hugging Face and saves
train/test/valid splits as Parquet files in src/data/.

Dataset: cajcodes/political-bias
  - columns: 'text', 'label'  (0=Left, 1=Center, 2=Right)

We rename columns to match what the training scripts expect:
  - 'text'  -> 'content'
  - 'label' -> 'bias'  (mapped to string: Left / Center / Right)
"""

import os
import pandas as pd
from datasets import load_dataset

# ── destination folder ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

LABEL_MAP = {0: "Left", 1: "Center", 2: "Right"}

print("⬇️  Downloading dataset from Hugging Face (cajcodes/political-bias)…")
ds = load_dataset("cajcodes/political-bias")
print(f"✅ Downloaded. Splits available: {list(ds.keys())}")
print(f"   Columns: {ds[list(ds.keys())[0]].column_names}")

# ── helper: convert one split to a cleaned DataFrame ────────────────────────
def to_df(split):
    df = split.to_pandas()

    # rename 'text' -> 'content'
    if "text" in df.columns:
        df = df.rename(columns={"text": "content"})

    # rename / map 'label' -> 'bias'
    if "label" in df.columns:
        df["bias"] = df["label"].map(LABEL_MAP).fillna(df["label"].astype(str))
        df = df.drop(columns=["label"])

    # keep only what the training scripts use
    keep = [c for c in ["content", "bias"] if c in df.columns]
    df = df[keep].dropna()
    df = df[df["content"].str.strip() != ""]
    return df.reset_index(drop=True)


# ── figure out which Hugging Face splits map to train / test / valid ─────────
split_keys = list(ds.keys())

# fallback: if there's only a "train" split, carve out test & valid manually
if set(split_keys) >= {"train", "test", "validation"}:
    splits = {
        "train": to_df(ds["train"]),
        "test":  to_df(ds["test"]),
        "valid": to_df(ds["validation"]),
    }
elif set(split_keys) >= {"train", "test"}:
    splits = {
        "train": to_df(ds["train"]),
        "test":  to_df(ds["test"]),
        "valid": to_df(ds["test"]),   # reuse test as valid
    }
else:
    # only "train" — split manually
    full_df = to_df(ds["train"])
    train_df = full_df.sample(frac=0.8,  random_state=42)
    rest_df  = full_df.drop(train_df.index)
    test_df  = rest_df.sample(frac=0.5,  random_state=42)
    valid_df = rest_df.drop(test_df.index)
    splits = {"train": train_df, "test": test_df, "valid": valid_df}

# ── save ─────────────────────────────────────────────────────────────────────
for name, df in splits.items():
    out = os.path.join(DATA_DIR, f"{name}.parquet")
    df.to_parquet(out, index=False)
    print(f"💾 Saved {name} split → {out}  ({len(df):,} rows, columns: {list(df.columns)})")

print("\n✅ Dataset ready! Now run:")
print("   python src/data_prep.py")
print("   python src/train_baseline.py")
