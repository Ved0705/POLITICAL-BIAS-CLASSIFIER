import os
import logging
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3

# ── Load & preprocess data ────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_parquet(path)

    label_map = {"Left": 0, "Center": 1, "Right": 2}
    df["labels"] = df["bias"].map(label_map)

    df = df.dropna(subset=["labels"]).reset_index(drop=True)
    df["labels"] = df["labels"].astype(int)

    logger.info(f"Loaded dataset with {len(df)} rows")
    return df


def balance_dataset(df):
    min_size = df["labels"].value_counts().min()

    df_balanced = (
        df.groupby("labels", group_keys=False)
        .apply(lambda x: x.sample(min_size, random_state=SEED))
        .reset_index(drop=True)
    )

    logger.info(f"Balanced dataset size: {len(df_balanced)}")
    logger.info(f"\n{df_balanced['labels'].value_counts()}")
    return df_balanced


# ── Tokenization ──────────────────────────────────────────────────────────────
def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["content"], truncation=True, max_length=256)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["content"])
    return dataset


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "cleaned_train.parquet")
    output_dir = os.path.join(base_dir, "../models/saved_model")

    # Load + balance
    df = load_data(data_path)
    df = balance_dataset(df)

    # Convert to HF dataset
    dataset = Dataset.from_pandas(df[["content", "labels"]])
    dataset = dataset.train_test_split(test_size=0.2, seed=SEED)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # Tokenize
    train_ds = tokenize_dataset(dataset["train"], tokenizer)
    test_ds = tokenize_dataset(dataset["test"], tokenizer)

    # Dynamic padding (faster than padding=True earlier)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    logger.info("Final evaluation results:")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
