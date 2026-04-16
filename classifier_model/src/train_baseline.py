import os
import joblib
import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
VALID_LABELS = ["Left", "Center", "Right"]

# ── Load & Clean Data ─────────────────────────────────────────────────────────
def load_and_clean_data(path):
    df = pd.read_parquet(path)

    # Normalize labels
    df["bias"] = df["bias"].astype(str).str.strip().str.title()

    # Fix numeric labels if present
    bias_map = {
        "0": "Left", "1": "Center", "2": "Right",
        "0.0": "Left", "1.0": "Center", "2.0": "Right"
    }
    df["bias"] = df["bias"].replace(bias_map)

    # Keep valid labels only
    df = df[df["bias"].isin(VALID_LABELS)]

    # Fix flipped dataset issue
    df["bias"] = df["bias"].replace({"Left": "Right", "Right": "Left"})

    logger.info(f"Dataset size after cleaning: {len(df)}")
    logger.info(f"\nClass distribution:\n{df['bias'].value_counts()}")

    return df


# ── Balance Dataset ───────────────────────────────────────────────────────────
def balance_dataset(df):
    min_size = df["bias"].value_counts().min()

    df_balanced = (
        df.groupby("bias", group_keys=False)
        .apply(lambda x: x.sample(min_size, random_state=SEED))
        .reset_index(drop=True)
    )

    logger.info(f"\nBalanced class distribution:\n{df_balanced['bias'].value_counts()}")
    return df_balanced


# ── Train Model ───────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LinearSVC(class_weight="balanced", random_state=SEED))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


# ── Evaluate Model ────────────────────────────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    logger.info(f"\nTrain Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    logger.info(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    logger.info("\nClassification Report (Test Set):")
    logger.info("\n" + classification_report(y_test, y_test_pred))


# ── Save Model ────────────────────────────────────────────────────────────────
def save_model(model, base_dir):
    models_dir = os.path.join(base_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model, os.path.join(models_dir, "pipeline.pkl"))
    logger.info(f"\n[Saved] pipeline.pkl to {models_dir}")


# ── Manual Testing ────────────────────────────────────────────────────────────
def manual_test(model):
    logger.info("\n--- Manual Test Predictions ---")

    samples = [
        "The progressive policies proposed will ensure equality, universal healthcare, and stronger unions for workers.",
        "Taxes must be lowered to stimulate free market growth and protect individual liberties and constitutional rights.",
        "Bipartisan infrastructure bills are a sensible moderate approach to repairing roads while maintaining a balanced budget."
    ]

    preds = model.predict(samples)

    for text, pred in zip(samples, preds):
        logger.info(f"\nText: {text}\nPredicted Bias: {pred}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "cleaned_train.parquet")

    df = load_and_clean_data(data_path)
    df = balance_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["bias"],
        test_size=0.2,
        stratify=df["bias"],   # IMPORTANT improvement
        random_state=SEED
    )

    logger.info("\nTraining model...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_train, X_test, y_train, y_test)
    save_model(model, base_dir)
    manual_test(model)


if __name__ == "__main__":
    main()
