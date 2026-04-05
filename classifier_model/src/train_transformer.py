import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# ── Load cleaned data ────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "data", "cleaned_train.parquet")
df = pd.read_parquet(data_path)

# Always recompute numeric 'labels' from 'bias' to avoid stale/NaN values
label_map = {"Left": 0, "Center": 1, "Right": 2}
df['labels'] = df['bias'].map(label_map)

# Drop rows where bias wasn't in the map
df = df.dropna(subset=['labels']).reset_index(drop=True)
df['labels'] = df['labels'].astype(int)
print(f"✅ Loaded {len(df)} rows. Label distribution:\n{df['labels'].value_counts()}\n")

# ── Build HuggingFace Dataset (only content + labels needed) ─────────────────
hf_dataset = Dataset.from_pandas(df[['content', 'labels']])
hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
train_ds = hf_dataset['train']
test_ds  = hf_dataset['test']

# ── Model & Tokenizer ─────────────────────────────────────────────────────────
model_name = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# ── Tokenization ──────────────────────────────────────────────────────────────
def tokenize(batch):
    return tokenizer(batch['content'], padding=True, truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

# Remove the raw text column — model only needs input_ids, attention_mask, labels
train_ds = train_ds.remove_columns(['content'])
test_ds  = test_ds.remove_columns(['content'])

print(f"Train columns: {train_ds.column_names}")   # should include 'labels'
print(f"Test  columns: {test_ds.column_names}\n")

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ── Training arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=os.path.join(os.path.dirname(__file__), "../models/saved_model"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ── Train & evaluate ──────────────────────────────────────────────────────────
trainer.train()
results = trainer.evaluate()
print("\n🏁 Final evaluation results:")
for k, v in results.items():
    print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
