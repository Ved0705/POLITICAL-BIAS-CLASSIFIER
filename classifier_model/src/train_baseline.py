import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Load cleaned train Parquet
data_path = os.path.join(os.path.dirname(__file__), "data", "cleaned_train.parquet")
df = pd.read_parquet(data_path)

print("Columns:", df.columns.tolist())

# 1. Fix label issues
df['bias'] = df['bias'].astype(str).str.strip().str.title()
# Map any residual numeric string labels just in case
bias_map = {'0': 'Left', '1': 'Center', '2': 'Right', '0.0': 'Left', '1.0': 'Center', '2.0': 'Right'}
df['bias'] = df['bias'].replace(bias_map)

# Keep only the valid labels
valid_labels = ['Left', 'Center', 'Right']
df = df[df['bias'].isin(valid_labels)]

# THE DATASET MAINTAINER FLIPPED THE LABELS! 
# "Left" in the raw data contains right-wing text (low taxes, military) 
# and "Right" in the raw data contains left-wing text (unions, worker rights).
# We MUST flip them here so the model learns the correct definitions.
df['bias'] = df['bias'].replace({'Left': 'Right', 'Right': 'Left'})

# 2. Add label validation
print("\nUnique labels before balancing:")
print(df['bias'].unique())

# 5. Add debugging outputs (class distribution)
print("\nClass distribution before balancing:")
print(df['bias'].value_counts())

# 3. Balance the dataset
min_class_size = df['bias'].value_counts().min()
df = df.groupby('bias').sample(n=min_class_size, random_state=42)

print("\nClass distribution after balancing:")
print(df['bias'].value_counts())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['bias'], test_size=0.2, random_state=42
)

# 4. Improve model training
print("\nTraining model...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LinearSVC(class_weight="balanced", random_state=42)
clf.fit(X_train_vec, y_train)

# 5. Debugging outputs (Train and Test accuracy)
y_train_pred = clf.predict(X_train_vec)
y_test_pred = clf.predict(X_test_vec)

print(f"\nTrain Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Save model and vectorizer for the API backend
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(clf, os.path.join(models_dir, "model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
print(f"\n💾 Saved model.pkl and vectorizer.pkl to {models_dir}")

# 6. Add manual test predictions
print("\n--- Manual Test Predictions ---")
test_samples = [
    "The progressive policies proposed will ensure equality, universal healthcare, and stronger unions for workers.",
    "Taxes must be lowered to stimulate free market growth and protect individual liberties and constitutional rights.",
    "Bipartisan infrastructure bills are a sensible moderate approach to repairing roads while maintaining a balanced budget."
]

test_vecs = vectorizer.transform(test_samples)
predictions = clf.predict(test_vecs)

for text, pred in zip(test_samples, predictions):
    print(f"Text: '{text}'")
    print(f"Predicted Bias: {pred}\n")
