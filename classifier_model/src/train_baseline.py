import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Load cleaned train Parquet
df = pd.read_parquet("src\data\cleaned_train.parquet")
print(df.columns)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['bias'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train LinearSVC
clf = LinearSVC()
clf.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))
