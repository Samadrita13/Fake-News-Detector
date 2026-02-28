import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("data/news_cleaned.csv")

# Clean dataset
data = data.dropna(subset=["text", "label"])
data = data[data["text"].str.strip() != ""]
data["label"] = data["label"].astype(int)

print("Dataset size:", len(data))
print(data["label"].value_counts())

# Split FIRST (very important)
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]  # keeps class balance
)

# Vectorize AFTER splitting
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7,min_df=5,ngram_range=(1,2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000,solver="liblinear")
model.fit(X_train_vec, y_train)

# Train accuracy (overfitting check)
train_pred = model.predict(X_train_vec)
print("\nTrain Accuracy:", accuracy_score(y_train, train_pred))

# Predict
y_pred = model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
      
# Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")