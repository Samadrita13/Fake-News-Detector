import joblib
import re

# ---------------------------
# Text cleaning function
# (must match training)
# ---------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Load model and vectorizer
# ---------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------------------
# Take user input
# ---------------------------
news = input("Enter news text: ")

# Clean input
news_cleaned = clean_text(news)

# Transform
news_vector = vectorizer.transform([news_cleaned])

# Predict
prediction = model.predict(news_vector)[0]
probability = model.predict_proba(news_vector)[0][prediction]

# ---------------------------
# Output result
# ---------------------------
if prediction == 0:
    print("\nPrediction: FAKE")
else:
    print("\nPrediction: REAL")

print("Confidence: {:.2f}%".format(probability * 100))