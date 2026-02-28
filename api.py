from flask import Flask, request, jsonify
import joblib
import re

# -----------------------
# Text Cleaning Function
# -----------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# Load model and vectorizer
# -----------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Fake News Detector API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Please provide 'text' field in JSON"}), 400

        text = clean_text(data["text"])

        text_vector = vectorizer.transform([text])

        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0][prediction]

        return jsonify({
            "prediction": "REAL" if prediction == 1 else "FAKE",
            "confidence": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)