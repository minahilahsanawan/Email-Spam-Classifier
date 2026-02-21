from flask import Flask, render_template, request, jsonify
from spam_classifier import classifier
app = Flask(__name__)
def ensure_model_loaded():
    # Load existing model; if not found, train and save once
    if not classifier.load_model():
        classifier.train_model()
        classifier.save_model()
# Load/train at startup
ensure_model_loaded()
@app.get("/")
def home():
    return render_template("index.html")
@app.post("/api/classify")
def api_classify():
    data = request.get_json(silent=True) or {}
    email_text = str(data.get("email", "")).strip()
    if not email_text:
        return jsonify({"error": "Please enter some email text!"}), 400
    # Safety: if server restarted and model not loaded for any reason
    if not getattr(classifier, "is_trained", False):
        ensure_model_loaded()
    result = classifier.predict(email_text)
    ham = float(result["probabilities"]["Ham"]) * 100.0
    spam = float(result["probabilities"]["Spam"]) * 100.0
    confidence = float(result["confidence"]) * 100.0
    return jsonify({
        "prediction": result["prediction"],            # "Spam" or "Ham"
        "confidence": confidence,                      # percent
        "ham_probability": ham,                        # percent
        "spam_probability": spam                       # percent
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)