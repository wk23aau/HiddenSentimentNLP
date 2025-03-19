from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from joblib import load
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

# --- 1. Load BERT model + tokenizer ---
bert_model_dir = "models/bert-sentiment"
tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
bert_model.eval()

# --- 2. Load logistic regression model + vectorizer ---
lr_model_path = "models/logistic-regression/logistic_regression_model.joblib"
tfidf_path    = "models/logistic-regression/lr_tfidf_vectorizer.joblib"

lr_model = load(lr_model_path)
tfidf_vectorizer = load(tfidf_path)

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # --- BERT inference ---
    bert_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**bert_inputs)
    logits_bert = outputs.logits.numpy()[0]  # shape [2] for binary classification
    probs_bert = softmax(logits_bert)
    pred_bert = int(logits_bert.argmax())
    confidence_bert = float(probs_bert.max() * 100.0)

    # --- Logistic Regression inference ---
    X_vec = tfidf_vectorizer.transform([text])    # transform text with TF-IDF
    lr_probs = lr_model.predict_proba(X_vec)[0]   # shape [2]
    pred_lr = int(lr_probs.argmax())
    confidence_lr = float(lr_probs.max() * 100.0)

    label_map = {0: "Negative", 1: "Positive"}
    return jsonify({
        "bert": {
            "label": label_map[pred_bert],
            "confidence": confidence_bert
        },
        "lr": {
            "label": label_map[pred_lr],
            "confidence": confidence_lr
        }
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
