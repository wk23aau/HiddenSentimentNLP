# compare_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

app = Flask(__name__)
CORS(app)

# ----- Load Models and Tokenizers -----
# Load BERT model & tokenizer (assumes fine-tuned for sentiment classification)
bert_model_dir = "models/bert-sentiment"
tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
bert_model.eval()

# Load logistic regression model and its TF-IDF vectorizer via joblib
lr_model = load("D:\HiddenSentimentNLP\models\logistic-regression\logistic_regression_model.joblib")
tfidf = load("D:\HiddenSentimentNLP\models\logistic-regression\lr_tfidf_vectorizer.joblib")

# Label mapping for clarity
label_map = {0: "Negative", 1: "Positive"}

# ----- Single-Review Comparison Endpoint -----
@app.route('/compare', methods=['POST'])
def compare():
    """
    Expects JSON: { "text": "Review text goes here..." }
    Returns JSON with side-by-side predictions from BERT and Logistic Regression.
    Example response:
    {
      "bert": { "label": "Positive", "confidence": 92.3 },
      "lr":   { "label": "Negative", "confidence": 71.9 }
    }
    """
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # --- BERT Inference ---
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits.numpy()[0]  # For binary classification, shape is [2]
    probs = softmax(logits)
    pred_bert = int(logits.argmax())
    conf_bert = float(probs.max() * 100.0)

    # --- Logistic Regression Inference ---
    X_vec = tfidf.transform([text])
    lr_probs = lr_model.predict_proba(X_vec)[0]
    pred_lr = int(lr_probs.argmax())
    conf_lr = float(lr_probs.max() * 100.0)

    return jsonify({
        "bert": {"label": label_map[pred_bert], "confidence": conf_bert},
        "lr":   {"label": label_map[pred_lr], "confidence": conf_lr}
    })

# ----- Batch Evaluation Endpoint -----
@app.route('/batch-compare', methods=['POST'])
def batch_compare():
    """
    Expects JSON: {
      "samples": [
         { "text": "Review text...", "label": 1 },
         { "text": "Another review...", "label": 0 },
         ...
      ]
    }
    Returns a JSON object with:
      - metrics_bert: Evaluation metrics for BERT.
      - metrics_lr: Evaluation metrics for Logistic Regression.
      - predictions: Array of per-sample results including ground truth and model predictions.
    """
    data = request.get_json()
    samples = data.get("samples", [])
    if not samples:
        return jsonify({"error": "No samples provided"}), 400

    y_true = []
    bert_preds = []
    lr_preds = []
    predictions = []

    for sample in samples:
        text = sample.get("text", "").strip()
        true_label = sample.get("label", None)
        if not text or true_label is None:
            continue  # Skip invalid samples

        y_true.append(true_label)

        # --- BERT Inference ---
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        logits = outputs.logits.numpy()[0]
        probs = softmax(logits)
        pred_bert = int(logits.argmax())
        conf_bert = float(probs.max() * 100.0)
        bert_preds.append(pred_bert)

        # --- Logistic Regression Inference ---
        X_vec = tfidf.transform([text])
        lr_probs = lr_model.predict_proba(X_vec)[0]
        pred_lr = int(lr_probs.argmax())
        conf_lr = float(lr_probs.max() * 100.0)
        lr_preds.append(pred_lr)

        predictions.append({
            "text": text,
            "true_label": true_label,
            "bert_pred": pred_bert,
            "bert_confidence": conf_bert,
            "lr_pred": pred_lr,
            "lr_confidence": conf_lr
        })

    # ----- Compute Metrics for BERT -----
    cm_bert = confusion_matrix(y_true, bert_preds).tolist()  # Convert to list for JSON serialization
    accuracy_bert = accuracy_score(y_true, bert_preds)
    precision_bert, recall_bert, f1_bert, _ = precision_recall_fscore_support(
        y_true, bert_preds, average='binary', zero_division=0
    )
    metrics_bert = {
        "confusion_matrix": cm_bert,
        "accuracy": accuracy_bert,
        "precision": precision_bert,
        "recall": recall_bert,
        "f1": f1_bert
    }

    # ----- Compute Metrics for Logistic Regression -----
    cm_lr = confusion_matrix(y_true, lr_preds).tolist()
    accuracy_lr = accuracy_score(y_true, lr_preds)
    precision_lr, recall_lr, f1_lr, _ = precision_recall_fscore_support(
        y_true, lr_preds, average='binary', zero_division=0
    )
    metrics_lr = {
        "confusion_matrix": cm_lr,
        "accuracy": accuracy_lr,
        "precision": precision_lr,
        "recall": recall_lr,
        "f1": f1_lr
    }

    return jsonify({
        "metrics_bert": metrics_bert,
        "metrics_lr": metrics_lr,
        "predictions": predictions
    })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
