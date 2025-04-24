# File: src/amazon_nlp/inference/sentiment_predictor.py

import os
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentPredictor:
    """
    Unified interface for logistic‐regression and transformer models.
    """

    def __init__(self, model_dir, model_type="transformer"):
        """
        model_dir: path to local model folder (or HF repo ID if using transformer)
        model_type: "lr" for scikit‐learn, "transformer" for HF models
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        if self.model_type == "lr":
            # scikit‐learn pipeline
            self.model      = joblib.load(os.path.join(self.model_dir, "model.joblib"))
            self.vectorizer = joblib.load(os.path.join(self.model_dir, "vectorizer.joblib"))
        else:
            # Hugging Face transformer
            self.model     = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()

    def predict(self, text):
        """
        Predict sentiment for a single review.
        Returns: {"sentiment": "positive"/"negative", "confidence": float, "scores": {...}}
        """
        if self.model_type == "lr":
            X = self.vectorizer.transform([text])
            probs = self.model.predict_proba(X)[0]
        else:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        label = "positive" if probs[1] > probs[0] else "negative"
        return {
            "sentiment": label,
            "confidence": float(max(probs)),
            "scores": {"negative": float(probs[0]), "positive": float(probs[1])}
        }
