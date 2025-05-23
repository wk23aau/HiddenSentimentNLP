#app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

app = Flask(__name__)

# Enable CORS for all routes (this solves the preflight CORS error)
CORS(app)

# Model setup
model_dir = "/content/drive/MyDrive/models/bert-sentiment/checkpoint-200"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = softmax(outputs.logits.numpy(), axis=1)
    pred = int(torch.argmax(outputs.logits, dim=1).item())

    return jsonify({"prediction": pred, "probabilities": probs.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
