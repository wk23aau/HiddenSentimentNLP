#app2.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

app = Flask(__name__)
CORS(app)

# Adjust this path according to your local setup
model_dir = "models/bert-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = softmax(outputs.logits.numpy(), axis=1)
    prediction = int(torch.argmax(outputs.logits, dim=1).item())

    return jsonify({
        "prediction": prediction,
        "probabilities": probabilities.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
