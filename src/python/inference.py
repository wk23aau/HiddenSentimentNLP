import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_dir):
    """
    Loads the tokenizer and model from the specified directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    """
    Tokenizes the input text and returns the sentiment prediction.
    Assumes a binary classification: 1 (positive) or 0 (negative).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

if __name__ == "__main__":
    # Use the first argument as the model directory (default: models/bert-sentiment)
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/bert-sentiment"
    # Use remaining arguments as the text to predict; otherwise use a default sample
    sample_text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "The product is amazing and exceeded my expectations!"

    tokenizer, model = load_model(model_dir)
    prediction = predict_sentiment(sample_text, tokenizer, model)
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Input: {sample_text}")
    print(f"Predicted sentiment: {sentiment}")
