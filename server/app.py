from fastapi import FastAPI
from amazon_nlp.inference.sentiment_predictor import SentimentPredictor
import os

app = FastAPI()
model_dir = os.getenv("MODEL_OUTPUT_DIR", "models/unbalanced")
predictor = SentimentPredictor(model_dir, model_type="transformer")

@app.post("/analyze")
async def analyze(text: str):
    return predictor.predict(text)
