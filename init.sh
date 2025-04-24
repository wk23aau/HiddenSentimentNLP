#!/usr/bin/env bash
#
# init.sh — scaffold the amazon_nlp project from scratch
#
set -e

echo "🌱 Initializing new NLP project…"

# 1. Create top-level directories
mkdir -p data/raw data/database data/processed data/datasets
mkdir -p extension/web-extension
mkdir -p server notebooks tests docs
mkdir -p src/amazon_nlp/config src/amazon_nlp/utils \
         src/amazon_nlp/preprocessing src/amazon_nlp/train \
         src/amazon_nlp/inference src/amazon_nlp/kaggle

# 2. Create __init__.py in each Python package
for pkg in src/amazon_nlp src/amazon_nlp/config src/amazon_nlp/utils \
           src/amazon_nlp/preprocessing src/amazon_nlp/train \
           src/amazon_nlp/inference src/amazon_nlp/kaggle; do
  touch "$pkg/__init__.py"
done

# 3. README.md
cat > README.md << 'EOF'
# Amazon Sentiment NLP

End-to-end pipeline for scraping, cleaning, annotating, training, and serving
sentiment analysis models on Amazon reviews.
EOF

# 4. .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
env/
myenv/
.DS_Store
data/*
!data/raw/
!data/processed/
!data/datasets/
logs/
models/
extension/web-extension/*.zip
.env
EOF

# 5. .env.example
cat > .env.example << 'EOF'
GOOGLE_API_KEY=
MODEL_ID=gemini-2.0-flash
BATCH_SIZE=20
MAX_RETRIES=5
INITIAL_WAIT=2
HUGGINGFACE_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
EOF

# 6. requirements.txt
cat > requirements.txt << 'EOF'
pandas
scikit-learn
tqdm
google-generativeai
python-dotenv
transformers
huggingface-hub
torch
psutil
imblearn
tenacity
fastapi
uvicorn
kaggle
click
EOF

# 7. setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="amazon_nlp",
    version="0.1.0",
    description="End-to-end Amazon sentiment NLP pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "scikit-learn",
        "tqdm",
        "google-generativeai",
        "python-dotenv",
        "transformers",
        "huggingface-hub",
        "torch",
        "psutil",
        "imblearn",
        "tenacity",
        "fastapi",
        "uvicorn",
        "kaggle",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "amazon-nlp=amazon_nlp.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
EOF

# 8. CLI entry-point
cat > src/amazon_nlp/cli.py << 'EOF'
import os
import click
import uvicorn

from amazon_nlp.preprocessing.db_cleanup import main as clean_db
from amazon_nlp.preprocessing.annotate_reviews import main as annotate
from amazon_nlp.preprocessing.prepare_datasets import main as prepare
from amazon_nlp.train.train_all_transformers import train_all_models
from amazon_nlp.kaggle.push_to_kaggle import main as push_models
from amazon_nlp.inference.predict import main as predict_main

@click.group()
def cli():
    """Amazon NLP end-to-end pipeline CLI."""

@cli.command()
def clean_database():
    """Clean the raw SQLite database."""
    clean_db()

@cli.command()
def annotate_reviews():
    """Annotate reviews using Gemini."""
    annotate()

@cli.command()
def prepare_data():
    """Prepare train/val/test CSV datasets."""
    prepare()

@cli.command()
def train():
    """Train transformer models."""
    train_all_models()

@cli.command()
def push_models():
    """Push models to the Hugging Face hub."""
    push_models()

@cli.command()
def predict():
    """Run example predictions."""
    predict_main()

@cli.command()
def serve():
    """Start the FastAPI server."""
    uvicorn.run("server.app:app", reload=True)

if __name__ == "__main__":
    cli()
EOF

# 9. Server stub
cat > server/app.py << 'EOF'
from fastapi import FastAPI
from amazon_nlp.inference.sentiment_predictor import SentimentPredictor
import os

app = FastAPI()
model_dir = os.getenv("MODEL_OUTPUT_DIR", "models/unbalanced")
predictor = SentimentPredictor(model_dir, model_type="transformer")

@app.post("/analyze")
async def analyze(text: str):
    return predictor.predict(text)
EOF

# 10. Browser extension stubs
cat > extension/web-extension/manifest.json << 'EOF'
{
  "manifest_version": 2,
  "name": "Amazon Sentiment Annotator",
  "version": "0.1",
  "content_scripts": [
    {
      "matches": ["*://*.amazon.com/*"],
      "js": ["content.js"]
    }
  ]
}
EOF
touch extension/web-extension/content.js

echo "✅ Scaffold complete!
Next steps:
  1) pip install -e .
  2) amazon-nlp --help
"
