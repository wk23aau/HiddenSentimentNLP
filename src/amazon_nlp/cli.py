# File: src/amazon_nlp/cli.py

import os
from dotenv import load_dotenv

# Load environment variables early so configs pick them up
load_dotenv()

import click
import uvicorn

@click.group()
def cli():
    """Amazon NLP end-to-end pipeline CLI."""

@cli.command()
def clean_database():
    """Clean the raw SQLite database."""
    from amazon_nlp.preprocessing.db_cleanup import main as clean_db
    clean_db()

@cli.command()
def annotate_reviews():
    """Annotate reviews using Gemini-2.0 Flash."""
    from amazon_nlp.preprocessing.annotate_reviews import main as annotate
    annotate()

@cli.command()
def prepare_data():
    """Prepare train/validation/test CSV datasets."""
    from amazon_nlp.preprocessing.prepare_datasets import main as prepare
    prepare()

@cli.command()
def train():
    """Train all transformer model variants."""
    from amazon_nlp.train.train_all_transformers import main as train_all_models
    train_all_models()

@cli.command()
def push_models():
    """Push datasets to Kaggle and models to Hugging Face."""
    from amazon_nlp.kaggle.push_to_kaggle import main as push_models
    push_models()
@cli.command()
def download_data():
    """Fetch raw DB &/or CSVs from the HF dataset into data/."""
    from amazon_nlp.utils.hf_data_utils import download_data
    download_data()

@cli.command()
def push_datasets():
    """Push CSV train/val/test back to the HF dataset repo."""
    from amazon_nlp.scripts.push_datasets_to_hf import main as push_ds
    push_ds()

@cli.command()
def predict():
    """Run example sentiment predictions."""
    from amazon_nlp.inference.predict import main as predict_main
    predict_main()

@cli.command()
def serve():
    """Start the FastAPI server."""
    uvicorn.run("server.app:app", reload=True)

if __name__ == "__main__":
    cli()
