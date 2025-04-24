# File: src/amazon_nlp/config/kaggle_config.py

import os
from pathlib import Path

class KaggleConfig:
    """
    Configuration settings for Kaggle dataset integration.
    """
    # Your Kaggle username and API key (from .env or env vars)
    USERNAME = os.getenv("KAGGLE_USERNAME")
    KEY      = os.getenv("KAGGLE_KEY")

    # Dataset identifier on Kaggle, e.g. "username/amazon-sentiment"
    DATASET_NAME = os.getenv("KAGGLE_DATASET_NAME", f"{USERNAME}/amazon-sentiment")

    # Where Kaggle mounts the input dataset in a Notebook
    INPUT_DIR  = Path(os.getenv("KAGGLE_INPUT_DIR", f"/kaggle/input/{DATASET_NAME}"))

    # Where to write outputs (models, logs) in a Kaggle run
    OUTPUT_DIR = Path(os.getenv("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
