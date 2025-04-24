# File: src/amazon_nlp/config/model_config.py

import os

class ModelConfig:
    """Configuration settings for transformer model training."""
    # Base Hugging Face model
    MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
    # Maximum sequence length for tokenization
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
    # Training hyperparameters
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
    EPOCHS = int(os.getenv("EPOCHS", 3))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 2))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 500))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 2))
    # Directory where trained models are saved
    OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "models/")
