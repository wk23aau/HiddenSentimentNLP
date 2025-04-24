# File: src/amazon_nlp/train/train_transformer_undersampled.py

#!/usr/bin/env python3
"""
Train the undersampled transformer model variant.
"""

import logging
from amazon_nlp.train.base_transformer_trainer import BaseTransformerTrainer

def train_undersampled():
    """
    Instantiate and run training for the 'undersampled' variant.
    Returns a dict of evaluation metrics.
    """
    trainer = BaseTransformerTrainer(model_type="undersampled")
    trainer.setup_logging()
    logging.info("Training undersampled model…")

    # Load and prepare data
    df_train, df_val, df_test = trainer.load_datasets()
    train_ds, val_ds, test_ds = trainer.prepare_data(df_train, df_val, df_test)

    # Train and evaluate
    metrics = trainer.train(train_ds, val_ds, test_ds)
    logging.info("Undersampled model metrics:")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    train_undersampled()
