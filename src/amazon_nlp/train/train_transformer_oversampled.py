# File: src/amazon_nlp/train/train_transformer_oversampled.py

#!/usr/bin/env python3
"""
Train the oversampled transformer model variant.
"""

import logging
from amazon_nlp.train.base_transformer_trainer import BaseTransformerTrainer

def train_oversampled():
    """
    Instantiate and run training for the 'oversampled' variant.
    Returns a dict of evaluation metrics.
    """
    trainer = BaseTransformerTrainer(model_type="oversampled")
    trainer.setup_logging()
    logging.info("Training oversampled model…")

    # Load and prepare data
    df_train, df_val, df_test = trainer.load_datasets()
    train_ds, val_ds, test_ds = trainer.prepare_data(df_train, df_val, df_test)

    # Train and evaluate
    metrics = trainer.train(train_ds, val_ds, test_ds)
    logging.info("Oversampled model metrics:")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    train_oversampled()
