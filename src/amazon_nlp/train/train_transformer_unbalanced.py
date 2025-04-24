# File: src/amazon_nlp/train/train_transformer_unbalanced.py

#!/usr/bin/env python3
"""
Train the unbalanced transformer model variant.
"""

import logging
from amazon_nlp.train.base_transformer_trainer import BaseTransformerTrainer

def train_unbalanced():
    """
    Instantiate and run training for the 'unbalanced' variant.
    Returns a dict of evaluation metrics.
    """
    trainer = BaseTransformerTrainer(model_type="unbalanced")
    trainer.setup_logging()
    logging.info("Training unbalanced model…")

    # Load and prepare data
    df_train, df_val, df_test = trainer.load_datasets()
    train_ds, val_ds, test_ds = trainer.prepare_data(df_train, df_val, df_test)

    # Train and evaluate
    metrics = trainer.train(train_ds, val_ds, test_ds)
    logging.info("Unbalanced model metrics:")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    train_unbalanced()
