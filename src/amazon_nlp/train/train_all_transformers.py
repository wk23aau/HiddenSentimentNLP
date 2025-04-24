# File: src/amazon_nlp/train/train_all_transformers.py

#!/usr/bin/env python3
"""
Train all transformer model variants in sequence:
- unbalanced
- undersampled
- oversampled
"""

import logging
from datetime import datetime

from amazon_nlp.train.train_transformer_unbalanced import train_unbalanced
from amazon_nlp.train.train_transformer_undersampled import train_undersampled
from amazon_nlp.train.train_transformer_oversampled import train_oversampled

def setup_logging():
    """Configure root logger to file + console with timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/all_transformers_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logging.info(f"Logging to {log_file}")

def main():
    setup_logging()
    logging.info("Starting training of all transformer variants…")

    results = {}
    try:
        logging.info("→ Training unbalanced model")
        results["unbalanced"] = train_unbalanced()

        logging.info("→ Training undersampled model")
        results["undersampled"] = train_undersampled()

        logging.info("→ Training oversampled model")
        results["oversampled"] = train_oversampled()

        logging.info("All trainings complete. Summary:")
        for variant, metrics in results.items():
            logging.info(f"--- {variant} ---")
            for name, value in metrics.items():
                logging.info(f"{name}: {value:.4f}")
    except Exception as e:
        logging.error(f"Error during full training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
