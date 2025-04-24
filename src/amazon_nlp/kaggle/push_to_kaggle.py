# File: src/amazon_nlp/kaggle/push_to_kaggle.py

#!/usr/bin/env python3
"""
Push the processed CSVs under data/datasets/ to your Kaggle Dataset.
If the dataset doesn’t exist yet, it creates one (using dataset-metadata.json);
otherwise it publishes a new version.
"""

import logging
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from amazon_nlp.config.kaggle_config import KaggleConfig

# ─── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("push_to_kaggle")

def main():
    # Verify credentials
    if not KaggleConfig.USERNAME or not KaggleConfig.KEY:
        logger.error("KAGGLE_USERNAME and KAGGLE_KEY must be set")
        return

    data_path = Path.cwd() / "data" / "datasets"
    if not data_path.exists() or not list(data_path.glob("*.csv")):
        logger.error("No CSV files found in %s", data_path)
        return

    owner, slug = KaggleConfig.DATASET_NAME.split("/")
    logger.info("Pushing %s → %s/%s", data_path, owner, slug)

    # 1) Create dataset-metadata.json if missing
    md_file = data_path / "dataset-metadata.json"
    if not md_file.exists():
        metadata = {
            "title": slug.replace("-", " ").title(),
            "id": KaggleConfig.DATASET_NAME,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(md_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Wrote metadata file %s", md_file.name)

    api = KaggleApi()
    api.authenticate()

    try:
        # Create a new dataset (will read dataset-metadata.json)
        api.dataset_create_new(
            str(data_path),
            convert_to_csv=True,
            dir_mode="zip"
        )
        logger.info("Created new Kaggle dataset: %s/%s", owner, slug)
    except Exception as create_err:
        logger.warning("Create failed (maybe exists): %s", create_err)
        try:
            # Publish a new version
            api.dataset_create_version(
                str(data_path),
                version_notes="Automated update of CSVs",
                convert_to_csv=True,
                delete_old_versions=True,
                dir_mode="zip"
            )
            logger.info("Published new version: %s/%s", owner, slug)
        except Exception as version_err:
            logger.error("Version publish failed: %s", version_err)

if __name__ == "__main__":
    main()
