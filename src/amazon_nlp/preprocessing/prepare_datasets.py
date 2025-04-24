#!/usr/bin/env python3
# File: src/amazon_nlp/preprocessing/prepare_datasets.py

"""
Reads the cleaned & annotated SQLite DB and creates train/validation/test
CSV datasets (unbalanced, oversampled, undersampled) under data/datasets/.
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from dotenv import load_dotenv

def main():
    load_dotenv()
    BASE_DIR      = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR      = BASE_DIR / "data"
    DB_CLEANED    = DATA_DIR / "database" / "amazon_scraper_cleaned.db"
    DATASETS_DIR  = DATA_DIR / "datasets"
    PROCESSED_DIR = DATA_DIR / "processed"
    LOGS_DIR      = BASE_DIR / "logs"

    for d in (DATASETS_DIR, PROCESSED_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    TEST_SIZE    = float(os.getenv("TEST_SIZE", "0.2"))
    VAL_SIZE     = float(os.getenv("VALIDATION_SIZE", "0.1"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

    log_file = LOGS_DIR / f"prepare_datasets_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file))
        ]
    )
    logging.info("Starting dataset preparation")
    logging.info(f"Cleaned DB path: {DB_CLEANED}")

    if not DB_CLEANED.exists():
        logging.error("Cleaned DB not found at %s", DB_CLEANED)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_CLEANED))
    df = pd.read_sql("SELECT review_text AS text, sentiment FROM review_sentiments", conn)
    conn.close()
    logging.info("Loaded %d annotated reviews", len(df))

    # Encode labels
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
    df["text_clean"] = (
        df["text"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["text_clean"] != ""]
    logging.info("After dropping empty texts: %d rows", len(df))

    # Unbalanced split
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_frac,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )
    logging.info(
        "Splits (train/val/test): %d / %d / %d",
        len(train_df), len(val_df), len(test_df)
    )

    # Oversample & undersample
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)

    X_train = train_df[["text_clean"]]
    y_train = train_df["label"]
    X_over, y_over = ros.fit_resample(X_train, y_train)
    X_under, y_under = rus.fit_resample(X_train, y_train)

    over_df  = pd.DataFrame({"text_clean": X_over["text_clean"], "label": y_over})
    under_df = pd.DataFrame({"text_clean": X_under["text_clean"], "label": y_under})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {
        f"train_unbalanced_{timestamp}.csv": train_df[["text_clean", "label"]],
        f"train_oversampled_{timestamp}.csv": over_df,
        f"train_undersampled_{timestamp}.csv": under_df,
        f"validation_{timestamp}.csv": val_df[["text_clean", "label"]],
        f"test_{timestamp}.csv": test_df[["text_clean", "label"]],
    }

    for fname, subdf in outputs.items():
        path = DATASETS_DIR / fname
        subdf.to_csv(path, index=False)
        logging.info("Wrote %s (%d rows)", path.relative_to(BASE_DIR), len(subdf))

    logging.info("Dataset preparation complete. CSVs in %s", DATASETS_DIR)

if __name__ == "__main__":
    main()
