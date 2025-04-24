# File: src/amazon_nlp/preprocessing/db_cleanup.py

#!/usr/bin/env python3
"""
amazon_nlp/preprocessing/db_cleanup.py

Cleans the raw SQLite database of scraped Amazon reviews and exports
the products table to JSON.
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from dotenv import load_dotenv

# ─── Load environment & configure paths ─────────────────────────────────────────
load_dotenv()
BASE_DIR        = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR        = BASE_DIR / "data"
DATABASE_DIR    = DATA_DIR / "database"
DATABASE_PATH   = DATABASE_DIR / "amazon_scraper.db"
CLEANED_DB_PATH = DATABASE_DIR / "amazon_scraper_cleaned.db"
PRODUCTS_JSON   = DATABASE_DIR / "amazon_products.json"
LOGS_DIR        = BASE_DIR / "logs"

# Ensure directories exist
for d in (DATABASE_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging Configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "db_cleanup.log")
    ]
)

def verify_database():
    """Ensure the raw scraped DB exists."""
    if not DATABASE_PATH.exists():
        logging.error(f"Database not found at: {DATABASE_PATH}")
        raise FileNotFoundError(f"Database not found at: {DATABASE_PATH}")
    logging.info(f"Found database at: {DATABASE_PATH}")

def get_db_connection(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection to the given path."""
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

def fetch_products(conn: sqlite3.Connection) -> list[dict]:
    """
    Fetch all products (and their raw review JSON) from the DB.
    Returns a list of dicts.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE reviews IS NOT NULL")
    cols = [c[0] for c in cursor.description]
    rows = cursor.fetchall()
    products = [dict(zip(cols, row)) for row in rows]
    logging.info(f"Fetched {len(products)} products with reviews")
    return products

def save_cleaned_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    """
    Make a cleaned copy of the DB file (currently a straight copy).
    Returns a fresh connection to the original DB.
    """
    conn.close()
    DATABASE_PATH_str = str(DATABASE_PATH)
    CLEANED_str       = str(CLEANED_DB_PATH)
    # Copy raw DB to cleaned DB path
    with open(DATABASE_PATH_str, "rb") as src, open(CLEANED_str, "wb") as dst:
        dst.write(src.read())
    logging.info(f"Saved cleaned database to: {CLEANED_DB_PATH}")
    # Re-open the original DB
    return get_db_connection()

def export_products_json(products: list[dict]):
    """
    Export the fetched products list to JSON for inspection or external use.
    """
    with open(PRODUCTS_JSON, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)
    logging.info(f"Exported {len(products)} products to JSON: {PRODUCTS_JSON}")

def main():
    """Entry point: verify, clean DB, and export JSON."""
    logging.info("Starting database cleanup…")
    verify_database()

    conn = get_db_connection()
    try:
        products = fetch_products(conn)
        conn = save_cleaned_db(conn)
        export_products_json(products)
        logging.info("Database cleanup complete.")
    except Exception as e:
        logging.error(f"Fatal error during cleanup: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
