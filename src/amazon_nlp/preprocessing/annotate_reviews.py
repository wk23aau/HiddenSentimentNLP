#!/usr/bin/env python3
"""
amazon_nlp/preprocessing/annotate_reviews.py

Annotates Amazon product reviews with binary sentiment (positive/negative)
using Gemini 2.0 API, asking the model to return 0 or 1 labels in JSON.
"""

import os
import sys
import json
import time
import logging
import sqlite3
import re
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# ─── Ensure console can emit UTF-8 ───────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY      = os.getenv("GENERATIVE_API_KEY")
MODEL_ID     = os.getenv("MODEL_ID",        "gemini-2.0-flash")
BATCH_SIZE   = int(os.getenv("BATCH_SIZE",  "20"))
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "5"))
WAIT_S       = int(os.getenv("INITIAL_WAIT","2"))

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR     = BASE_DIR / "data"
DB_DIR       = DATA_DIR / "database"
DB_PATH      = DB_DIR / "amazon_scraper_cleaned.db"
LOGS_DIR     = BASE_DIR / "logs"

for d in (DB_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "annotation.log", encoding="utf-8"),
    ]
)

# ─── Bootstrap prints ───────────────────────────────────────────────────────────
print(f"Base Directory: {BASE_DIR}")
print(f"Database Path: {DB_PATH}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Model ID: {MODEL_ID}")

def get_db_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS review_sentiments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      asin TEXT NOT NULL,
      review_text TEXT NOT NULL,
      review_title TEXT,
      rating REAL,
      sentiment TEXT CHECK(sentiment IN ('positive','negative')),
      processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    logging.info("Available tables: %s", tables)

def verify_database():
    conn = get_db_connection()
    try:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM products WHERE reviews IS NOT NULL"
        ).fetchone()[0]
        logging.info("Found %d products with reviews", cnt)
        if cnt == 0:
            raise ValueError("No products with reviews found")
    finally:
        conn.close()

def get_unprocessed_reviews(conn, asin):
    done_texts = {
        r[0] for r in conn.execute(
            "SELECT review_text FROM review_sentiments WHERE asin=?", (asin,)
        )
    }
    row = conn.execute(
        "SELECT reviews FROM products WHERE asin=?", (asin,)
    ).fetchone()
    if not row or not row["reviews"]:
        return []

    try:
        payload = json.loads(row["reviews"])
    except json.JSONDecodeError:
        logging.error("Invalid JSON for ASIN %s", asin)
        return []

    out = []
    for pool in ("all", "critical"):
        for r in payload.get(pool, []):
            txt = r.get("body", "").strip()
            if txt and txt not in done_texts:
                out.append({
                    "text":  txt[:30000],
                    "title": r.get("title", "")[:1000],
                    "rating": r.get("rating")
                })
    return out

def _strip_markdown(text: str) -> str:
    # remove leading/trailing ```json fences if present
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=WAIT_S, min=WAIT_S, max=30)
)
def get_sentiment_batch(model, reviews):
    """
    Ask Gemini to return a JSON object:
      {"labels": [0,1, ...]}
    where 1 = positive, 0 = negative.
    """
    payload = [{"rating": rv["rating"], "text": rv["text"]} for rv in reviews]
    prompt = f"""
You are a sentiment classifier.  Given the JSON input below,
classify each review as either negative (0) or positive (1).

OUTPUT RULES:
- Return valid JSON and nothing else.
- JSON must be an object with a single key "labels".
- "labels" value must be an array of integers, exactly {len(reviews)} elements.
- 1 = positive, 0 = negative.

Here is the input JSON:
{json.dumps({"reviews": payload}, indent=2)}
""".strip()

    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "top_p":      1.0,
            "max_output_tokens": 1024,
        }
    )
    raw = _strip_markdown(resp.text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logging.error("JSON decode error: %s\nModel output:\n%s", e, resp.text)
        return None

    labels = data.get("labels")
    if (
        isinstance(labels, list)
        and len(labels) == len(reviews)
        and all(isinstance(x, int) and x in (0, 1) for x in labels)
    ):
        return labels
    else:
        logging.error("Invalid JSON shape or labels: %r", data)
        return None

def process_asin(conn, asin):
    reviews = get_unprocessed_reviews(conn, asin)
    if not reviews:
        return 0

    logging.info("Processing %d reviews for ASIN %s", len(reviews), asin)
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_ID)
    cur = conn.cursor()
    inserted = 0

    for i in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[i : i + BATCH_SIZE]
        try:
            raw_labels = get_sentiment_batch(model, batch)
        except RetryError as e:
            logging.error("Retries exhausted on ASIN %s: %s", asin, e)
            return inserted

        if raw_labels is None:
            logging.error("Skipping ASIN %s after parse failures", asin)
            return inserted

        # map 1→positive, 0→negative
        sentiments = [
            "positive" if lab == 1 else "negative"
            for lab in raw_labels
        ]

        for rv, sent in zip(batch, sentiments):
            cur.execute("""
                INSERT INTO review_sentiments
                  (asin, review_text, review_title, rating, sentiment)
                VALUES (?, ?, ?, ?, ?)
            """, (asin, rv["text"], rv["title"], rv["rating"], sent))
            inserted += 1

        conn.commit()
        logging.info("Inserted %d/%d for ASIN %s",
                     inserted, len(reviews), asin)
        time.sleep(WAIT_S)

    return inserted

def get_next_asin(conn):
    row = conn.execute("""
        SELECT p.asin
        FROM products p
        LEFT JOIN review_sentiments rs ON p.asin=rs.asin
        WHERE p.reviews IS NOT NULL
          AND p.reviews!='{\"all\": [], \"critical\": []}'
          AND p.scrape_status='completed'
        GROUP BY p.asin
        HAVING COUNT(rs.id)=0
        LIMIT 1
    """).fetchone()
    return row[0] if row else None

def print_final_stats(conn):
    r = conn.execute("""
        SELECT
          COUNT(DISTINCT asin)            as products,
          COUNT(*)                        as reviews,
          SUM(sentiment='positive')       as positive,
          SUM(sentiment='negative')       as negative,
          AVG(sentiment='positive')*100   as pos_pct
        FROM review_sentiments
    """).fetchone()
    logging.info(
        "Final Stats: %d products, %d reviews – %d positive (%.1f%%), %d negative (%.1f%%)",
        r["products"], r["reviews"],
        r["positive"], r["pos_pct"], r["negative"], 100 - r["pos_pct"]
    )

def main():
    logging.info("Starting sentiment annotation with Gemini 2.0")
    logging.info("DB: %s | Batch size: %d", DB_PATH, BATCH_SIZE)

    try:
        verify_database()
        conn = get_db_connection()
        init_db(conn)

        total = 0
        while True:
            asin = get_next_asin(conn)
            if not asin:
                logging.info("No more unprocessed ASINs")
                break
            count = process_asin(conn, asin)
            total += count
            if count:
                logging.info("Completed ASIN %s: %d reviews", asin, count)

    except Exception as e:
        logging.error("Fatal error: %s", e)

    finally:
        if 'conn' in locals():
            print_final_stats(conn)
            conn.close()

if __name__ == "__main__":
    main()