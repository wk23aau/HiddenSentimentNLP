from flask import Flask, request, jsonify
import sqlite3
import json
import os
from flask_cors import CORS

# Path Configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DB_DIR = os.path.join(DATA_DIR, 'database')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Create necessary directories
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# File paths
DATABASE = os.path.join(DB_DIR, 'amazon_scraper.db')
QUEUE_FILE = os.path.join(BASE_DIR, 'queue.json')
SCHEMA_FILE = os.path.join(BASE_DIR, 'schema.sql')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

SCRAPING_STATE = "stopped"

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db_exists = os.path.exists(DATABASE)
    conn = get_db_connection()

    if not db_exists:
        print(f"Creating database schema at {DATABASE}...")
        try:
            with open(SCHEMA_FILE, 'r') as f:
                conn.executescript(f.read())

            if os.path.exists(QUEUE_FILE):
                with open(QUEUE_FILE, 'r') as f:
                    queue_data = json.load(f)
                    cursor = conn.cursor()
                    for asin in queue_data.get('asinQueue', []):
                        cursor.execute("INSERT OR IGNORE INTO products (asin, scrape_status) VALUES (?, ?)",
                                    (asin, 'pending'))
                    conn.commit()
                    print(f"Initialized database with ASINs from queue")
            
            # Create a backup of the empty database
            backup_file = os.path.join(DB_DIR, 'amazon_scraper_backup.db')
            with open(DATABASE, 'rb') as source:
                with open(backup_file, 'wb') as target:
                    target.write(source.read())
            print(f"Created backup of empty database at {backup_file}")

        except Exception as e:
            print(f"Error initializing database: {str(e)}")
        finally:
            conn.close()

@app.route('/api/asins', methods=['GET'])
def get_asins():
    if SCRAPING_STATE not in ["running", "resumed"]:
        return jsonify([])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT asin FROM products WHERE scrape_status = 'pending' LIMIT 20")
    asins = [row['asin'] for row in cursor.fetchall()]
    conn.close()
    return jsonify(asins)

@app.route('/api/scrape-result', methods=['POST', 'OPTIONS'])
def receive_scrape_result():
    if request.method == 'OPTIONS':
        return _build_cors_response()

    data = request.json
    if not data or 'asin' not in data:
        return jsonify({'error': 'Invalid data: asin missing'}), 400

    asin = data['asin']
    product_data = data.get('productData', {})
    partial_scrape = data.get('partial', False)

    # Save raw data
    raw_data_file = os.path.join(RAW_DIR, f'{asin}_raw.json')
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(product_data, f, ensure_ascii=False, indent=2)

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if 'error' in product_data:
            cursor.execute(
                "UPDATE products SET scrape_status = 'error', error_message = ? WHERE asin = ?",
                (product_data['error'], asin)
            )
        elif product_data:
            if partial_scrape:
                cursor.execute("""
                    UPDATE products SET
                        scrape_status = 'completed',
                        reviews = ?,
                        last_scraped = CURRENT_TIMESTAMP
                    WHERE asin = ?
                """, (
                    json.dumps(product_data.get('reviews')),
                    asin
                ))
            else:
                cursor.execute("""
                    UPDATE products SET
                        scrape_status = 'completed',
                        product_title = ?,
                        categories = ?,
                        brand = ?,
                        pricing = ?,
                        best_seller_rank = ?,
                        specs = ?,
                        features = ?,
                        details = ?,
                        buy_box = ?,
                        available_deals = ?,
                        reviews = ?,
                        all_offers = ?,
                        monthly_sales = ?,
                        last_scraped = CURRENT_TIMESTAMP
                    WHERE asin = ?
                """, (
                    product_data.get('title'),
                    json.dumps(product_data.get('categories')),
                    product_data.get('brand'),
                    json.dumps(product_data.get('pricing')),
                    product_data.get('bestSellerRank'),
                    json.dumps(product_data.get('specs')),
                    json.dumps(product_data.get('features')),
                    json.dumps(product_data.get('details')),
                    json.dumps(product_data.get('buyBox')),
                    product_data.get('availableDeals'),
                    json.dumps(product_data.get('reviews')),
                    json.dumps(product_data.get('offers')),
                    product_data.get('monthlySales'),
                    asin
                ))
        conn.commit()
        return jsonify({'status': 'success'})

    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

    finally:
        conn.close()

@app.route('/api/start-scrape', methods=['POST'])
def start_scrape():
    global SCRAPING_STATE
    SCRAPING_STATE = "running"
    return jsonify({'status': 'success'})

@app.route('/api/pause-scrape', methods=['POST'])
def pause_scrape():
    global SCRAPING_STATE
    SCRAPING_STATE = "paused"
    return jsonify({'status': 'success'})

@app.route('/api/resume-scrape', methods=['POST'])
def resume_scrape():
    global SCRAPING_STATE
    SCRAPING_STATE = "resumed"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE products SET scrape_status = 'pending' WHERE scrape_status = 'error'")
    conn.commit()
    conn.close()
    
    return jsonify({'status': 'success'})

@app.route('/api/stop-scrape', methods=['POST'])
def stop_scrape():
    global SCRAPING_STATE
    SCRAPING_STATE = "stopped"
    return jsonify({'status': 'success'})

@app.route('/api/scrape-status', methods=['GET'])
def get_scrape_status():
    return jsonify({'status': SCRAPING_STATE})

@app.route('/api/add-asins', methods=['POST'])
def add_asins():
    data = request.json
    asins = data.get('asins', [])
    if not asins:
        return jsonify({'status': 'error', 'message': 'No ASINs provided'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    added_count = 0
    try:
        for asin in asins:
            cursor.execute("INSERT OR IGNORE INTO products (asin, scrape_status) VALUES (?, 'pending')", (asin,))
            if cursor.rowcount > 0:
                added_count += 1
        conn.commit()
        return jsonify({'status': 'success', 'added_count': added_count})
    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()

def _build_cors_response():
    response = jsonify({'message': 'CORS preflight successful'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

if __name__ == '__main__':
    print(f"Data Directory: {DATA_DIR}")
    print(f"Database path: {DATABASE}")
    print(f"Schema file: {SCHEMA_FILE}")
    print(f"Queue file: {QUEUE_FILE}")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Processed data directory: {PROCESSED_DIR}")
    
    init_db()
    app.run(debug=True)