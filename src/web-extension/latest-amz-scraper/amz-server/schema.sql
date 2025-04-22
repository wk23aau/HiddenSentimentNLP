DROP TABLE IF EXISTS products;

CREATE TABLE products (
    asin TEXT PRIMARY KEY,
    scrape_status TEXT NOT NULL DEFAULT 'pending',
    brand TEXT,
    product_title TEXT,
    specs JSON,
    details JSON,
    features JSON,
    buy_box JSON,
    pricing JSON,
    package_contents JSON,
    categories JSON,
    comparison JSON,
    ratings JSON,
    available_deals TEXT,
    reviews JSON,
    all_offers JSON,
    monthly_sales TEXT,
    best_seller_rank TEXT,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_scraped TIMESTAMP
);