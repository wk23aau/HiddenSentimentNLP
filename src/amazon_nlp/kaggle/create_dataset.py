# File: src/amazon_nlp/kaggle/create_dataset.py

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from amazon_nlp.config.kaggle_config import KaggleConfig

def main():
    """
    Publish the CSVs under data/datasets/ as a new Kaggle Dataset
    or version if it already exists.
    """
    cfg = KaggleConfig
    api = KaggleApi()
    api.authenticate()

    data_path = Path.cwd() / "data" / "datasets"
    if not data_path.exists():
        raise FileNotFoundError(f"No datasets found at {data_path}")

    slug = cfg.DATASET_NAME.split("/")[-1]  # e.g. "amazon-sentiment"
    owner = cfg.USERNAME                     # your Kaggle username

    # If the dataset doesn't exist yet, create it; otherwise create a new version.
    try:
        api.dataset_create_new(
            path=str(data_path),
            slug=slug,
            convert_to_csv=True,
            dir_mode="zip",
            title="Amazon Sentiment Dataset",
            description="Sentiment‐labeled Amazon reviews (positive/negative)."
        )
        print(f"Created new Kaggle dataset {owner}/{slug}")
    except Exception:
        # Fallback to version update
        api.dataset_create_version(
            path=str(data_path),
            version_notes="Update with latest CSVs",
            convert_to_csv=True,
            delete_old_versions=True,
            dir_mode="zip"
        )
        print(f"Updated Kaggle dataset {owner}/{slug}")

if __name__ == "__main__":
    main()
