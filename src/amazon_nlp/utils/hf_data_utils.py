# File: src/amazon_nlp/utils/hf_data_utils.py
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

HF_DS = os.getenv("HF_DATASET_REPO", "waseemrazakhan/amazon-sentiment-dataset")

def download_data():
    """Pull raw DB + CSVs from HF into data/database & data/datasets."""
    # make folders
    Path("data/database").mkdir(parents=True, exist_ok=True)
    Path("data/datasets").mkdir(parents=True, exist_ok=True)

    # raw DB
    hf_hub_download(
      repo_id=HF_DS, filename="amazon_scraper.db",
      repo_type="dataset", local_dir="data/database"
    )
    # CSVs
    for split in ("train","validation","test"):
        # picks the first matching file
        fname = next(p for p in os.listdir("/kaggle/input/amazon-sentiment-dataset")
                     if p.startswith(split))
        Path("data/datasets")\
          .joinpath(fname)\
          .write_bytes(Path("/kaggle/input/amazon-sentiment-dataset",fname).read_bytes())
