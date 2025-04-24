import json, os
from pathlib import Path
from huggingface_hub import HfApi, HfFolder

HF_DS = os.getenv("HF_DATASET_REPO", "waseemrazakhan/amazon-sentiment-dataset")
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HfFolder.save_token(TOKEN)

def main():
    api = HfApi()
    api.create_repo(repo_id=HF_DS, repo_type="dataset", exist_ok=True)

    data_dir = Path("data/datasets")
    # ensure metadata
    md = {
      "id": HF_DS,
      "licenses": [{"name":"CC0-1.0"}],
      "title": HF_DS.split("/")[-1].replace("-", " ").title()
    }
    (data_dir/"dataset-metadata.json").write_text(json.dumps(md,indent=2))

    # upload all CSVs
    for f in data_dir.glob("*.csv"):
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=HF_DS,
            repo_type="dataset",
            commit_message=f"update {f.name}"
        )
