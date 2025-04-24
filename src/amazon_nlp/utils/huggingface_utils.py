# File: src/amazon_nlp/utils/huggingface_utils.py

import os
import logging
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class HuggingFaceManager:
    """
    Helper for listing, verifying, uploading, and loading models
    on the Hugging Face Hub.
    """
    def __init__(self, token=None):
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("HUGGINGFACE_TOKEN not set in environment")
        self.api = HfApi(token=self.token)
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger("HuggingFaceManager")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)

    def list_models(self):
        """Return URLs for each strategy’s repo."""
        models = []
        for variant in ["unbalanced", "undersampled", "oversampled"]:
            repo_id = f"{os.getenv('HF_USERNAME')}/amazon-sentiment-{variant}"
            models.append({
                "strategy": variant,
                "url": f"https://huggingface.co/{repo_id}"
            })
        return models

    def verify_model_upload(self, variant):
        """Check if the given variant exists on the Hub."""
        repo_id = f"{os.getenv('HF_USERNAME')}/amazon-sentiment-{variant}"
        repos = self.api.list_repos(search=repo_id)
        exists = any(r.modelId == repo_id for r in repos)
        if exists:
            self.logger.info(f"✓ Found {repo_id} on Hugging Face")
        else:
            self.logger.error(f"✗ {repo_id} not found on Hugging Face")
        return exists

    def upload_folder(self, folder_path, repo_id, path_in_repo=".", commit_message="Upload"):
        """
        Upload a local folder (models, tokenizer) to a Hub repo.
        """
        self.api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            token=self.token,
            commit_message=commit_message
        )
        self.logger.info(f"Uploaded {folder_path} → {repo_id}")

    def load_model(self, variant="unbalanced"):
        """
        Download & return (model, tokenizer, device) for inference.
        """
        repo_id = f"{os.getenv('HF_USERNAME')}/amazon-sentiment-{variant}"
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=self.token)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id, use_auth_token=self.token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, tokenizer, device
