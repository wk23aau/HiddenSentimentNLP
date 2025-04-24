# File: src/amazon_nlp/train/base_transformer_trainer.py

import os
# Disable Weights & Biases in non‐interactive environments (e.g. Kaggle)
os.environ["WANDB_DISABLED"] = "true"

import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from amazon_nlp.config.model_config import ModelConfig

class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

class BaseTransformerTrainer:
    def __init__(self, model_type: str = "unbalanced"):
        self.config     = ModelConfig
        self.model_name = self.config.MODEL_NAME
        self.model_type = model_type

        # Output directory
        self.output_dir = Path(
            os.getenv("MODEL_OUTPUT_DIR", self.config.OUTPUT_DIR)
        ) / model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def setup_logging(self):
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = logs_dir / f"all_transformers_{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds  = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc    = accuracy_score(labels, preds)
        probas = torch.nn.functional.softmax(torch.tensor(logits), dim=1)
        auc    = roc_auc_score(labels, probas[:, 1].numpy())
        return {
            "accuracy":  acc,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "roc_auc":   auc
        }

    def load_datasets(self):
        """
        Find and load the latest CSVs for this model type.
        Expects files named like:
          train_<model_type>_YYYYMMDD_HHMMSS.csv
          validation_YYYYMMDD_HHMMSS.csv
          test_YYYYMMDD_HHMMSS.csv
        """
        data_dir  = Path("data/datasets")
        all_train = sorted(data_dir.glob(f"train_{self.model_type}_*.csv"))
        if not all_train:
            raise FileNotFoundError(
                f"No train CSVs found for {self.model_type} in {data_dir}"
            )
        latest    = all_train[-1]
        timestamp = latest.stem.split(f"train_{self.model_type}_")[-1]

        logging.info(f"Loading datasets with timestamp {timestamp} from {data_dir}")
        df_train = pd.read_csv(data_dir / f"train_{self.model_type}_{timestamp}.csv")
        df_val   = pd.read_csv(data_dir / f"validation_{timestamp}.csv")
        df_test  = pd.read_csv(data_dir / f"test_{timestamp}.csv")
        logging.info(
            f"Sizes → train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}"
        )
        return df_train, df_val, df_test

    def prepare_data(self, df_train, df_val, df_test):
        """
        Tokenize and wrap into PyTorch Datasets.
        Automatically uses 'text_clean' if present, otherwise 'text'.
        """
        def get_texts(df):
            if "text_clean" in df.columns:
                return df["text_clean"].astype(str).tolist()
            elif "text" in df.columns:
                return df["text"].astype(str).tolist()
            else:
                raise KeyError("No text column found in DataFrame")

        texts_train = get_texts(df_train)
        texts_val   = get_texts(df_val)
        texts_test  = get_texts(df_test)

        enc_train = self.tokenizer(
            texts_train,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors="pt"
        )
        enc_val   = self.tokenizer(
            texts_val,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors="pt"
        )
        enc_test  = self.tokenizer(
            texts_test,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors="pt"
        )

        train_ds = ReviewsDataset(enc_train, df_train["label"].tolist())
        val_ds   = ReviewsDataset(enc_val,   df_val["label"].tolist())
        test_ds  = ReviewsDataset(enc_test,  df_test["label"].tolist())
        return train_ds, val_ds, test_ds

    def train(self, train_ds, val_ds, test_ds):
        """Run the full training and final evaluation."""
        self.setup_logging()
        logging.info(f"Starting training for '{self.model_type}' variant")

        args = TrainingArguments(
            output_dir=                  str(self.output_dir),
            num_train_epochs=            self.config.EPOCHS,
            per_device_train_batch_size= self.config.BATCH_SIZE,
            per_device_eval_batch_size=  self.config.BATCH_SIZE,
            learning_rate=               self.config.LEARNING_RATE,
            weight_decay=                self.config.WEIGHT_DECAY,
            logging_dir="logs",
            logging_steps=100,

            # <— use eval_strategy (not evaluation_strategy) and match save_strategy
            eval_strategy=IntervalStrategy.STEPS,
            eval_steps=    500,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=    500,

            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,

            # Disable built-in loggers (wandb, etc.) when running in notebooks
            report_to=[],
            run_name=None,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE
            )],
        )

        trainer.train()
        metrics = trainer.evaluate(eval_dataset=test_ds)
        logging.info(f"Final test metrics: {metrics}")
        return metrics