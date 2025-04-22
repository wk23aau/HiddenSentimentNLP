#model_training.py

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ReviewsDataset(Dataset):
    """
    Custom Dataset class to handle tokenized reviews and labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score for model evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def load_and_prepare_data(file_path, text_col='reviewText_clean', rating_col='ratingValue'):
    """
    Loads a cleaned CSV, creates a binary label from ratingValue, and returns a DataFrame.

    - text_col: column containing cleaned review text
    - rating_col: column containing numeric rating
    """
    df = pd.read_csv(file_path)
    
    # Ensure ratingValue is valid
    df = df[df[rating_col].notna()].copy()
    
    # Create a binary label: rating >= 4 => positive (1), else negative (0)
    df['label'] = (df[rating_col] >= 4.0).astype(int)
    
    # If text is missing, fill with empty strings
    df[text_col] = df[text_col].fillna('')
    
    # Keep only necessary columns
    df = df[[text_col, 'label']]
    df.columns = ['text', 'label']  # rename for consistency
    return df

def train_model(train_file, model_name='bert-base-uncased', output_dir='models/bert-sentiment', epochs=2, batch_size=8):
    """
    Trains a Hugging Face transformer model for binary sentiment classification.

    - train_file: Path to the cleaned CSV file
    - model_name: Pretrained model from Hugging Face (e.g., 'bert-base-uncased')
    - output_dir: Directory to save the fine-tuned model
    - epochs: Number of training epochs
    - batch_size: Training batch size
    """
    # 1. Load data
    df = load_and_prepare_data(train_file)

    # 2. Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Tokenize using a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True)

    # 4. Create PyTorch datasets
    train_dataset = ReviewsDataset(train_encodings, list(train_df['label']))
    test_dataset = ReviewsDataset(test_encodings, list(test_df['label']))

    # 5. Load a pretrained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 6. Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )

    # 7. Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 8. Train the model
    trainer.train()

    # 9. Evaluate the model
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    # 10. Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model for sentiment classification.")
    parser.add_argument('--train_file', type=str, required=True, help="Path to the cleaned CSV file")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="Hugging Face model to use")
    parser.add_argument('--output_dir', type=str, default='models/bert-sentiment', help="Where to save the model")
    parser.add_argument('--epochs', type=int, default=2, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    args = parser.parse_args()

    train_model(
        train_file=args.train_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
