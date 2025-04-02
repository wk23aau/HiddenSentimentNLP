import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ReviewsDataset(Dataset):
    """
    Custom Dataset for loading tokenized reviews and sentiment labels.
    """
    def __init__(self, encodings, sentiments):
        self.encodings = encodings
        self.sentiments = sentiments
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.sentiments[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.sentiments)

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def load_and_prepare_data(file_path, text_col="reviewText_clean", sentiment_col="sentiment"):
    """
    Loads the CSV file, fills missing text values, and ensures that a sentiment column exists.
    If the sentiment column is missing, it is created from 'reviewRating' or 'rating' using a threshold (>= 4 -> 1, else 0).
    """
    df = pd.read_csv(file_path)
    df[text_col] = df[text_col].fillna("")
    
    if sentiment_col not in df.columns:
        if "reviewRating" in df.columns:
            df[sentiment_col] = df["reviewRating"].apply(lambda x: 1 if x >= 4 else 0)
            print("Created sentiment column from 'reviewRating'.")
        elif "rating" in df.columns:
            df[sentiment_col] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
            print("Created sentiment column from 'rating'.")
        else:
            print(f"Error: Sentiment column '{sentiment_col}' not found and no valid rating column to create it.")
            exit(1)
    return df

def train_transformer(train_file, model_name="bert-base-uncased", output_dir="models/bert-sentiment-undersampled", epochs=2, batch_size=8):
    """
    Fine-tunes a transformer model on the provided training data from an undersampled dataset.
    """
    # Load and prepare data
    df = load_and_prepare_data(train_file, text_col="reviewText_clean", sentiment_col="sentiment")
    
    # Split into training and evaluation sets (80/20 split)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Load pretrained tokenizer and tokenize texts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(list(train_df["reviewText_clean"]), truncation=True, padding=True)
    eval_encodings = tokenizer(list(eval_df["reviewText_clean"]), truncation=True, padding=True)
    
    # Create PyTorch datasets using the 'sentiment' column
    train_dataset = ReviewsDataset(train_encodings, list(train_df["sentiment"]))
    eval_dataset = ReviewsDataset(eval_encodings, list(eval_df["sentiment"]))
    
    # Load pretrained model for sequence classification (binary classification)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Save the trained model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    # Initialize Tkinter and hide the root window.
    root = tk.Tk()
    root.withdraw()

    # Prompt the user to select the undersampled, preprocessed CSV file.
    file_path = filedialog.askopenfilename(
        title="Select Undersampled Preprocessed CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        print("No file selected, exiting.")
        exit(1)
    
    # Set the output directory for the trained transformer model.
    MODEL_OUTPUT_DIR = "models/bert-sentiment-undersampled"
    
    # Train the transformer model using the selected CSV file.
    train_transformer(file_path, model_name="bert-base-uncased", output_dir=MODEL_OUTPUT_DIR, epochs=2, batch_size=8)
