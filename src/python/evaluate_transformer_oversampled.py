import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

def clean_text(text):
    """
    Cleans input text by removing HTML tags and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    return ' '.join(text.split())

def load_and_prepare_eval_data(file_path, text_col="reviewText_clean", sentiment_col="sentiment"):
    """
    Loads the CSV evaluation file and ensures that text and sentiment columns are present.
    If sentiment column is missing, attempts to create it using 'reviewRating' or 'rating'.
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
            print(f"Error: Neither '{sentiment_col}' nor a valid rating column found.")
            exit(1)
    return df

def evaluate_transformer(model_dir, eval_file, text_col="reviewText_clean", sentiment_col="sentiment"):
    # Load evaluation data
    df = load_and_prepare_eval_data(eval_file, text_col, sentiment_col)
    
    # Load the tokenizer and model from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    
    # Tokenize evaluation texts
    texts = df[text_col].tolist()
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).numpy()
    
    # True labels
    y_true = df[sentiment_col].values
    
    # Print classification report
    print("Evaluation Metrics:")
    print(classification_report(y_true, preds, zero_division=0))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Transformer Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    # Initialize Tkinter and hide the root window.
    root = tk.Tk()
    root.withdraw()
    
    # Prompt the user to select the evaluation CSV file.
    eval_file = filedialog.askopenfilename(
        title="Select Evaluation CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not eval_file:
        print("No evaluation file selected. Exiting.")
        exit(1)
    
    # Set the directory where your trained transformer model and tokenizer are saved.
    MODEL_DIR = "models/bert-sentiment"
    
    # Run evaluation
    evaluate_transformer(MODEL_DIR, eval_file)
