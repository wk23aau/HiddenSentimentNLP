import pandas as pd
import numpy as np
import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt

def traditional_model(data, text_col='reviewText_clean', label_col='label'):
    # Split data (using a fixed seed for reproducibility)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    
    # Ensure no missing text values
    train_df[text_col] = train_df[text_col].fillna('')
    test_df[text_col] = test_df[text_col].fillna('')
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    y_train = train_df[label_col]
    y_test = test_df[label_col]
    
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return test_df, y_test, y_pred

def transformer_model(data, model_dir, text_col='reviewText_clean', label_col='label'):
    # Split data (same seed for consistency)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    y_test = test_df[label_col].values
    
    # Load the tokenizer and model from local files
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    
    # Ensure no missing text values
    test_df[text_col] = test_df[text_col].fillna('')
    inputs = tokenizer(list(test_df[text_col]), truncation=True, padding=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).numpy()
    
    return test_df, y_test, preds

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def main(data_file, model_dir):
    # Load the cleaned CSV
    data = pd.read_csv(data_file)
    
    # If label doesn't exist, create one based on ratingValue threshold (>= 4.0 is positive)
    if 'label' not in data.columns:
        data['label'] = (data['ratingValue'] >= 4.0).astype(int)
    
    # Ensure the review text is not missing
    if 'reviewText_clean' in data.columns:
        data['reviewText_clean'] = data['reviewText_clean'].fillna('')
    
    print("=== Traditional Model Evaluation ===")
    _, y_true_trad, y_pred_trad = traditional_model(data)
    print(classification_report(y_true_trad, y_pred_trad))
    plot_confusion_matrix(y_true_trad, y_pred_trad, "Traditional Model Confusion Matrix")
    
    print("=== Transformer Model Evaluation ===")
    _, y_true_trans, y_pred_trans = transformer_model(data, model_dir)
    print(classification_report(y_true_trans, y_pred_trans))
    plot_confusion_matrix(y_true_trans, y_pred_trans, "Transformer Model Confusion Matrix")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize and Compare Traditional and Transformer Models for Sentiment Classification"
    )
    parser.add_argument('--data_file', type=str, required=True, help="Path to the cleaned CSV file")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of the fine-tuned transformer model")
    args = parser.parse_args()
    
    main(args.data_file, args.model_dir)
