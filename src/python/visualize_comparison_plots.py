import pandas as pd
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def traditional_model_normal(data, text_col='reviewText_clean', label_col='label'):
    # Split into train and test sets (fixed random seed for reproducibility)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df[text_col] = train_df[text_col].fillna('')
    test_df[text_col] = test_df[text_col].fillna('')
    
    vectorizer = TfidfVectorizer(max_features=5000)  # using unigrams
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    y_train = train_df[label_col]
    y_test = test_df[label_col]
    
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Get predicted probabilities for the positive class
    y_prob = clf.predict_proba(X_test)[:, 1]
    return y_test, y_pred, y_prob

def traditional_model_fair(data, text_col='reviewText_clean', label_col='label'):
    # Split into train and test sets
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df[text_col] = train_df[text_col].fillna('')
    test_df[text_col] = test_df[text_col].fillna('')
    
    # Use unigrams and bigrams for feature extraction
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    y_train = train_df[label_col]
    y_test = test_df[label_col]
    
    # Apply balanced class weights to handle potential imbalance
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return y_test, y_pred, y_prob

def transformer_model(data, model_dir, text_col='reviewText_clean', label_col='label'):
    # Split into train and test sets
    _, test_df = train_test_split(data, test_size=0.2, random_state=42)
    y_test = test_df[label_col].values
    
    # Load tokenizer and model from local directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    
    test_df[text_col] = test_df[text_col].fillna('')
    inputs = tokenizer(list(test_df[text_col]), truncation=True, padding=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    y_pred = torch.argmax(logits, dim=1).numpy()
    # Compute probabilities using softmax and take the probability for class 1
    y_prob = softmax(logits.numpy(), axis=1)[:, 1]
    return y_test, y_pred, y_prob

def main(data_file, model_dir):
    data = pd.read_csv(data_file)
    
    # Create a binary label if not already present using ratingValue threshold (>= 4.0 is positive)
    if 'label' not in data.columns:
        data['label'] = (data['ratingValue'] >= 4.0).astype(int)
    
    # Ensure no missing text in reviewText_clean
    if 'reviewText_clean' in data.columns:
        data['reviewText_clean'] = data['reviewText_clean'].fillna('')
    
    print("=== Normal Traditional Model Evaluation ===")
    y_true_norm, y_pred_norm, y_prob_norm = traditional_model_normal(data)
    print(classification_report(y_true_norm, y_pred_norm, zero_division=0))
    plot_confusion_matrix(y_true_norm, y_pred_norm, "Normal Traditional Model Confusion Matrix")
    plot_roc_curve(y_true_norm, y_prob_norm, "Normal Traditional Model ROC Curve")
    
    print("=== Fair Traditional Model Evaluation ===")
    y_true_fair, y_pred_fair, y_prob_fair = traditional_model_fair(data)
    print(classification_report(y_true_fair, y_pred_fair, zero_division=0))
    plot_confusion_matrix(y_true_fair, y_pred_fair, "Fair Traditional Model Confusion Matrix")
    plot_roc_curve(y_true_fair, y_prob_fair, "Fair Traditional Model ROC Curve")
    
    print("=== Transformer Model Evaluation ===")
    y_true_trans, y_pred_trans, y_prob_trans = transformer_model(data, model_dir)
    print(classification_report(y_true_trans, y_pred_trans, zero_division=0))
    plot_confusion_matrix(y_true_trans, y_pred_trans, "Transformer Model Confusion Matrix")
    plot_roc_curve(y_true_trans, y_prob_trans, "Transformer Model ROC Curve")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize and Compare Models (Traditional Normal, Fair Traditional, and Transformer) for Sentiment Classification"
    )
    parser.add_argument('--data_file', type=str, required=True, help="Path to the cleaned CSV file")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of the fine-tuned transformer model")
    args = parser.parse_args()
    
    main(args.data_file, args.model_dir)
