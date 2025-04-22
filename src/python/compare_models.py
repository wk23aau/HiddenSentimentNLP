#compare_models.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1 score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f1

def traditional_model(data, text_col='reviewText_clean', label_col='label'):
    """Train and evaluate a traditional model using TF-IDF + Logistic Regression."""
    # Split into train/test using a fixed seed
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Fill missing text values with empty strings
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
    return y_test, y_pred

def transformer_model(data, model_dir, text_col='reviewText_clean', label_col='label'):
    """Evaluate the transformer-based model on the test set."""
    # Split data using the same random seed for consistency
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    y_test = test_df[label_col].values

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    # Ensure missing values are filled
    test_df[text_col] = test_df[text_col].fillna('')

    # Tokenize the test set
    inputs = tokenizer(list(test_df[text_col]), truncation=True, padding=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).numpy()
    return y_test, preds

def main(data_file, model_dir):
    """Main function to compare traditional and transformer models."""
    # Load the cleaned CSV
    data = pd.read_csv(data_file)

    # If a label column doesn't exist, create one from ratingValue
    if 'label' not in data.columns:
        data['label'] = (data['ratingValue'] >= 4.0).astype(int)

    # Fill missing reviewText_clean with empty strings
    if 'reviewText_clean' in data.columns:
        data['reviewText_clean'] = data['reviewText_clean'].fillna('')

    # Traditional Model
    y_true_trad, y_pred_trad = traditional_model(data)
    acc_trad, prec_trad, rec_trad, f1_trad = compute_metrics(y_true_trad, y_pred_trad)
    print("Traditional Model (TF-IDF + Logistic Regression) Metrics:")
    print(f"  Accuracy:  {acc_trad:.4f}")
    print(f"  Precision: {prec_trad:.4f}")
    print(f"  Recall:    {rec_trad:.4f}")
    print(f"  F1 Score:  {f1_trad:.4f}\n")

    # Transformer Model
    y_true_trans, y_pred_trans = transformer_model(data, model_dir)
    acc_trans, prec_trans, rec_trans, f1_trans = compute_metrics(y_true_trans, y_pred_trans)
    print("Transformer Model (BERT-based) Metrics:")
    print(f"  Accuracy:  {acc_trans:.4f}")
    print(f"  Precision: {prec_trans:.4f}")
    print(f"  Recall:    {rec_trans:.4f}")
    print(f"  F1 Score:  {f1_trans:.4f}")

if __name__ == '__main__':
    # Hardcoded file and directory paths
    data_file_path = 'data/processed/airpods.csv' # <--- HARDCODED DATA FILE PATH
    transformer_model_dir = 'models/bert-sentiment' # <--- HARDCODED MODEL DIRECTORY

    main(data_file_path, transformer_model_dir)