import pandas as pd
import numpy as np
import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1 score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return accuracy, precision, recall, f1

def traditional_model_normal(data, text_col='reviewText_clean', label_col='label'):
    """Train and evaluate a traditional model using default TF-IDF + Logistic Regression."""
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df[text_col] = train_df[text_col].fillna('')
    test_df[text_col] = test_df[text_col].fillna('')
    
    vectorizer = TfidfVectorizer(max_features=5000)  # default unigrams
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    y_train = train_df[label_col]
    y_test = test_df[label_col]
    
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_test, y_pred

def traditional_model_fair(data, text_col='reviewText_clean', label_col='label'):
    """Train and evaluate a fairer traditional model using TF-IDF with bigrams and balanced class weights."""
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    train_df[text_col] = train_df[text_col].fillna('')
    test_df[text_col] = test_df[text_col].fillna('')
    
    # Use both unigrams and bigrams to capture more context
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_df[text_col])
    X_test = vectorizer.transform(test_df[text_col])
    y_train = train_df[label_col]
    y_test = test_df[label_col]
    
    # Use balanced class weights to account for any class imbalance
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_test, y_pred

def transformer_model(data, model_dir, text_col='reviewText_clean', label_col='label'):
    """Evaluate the transformer-based model on the test set."""
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    y_test = test_df[label_col].values
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    
    test_df[text_col] = test_df[text_col].fillna('')
    inputs = tokenizer(list(test_df[text_col]), truncation=True, padding=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).numpy()
    return y_test, preds

def main(data_file, model_dir):
    # Load the cleaned CSV
    data = pd.read_csv(data_file)
    
    # If label doesn't exist, create one using ratingValue (>= 4.0 is positive)
    if 'label' not in data.columns:
        data['label'] = (data['ratingValue'] >= 4.0).astype(int)
    
    if 'reviewText_clean' in data.columns:
        data['reviewText_clean'] = data['reviewText_clean'].fillna('')
    
    print("=== Normal Traditional Model Evaluation ===")
    y_true_norm, y_pred_norm = traditional_model_normal(data)
    acc_norm, prec_norm, rec_norm, f1_norm = compute_metrics(y_true_norm, y_pred_norm)
    print(f"Accuracy:  {acc_norm:.4f}")
    print(f"Precision: {prec_norm:.4f}")
    print(f"Recall:    {rec_norm:.4f}")
    print(f"F1 Score:  {f1_norm:.4f}")
    print(classification_report(y_true_norm, y_pred_norm, zero_division=0))
    print("\n")
    
    print("=== Fair Traditional Model Evaluation ===")
    y_true_fair, y_pred_fair = traditional_model_fair(data)
    acc_fair, prec_fair, rec_fair, f1_fair = compute_metrics(y_true_fair, y_pred_fair)
    print(f"Accuracy:  {acc_fair:.4f}")
    print(f"Precision: {prec_fair:.4f}")
    print(f"Recall:    {rec_fair:.4f}")
    print(f"F1 Score:  {f1_fair:.4f}")
    print(classification_report(y_true_fair, y_pred_fair, zero_division=0))
    print("\n")
    
    print("=== Transformer Model Evaluation ===")
    y_true_trans, y_pred_trans = transformer_model(data, model_dir)
    acc_trans, prec_trans, rec_trans, f1_trans = compute_metrics(y_true_trans, y_pred_trans)
    print(f"Accuracy:  {acc_trans:.4f}")
    print(f"Precision: {prec_trans:.4f}")
    print(f"Recall:    {rec_trans:.4f}")
    print(f"F1 Score:  {f1_trans:.4f}")
    print(classification_report(y_true_trans, y_pred_trans, zero_division=0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compare Normal and Fair Traditional Models with a Transformer Model for Sentiment Classification"
    )
    parser.add_argument('--data_file', type=str, required=True, help="Path to the cleaned CSV file")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of the fine-tuned transformer model")
    args = parser.parse_args()
    
    main(args.data_file, args.model_dir)
