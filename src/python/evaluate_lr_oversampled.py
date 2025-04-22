# evaluate_lr_oversampled.py

import pandas as pd
import joblib
import tkinter as tk
from tkinter import filedialog
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

def evaluate_model(evaluation_file, model_path='models/logistic-regression/logistic_regression_model.joblib', vectorizer_path='models/logistic-regression/lr_tfidf_vectorizer.joblib'):
    # Load evaluation data
    df = pd.read_csv(evaluation_file)
    
    # Ensure the reviewText_clean column exists; if not, create it using reviewText or body
    if 'reviewText_clean' not in df.columns:
        if 'reviewText' in df.columns:
            df['reviewText_clean'] = df['reviewText'].apply(clean_text)
            print("Created 'reviewText_clean' from 'reviewText' column.")
        elif 'body' in df.columns:
            df['reviewText_clean'] = df['body'].apply(clean_text)
            print("Created 'reviewText_clean' from 'body' column.")
        else:
            print("Error: No review text column found for evaluation.")
            return
    
    # Ensure that a label column exists; if not, try to create one using reviewRating or rating
    if 'label' not in df.columns:
        if 'reviewRating' in df.columns:
            df['label'] = df['reviewRating'].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'label' column from 'reviewRating'.")
        elif 'rating' in df.columns:
            df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'label' column from 'rating'.")
        else:
            print("Error: No label or rating column available for evaluation.")
            return

    # Load the saved logistic regression model and TF-IDF vectorizer
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Vectorize the evaluation text
    texts = df['reviewText_clean'].fillna("")
    X_eval = vectorizer.transform(texts)
    y_true = df['label']
    
    # Predict sentiment labels
    y_pred = clf.predict(X_eval)
    
    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()

if __name__ == '__main__':
    # Initialize Tkinter and hide the root window.
    root = tk.Tk()
    root.withdraw()
    
    # Prompt the user to select an evaluation CSV file.
    evaluation_file = filedialog.askopenfilename(
        title="Select Evaluation CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not evaluation_file:
        print("No evaluation file selected. Exiting.")
    else:
        evaluate_model(evaluation_file)
