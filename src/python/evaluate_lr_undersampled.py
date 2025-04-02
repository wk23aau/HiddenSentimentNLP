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

def load_and_prepare_data(file_path, text_col="reviewText_clean", sentiment_col="sentiment"):
    """
    Loads the CSV evaluation file and ensures that the specified text and sentiment columns exist.
    If the text column is missing, it creates one from 'reviewText' or 'body'.
    If the sentiment column is missing, it creates one using 'reviewRating' or 'rating' with a threshold of 4.
    """
    df = pd.read_csv(file_path)
    
    # Ensure the review text column exists
    if text_col not in df.columns:
        if "reviewText" in df.columns:
            df[text_col] = df["reviewText"].apply(clean_text)
            print("Created 'reviewText_clean' from 'reviewText'.")
        elif "body" in df.columns:
            df[text_col] = df["body"].apply(clean_text)
            print("Created 'reviewText_clean' from 'body'.")
        else:
            print("Error: No review text column found ('reviewText' or 'body').")
            exit(1)
    else:
        df[text_col] = df[text_col].fillna("")
    
    # Ensure the sentiment column exists
    if sentiment_col not in df.columns:
        if "reviewRating" in df.columns:
            df[sentiment_col] = df["reviewRating"].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'sentiment' column from 'reviewRating'.")
        elif "rating" in df.columns:
            df[sentiment_col] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'sentiment' column from 'rating'.")
        else:
            print(f"Error: No '{sentiment_col}' column found and no valid rating column to create it.")
            exit(1)
    return df

def evaluate_model(eval_file, model_path, vectorizer_path, text_col="reviewText_clean", sentiment_col="sentiment"):
    # Load and prepare the evaluation data
    df = load_and_prepare_data(eval_file, text_col, sentiment_col)
    
    # Load the trained logistic regression model and TF-IDF vectorizer
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Vectorize the evaluation text and predict sentiment
    texts = df[text_col].tolist()
    X_eval = vectorizer.transform(texts)
    y_true = df[sentiment_col].values
    y_pred = clf.predict(X_eval)
    
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
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
    
    # Set the paths to your undersampled logistic regression model and vectorizer.
    MODEL_PATH = "models/logistic-regression/logistic_regression_model_undersampled.joblib"
    VECTORIZER_PATH = "models/logistic-regression/lr_tfidf_vectorizer_undersampled.joblib"
    
    evaluate_model(eval_file, MODEL_PATH, VECTORIZER_PATH)
