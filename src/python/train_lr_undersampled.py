import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import tkinter as tk
from tkinter import filedialog

def clean_text(text):
    """
    Cleans input text by removing HTML tags and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    return ' '.join(text.split())

def train_logistic_regression(data_file, 
                              output_model_path='models/logistic-regression/logistic_regression_model_undersampled.joblib', 
                              output_vectorizer_path='models/logistic-regression/lr_tfidf_vectorizer_undersampled.joblib'):
    # Load the undersampled preprocessed CSV file.
    data = pd.read_csv(data_file)
    
    # Create 'reviewText_clean' if it doesn't exist.
    if 'reviewText_clean' not in data.columns:
        if 'reviewText' in data.columns:
            data['reviewText_clean'] = data['reviewText'].apply(clean_text)
            print("Created 'reviewText_clean' from 'reviewText'.")
        elif 'body' in data.columns:
            data['reviewText_clean'] = data['body'].apply(clean_text)
            print("Created 'reviewText_clean' from 'body'.")
        else:
            print("Error: No review text column found ('reviewText' or 'body').")
            return

    # Ensure the 'sentiment' column exists; if not, create it from 'reviewRating' or 'rating'
    if 'sentiment' not in data.columns:
        if 'reviewRating' in data.columns:
            data['sentiment'] = data['reviewRating'].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'sentiment' column from 'reviewRating'.")
        elif 'rating' in data.columns:
            data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'sentiment' column from 'rating'.")
        else:
            print("Error: No valid rating column found to create 'sentiment'.")
            return

    # Fill any missing review text.
    data['reviewText_clean'] = data['reviewText_clean'].fillna("")

    # Split the data into a training set (80%); we use only training for model fitting.
    train_df, _ = train_test_split(data, test_size=0.2, random_state=42)
    
    # Check if training set contains at least two classes.
    unique_classes = train_df['sentiment'].unique()
    if len(unique_classes) < 2:
        print(f"Error: Training data contains only one class: {unique_classes[0]}.")
        return

    # Vectorize the clean review text using TF-IDF.
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['reviewText_clean'])
    y_train = train_df['sentiment']

    # Train a logistic regression model.
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Save the trained model and the TF-IDF vectorizer.
    joblib.dump(clf, output_model_path)
    joblib.dump(vectorizer, output_vectorizer_path)
    print(f"Logistic Regression model saved to: {output_model_path}")
    print(f"TF-IDF vectorizer saved to: {output_vectorizer_path}")

if __name__ == '__main__':
    # Initialize Tkinter and hide the main window.
    root = tk.Tk()
    root.withdraw()
    
    # Prompt the user to select the undersampled, preprocessed CSV file.
    data_file = filedialog.askopenfilename(
        title="Select Undersampled Preprocessed CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    
    if not data_file:
        print("No file selected. Exiting.")
    else:
        # Set output paths for the model and vectorizer.
        MODEL_OUTPUT_PATH = 'models/logistic-regression/logistic_regression_model_undersampled.joblib'
        VECTORIZER_OUTPUT_PATH = 'models/logistic-regression/lr_tfidf_vectorizer_undersampled.joblib'
        train_logistic_regression(data_file, MODEL_OUTPUT_PATH, VECTORIZER_OUTPUT_PATH)
