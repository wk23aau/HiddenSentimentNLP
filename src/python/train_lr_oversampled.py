# train_lr_oversampled.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def clean_text(text):
    """
    Cleans input text by removing HTML tags and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    return ' '.join(text.split())

def train_logistic_regression(data_file, 
                              output_model_path='models/logistic-regression/logistic_regression_model.joblib', 
                              output_vectorizer_path='models/logistic-regression/lr_tfidf_vectorizer.joblib'):
    # Load the oversampled, preprocessed data from CSV.
    data = pd.read_csv(data_file)

    # Create the 'reviewText_clean' column if it doesn't exist.
    if 'reviewText_clean' not in data.columns:
        if 'reviewText' in data.columns:
            data['reviewText_clean'] = data['reviewText'].apply(clean_text)
            print("Created 'reviewText_clean' from 'reviewText' column.")
        elif 'body' in data.columns:
            data['reviewText_clean'] = data['body'].apply(clean_text)
            print("Created 'reviewText_clean' from 'body' column.")
        else:
            print("Error: No review text column found ('reviewText' or 'body') to create 'reviewText_clean'.")
            return

    # Create binary sentiment labels if not already present using reviewRating (or rating) column.
    if 'label' not in data.columns:
        if 'reviewRating' in data.columns:
            data['label'] = data['reviewRating'].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'label' column from 'reviewRating'.")
        elif 'rating' in data.columns:
            data['label'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)
            print("Created 'label' column from 'rating'.")
        else:
            print("Error: No valid rating column found to create labels.")
            return

    # Ensure no missing text.
    text_col = 'reviewText_clean'
    data[text_col] = data[text_col].fillna('')

    # Split the data into training (80%) and validation sets.
    train_df, _ = train_test_split(data, test_size=0.2, random_state=42)

    # Verify the training set has at least two classes.
    unique_classes = train_df['label'].unique()
    if len(unique_classes) < 2:
        print(f"Error: Training data contains only one class: {unique_classes[0]}. Please check your oversampled data.")
        return

    # Vectorize the review text using TF-IDF.
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df[text_col])
    y_train = train_df['label']

    # Train a logistic regression model.
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Save the trained model and the TF-IDF vectorizer.
    joblib.dump(clf, output_model_path)
    joblib.dump(vectorizer, output_vectorizer_path)
    print(f"Logistic Regression model saved to: {output_model_path}")
    print(f"TF-IDF vectorizer saved to: {output_vectorizer_path}")

if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog

    # Initialize Tkinter and hide the root window.
    root = tk.Tk()
    root.withdraw()

    # Prompt the user to select the oversampled, preprocessed CSV file.
    data_file = filedialog.askopenfilename(
        title="Select Oversampled Preprocessed CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not data_file:
        print("No file selected. Exiting.")
    else:
        # Set output paths for the model and vectorizer.
        MODEL_OUTPUT_PATH = 'models/logistic-regression/logistic_regression_model.joblib'
        VECTORIZER_OUTPUT_PATH = 'models/logistic-regression/lr_tfidf_vectorizer.joblib'
        train_logistic_regression(data_file, MODEL_OUTPUT_PATH, VECTORIZER_OUTPUT_PATH)
