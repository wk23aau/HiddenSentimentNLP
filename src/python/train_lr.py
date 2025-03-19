# train_lr.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib  # For saving models

def train_logistic_regression(data_file, output_model_path='models/logistic_regression_model.joblib', output_vectorizer_path='models/lr_tfidf_vectorizer.joblib'):
    """
    Trains a Logistic Regression model on review text data and saves the model and TF-IDF vectorizer.

    Args:
        data_file (str): Path to the cleaned CSV data file.
        output_model_path (str): Path to save the trained Logistic Regression model.
        output_vectorizer_path (str): Path to save the fitted TF-IDF vectorizer.
    """
    data = pd.read_csv(data_file)

    # Ensure label exists (create if not)
    if 'label' not in data.columns:
        data['label'] = (data['ratingValue'] >= 4.0).astype(int)

    # Fill missing text
    text_col = 'reviewText_clean' # Or your text column name
    if text_col not in data.columns:
        print(f"Error: Text column '{text_col}' not found in data.")
        return
    data[text_col] = data[text_col].fillna('')

    train_df, _ = train_test_split(data, test_size=0.2, random_state=42) # Using same split as BERT for consistency

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df[text_col])
    y_train = train_df['label']

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Save the trained model and vectorizer
    joblib.dump(clf, output_model_path)
    joblib.dump(vectorizer, output_vectorizer_path)
    print(f"Logistic Regression model saved to: {output_model_path}")
    print(f"TF-IDF vectorizer saved to: {output_vectorizer_path}")


if __name__ == '__main__':
    # --- Hardcoded Paths as requested ---
    DATA_FILE = r'data\processed\combined_processed_data.csv'  # <--- Hardcoded data path
    MODEL_OUTPUT_PATH = r'models\logistic-regression\logistic_regression_model.joblib' # <--- Hardcoded model output path
    VECTORIZER_OUTPUT_PATH = r'models\logistic-regression\lr_tfidf_vectorizer.joblib' # <--- Hardcoded vectorizer output path

    train_logistic_regression(DATA_FILE, MODEL_OUTPUT_PATH, VECTORIZER_OUTPUT_PATH)