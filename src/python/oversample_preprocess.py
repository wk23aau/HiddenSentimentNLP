import json
import pandas as pd
import re
import tkinter as tk
from tkinter import filedialog
from imblearn.over_sampling import RandomOverSampler

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

def load_reviews_from_json(file_path):
    """
    Loads reviews from a JSON file where each record has a "reviews" field.
    The "reviews" field is a JSON string containing "all" and "critical" arrays.
    Adds a 'review_type' column to distinguish between them and creates a clean review text.
    Also assigns a sentiment label using the numeric reviewRating (or rating) field.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    reviews_list = []
    for record in data:
        if 'reviews' in record:
            reviews_json_str = record['reviews']
            try:
                reviews_data = json.loads(reviews_json_str)
            except Exception as e:
                print("Error parsing reviews field:", e)
                continue
            
            # Process "all" reviews: add review_type and clean text (from body or title if needed)
            if 'all' in reviews_data:
                for review in reviews_data['all']:
                    review['review_type'] = 'all'
                    # Create a clean review text if not already present
                    if 'reviewText_clean' not in review:
                        if 'reviewText' in review:
                            review['reviewText_clean'] = clean_text(review['reviewText'])
                        elif 'body' in review:
                            review['reviewText_clean'] = clean_text(review['body'])
                        else:
                            review['reviewText_clean'] = ""
                    reviews_list.append(review)
            
            # Process "critical" reviews: similar processing
            if 'critical' in reviews_data:
                for review in reviews_data['critical']:
                    review['review_type'] = 'critical'
                    if 'reviewText_clean' not in review:
                        if 'reviewText' in review:
                            review['reviewText_clean'] = clean_text(review['reviewText'])
                        elif 'body' in review:
                            review['reviewText_clean'] = clean_text(review['body'])
                        else:
                            review['reviewText_clean'] = ""
                    reviews_list.append(review)
    
    df = pd.DataFrame(reviews_list)
    
    # Assign sentiment label based on numeric rating threshold (>= 4 -> positive, else negative)
    if 'reviewRating' in df.columns:
        df['sentiment'] = df['reviewRating'].apply(lambda x: 1 if x >= 4 else 0)
    elif 'rating' in df.columns:
        df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
    else:
        print("Warning: No valid rating column found. Cannot assign sentiment labels.")
    
    return df

def oversample_reviews(df, label_column='sentiment'):
    """
    Balances the DataFrame so that the count of each sentiment label is equal.
    Uses RandomOverSampler (applied on the DataFrame index as a proxy) to duplicate the minority class.
    """
    X = df.index.values.reshape(-1, 1)
    y = df[label_column]
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    df_resampled = df.iloc[X_res.flatten()].reset_index(drop=True)
    return df_resampled

def main():
    # Initialize Tkinter and hide the root window.
    root = tk.Tk()
    root.withdraw()
    
    # Prompt the user to select the input JSON file.
    input_file = filedialog.askopenfilename(
        title="Select Input JSON File",
        filetypes=[("JSON files", "*.json")]
    )
    if not input_file:
        print("No file selected. Exiting.")
        return
    print("Selected file:", input_file)
    
    # Load, parse, and process reviews from the selected file.
    df_reviews = load_reviews_from_json(input_file)
    print(f"Loaded {len(df_reviews)} reviews from the file.")
    
    # Verify sentiment distribution before oversampling.
    if 'sentiment' in df_reviews.columns:
        print("Sentiment counts before oversampling:")
        print(df_reviews['sentiment'].value_counts())
    else:
        print("No sentiment labels assigned.")
    
    # Oversample the data to balance sentiment labels.
    df_oversampled = oversample_reviews(df_reviews, label_column='sentiment')
    print("Sentiment counts after oversampling:")
    print(df_oversampled['sentiment'].value_counts())
    
    # Prompt the user for an output file to save the oversampled CSV.
    output_file = filedialog.asksaveasfilename(
        title="Save Oversampled Data as CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not output_file:
        print("No output file selected. Exiting.")
        return
    
    df_oversampled.to_csv(output_file, index=False)
    print(f"Oversampled data saved to {output_file}")

if __name__ == '__main__':
    main()
