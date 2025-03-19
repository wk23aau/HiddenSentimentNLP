import os
import glob
import pandas as pd
import re
import argparse

def extract_rating(rating_str):
    """
    Extract a float rating from a string like '5.0 out of 5 stars'.
    Returns None if no match is found or if input isn't a string.
    """
    if not isinstance(rating_str, str):
        return None
    # Look for patterns like "4.5 out of 5" or "5 out of 5"
    match = re.search(r'(\d+(?:\.\d+)?) out of 5', rating_str)
    if match:
        return float(match.group(1))
    return None

def extract_date(date_str):
    """
    Extract and parse a date from a string like:
    'Reviewed in the United States on February 17, 2025'.
    Returns a Python date object, or None if parsing fails.
    """
    if not isinstance(date_str, str):
        return None

    prefix = "Reviewed in the United States on "
    if date_str.startswith(prefix):
        date_str = date_str[len(prefix):]

    # Attempt to parse the remaining text as a date
    try:
        parsed = pd.to_datetime(date_str, errors='coerce')
        if pd.isnull(parsed):
            return None
        return parsed.date()
    except:
        return None

def clean_text(text):
    """
    Clean text by removing HTML tags and extra whitespace.
    """
    if not isinstance(text, str):
        return text
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace (including newlines)
    text = ' '.join(text.split())
    return text

def load_data(file_path):
    """
    Load CSV data from the given file path using robust parsing.
    Uses the Python engine to handle multiline reviews and
    on_bad_lines='skip' to ignore malformed rows.
    """
    try:
        df = pd.read_csv(
            file_path,
            delimiter=",",
            quotechar='"',
            engine="python",
            on_bad_lines="skip"
        )
        print(f"Loaded {os.path.basename(file_path)} with columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_dataframe(df):
    """
    Preprocess the dataframe by:
      - Extracting numeric rating from 'reviewRating'
      - Cleaning 'reviewTitle' and 'reviewText'
      - Parsing 'reviewDate' into a proper date
      - Preserving original columns
      - Adding new columns: 'ratingValue', 'reviewTitle_clean', 'reviewText_clean', 'reviewDate_parsed'
    """
    if 'reviewRating' in df.columns:
        df['ratingValue'] = df['reviewRating'].apply(extract_rating)
    else:
        print("Warning: 'reviewRating' column not found.")

    if 'reviewTitle' in df.columns:
        df['reviewTitle_clean'] = df['reviewTitle'].apply(clean_text)
    else:
        print("Warning: 'reviewTitle' column not found.")

    if 'reviewText' in df.columns:
        df['reviewText_clean'] = df['reviewText'].apply(clean_text)
    else:
        print("Warning: 'reviewText' column not found.")

    if 'reviewDate' in df.columns:
        df['reviewDate_parsed'] = df['reviewDate'].apply(extract_date)
    else:
        print("Warning: 'reviewDate' column not found.")

    return df

def process_files_and_combine(input_dir, output_dir, output_filename="combined_processed_data.csv"):
    """
    Process all CSV files in the input directory, preprocess them,
    and combine all processed data into a single CSV file in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    if not csv_files:
        print("No CSV files found in the input directory.")
        return None  # Return None to indicate no combined file was created

    processed_dfs = []
    for file_path in csv_files:
        print(f"Processing {file_path}...")
        df = load_data(file_path)
        if df is None or df.empty:
            continue
        df_clean = preprocess_dataframe(df)
        processed_dfs.append(df_clean)

    if not processed_dfs:
        print("No dataframes were successfully processed. No combined file will be created.")
        return None  # Return None if no dataframes were processed

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    output_file = os.path.join(output_dir, output_filename)

    try:
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined cleaned data to {output_file}")
        return combined_df # Return the combined dataframe
    except Exception as e:
        print(f"Error saving combined file {output_file}: {e}")
        return None # Return None in case of error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch Data Preprocessing and Combine for HiddenSentimentNLP Dataset")
    parser.add_argument('--input_dir', type=str, default='data/raw', help="Directory containing raw CSV files")
    parser.add_argument('--output_dir', type=str, default='data/processed', help="Directory to save the combined cleaned CSV file")
    parser.add_argument('--output_filename', type=str, default='combined_processed_data.csv', help="Filename for the combined output CSV file")
    args = parser.parse_args()

    combined_data = process_files_and_combine(args.input_dir, args.output_dir, args.output_filename)
    if combined_data is not None:
        print("All files processed and combined.")
    else:
        print("Processing and combining completed with potential issues. Check logs for details.")