```markdown
# HiddenSentimentNLP

## Overview
HiddenSentimentNLP is an MSc project that detects hidden positive sentiment in customer reviews using transformer-based NLP models such as BERT and RoBERTa. The project focuses on uncovering subtle, implicit positivity that traditional sentiment analysis methods often overlook.

## Features
- **Data Scraping:** A custom Chrome extension for collecting review data.
- **Data Processing & Modeling:** Python scripts for data preprocessing, exploratory data analysis, and model training.
- **Testing:** Unit tests to ensure reproducibility and code quality.
- **Documentation:** Detailed project documentation, including project plans, progress reports, and evaluation metrics.

## Project Structure
```
project-root/
├── README.md                # Project overview and setup instructions
├── LICENSE                  # License file (MIT)
├── .gitignore               # Files and folders to ignore in Git
├── requirements.txt         # Python dependencies
├── docs/                    # Project documentation (project plan, reports, etc.)
│   ├── project_plan.md
│   └── literature_review.md
├── src/                     # Source code for the project
│   ├── chrome_extension/    # Chrome extension files (manifest.json, background.js, content.js, detail.js)
│   ├── python/              # Python scripts and notebooks for data processing & modeling
│   │   ├── data_preprocessing.py
│   │   ├── model_training.py
│   │   ├── utils.py
│   │   └── experiment_notebook.ipynb
│   └── tests/               # Unit tests for Python code
│       ├── test_data_preprocessing.py
│       └── test_model_training.py
├── data/                    # Raw and processed data files
│   ├── raw/
│   └── processed/
├── models/                  # Saved machine learning model checkpoints and final models
└── figures/                 # Visualizations, charts, and graphs
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/HiddenSentimentNLP.git
   cd HiddenSentimentNLP
   ```

2. **Set Up a Virtual Environment and Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

- **Chrome Extension:**
  - Navigate to the `src/chrome_extension` folder.
  - Load the extension in Chrome via `chrome://extensions` (enable Developer Mode and load unpacked).
  
- **Python Scripts:**
  - Run `data_preprocessing.py` and `model_training.py` from the `src/python` directory to preprocess data and train the model.
  - Use the Jupyter notebook (`experiment_notebook.ipynb`) for exploratory data analysis and experiments.

- **Tests:**
  - Run tests using a test runner (e.g., `pytest`) from the `src/tests` directory:
    ```bash
    pytest src/tests/
    ```

## License
This project is licensed under the MIT License. See the (LICENSE) file for details.

## Contribution
Contributions are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request. For larger changes, please discuss them first via an issue.

## Contact
For any questions or suggestions, please contact Waseem Khan at wk23aau@herts.ac.uk.
```