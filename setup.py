from setuptools import setup, find_packages

setup(
    name="amazon_nlp",
    version="0.1.0",
    description="End-to-end Amazon sentiment NLP pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "scikit-learn",
        "tqdm",
        "google-generativeai",
        "python-dotenv",
        "transformers",
        "huggingface-hub",
        "torch",
        "psutil",
        "imblearn",
        "tenacity",
        "fastapi",
        "uvicorn",
        "kaggle",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "amazon-nlp=amazon_nlp.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
