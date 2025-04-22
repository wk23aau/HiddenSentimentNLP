# train_logistic_regression_pkl.py

# In some training script (could be part of compare_models.py or a new script)
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Suppose you have a pandas DataFrame df with columns: "reviewText_clean" and "label" (0 or 1)
X = df["reviewText_clean"].fillna("")
y = df["label"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=200))
])

pipeline.fit(X, y)

# Then save
pickle.dump(pipeline, open("models/logistic-regression-pkl/lr_pipeline.pkl", "wb"))
