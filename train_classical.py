"""Model 1 (classical baseline, Area B): TF-IDF (word 1-2 grams) + logistic regression.
The rule baseline (lexicon + edit distance) needs no training."""
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from src.data import load_splits
from src.preprocess import normalize, tokenize
from src.models import path

os.makedirs("models", exist_ok=True)
d = load_splits()
Xtr, ytr = d["train"]
Xva, yva = d["validation"]

pipe = make_pipeline(
    TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
                    min_df=2, sublinear_tf=True),
    LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000),
)
pipe.fit([normalize(t) for t in Xtr], ytr)
pred = pipe.predict([normalize(t) for t in Xva])
print("TF-IDF + LR  validation macro-F1 = %.4f" % f1_score(yva, pred, average="macro"))
joblib.dump(pipe, path("tfidf_lr.joblib"))
print("saved models/tfidf_lr.joblib")
