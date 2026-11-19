"""Dataset: OLID / OffensEval-2019 (English tweets, offensive vs. not offensive),
loaded from the Hugging Face Hub as `cardiffnlp/tweet_eval`, config `offensive`.
Official splits: train 11,916 / validation 1,324 / test 860.
Label 0 = clean (NOT), 1 = offensive (OFF).
All four models are evaluated on the SAME held-out test split.
"""
from datasets import load_dataset

LABELS = ["clean", "offensive"]


def load_splits():
    ds = load_dataset("cardiffnlp/tweet_eval", "offensive")
    out = {}
    for split in ["train", "validation", "test"]:
        out[split] = (list(ds[split]["text"]), list(ds[split]["label"]))
    return out


if __name__ == "__main__":
    d = load_splits()
    for k, (x, y) in d.items():
        print(k, len(x), "offensive ratio = %.3f" % (sum(y) / len(y)))
