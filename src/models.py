"""Model definitions + a common predictor interface used by evaluate.py and app.py.

Every predictor has   .name   and   .predict(text) -> probability that text is offensive.
"""
import os
import joblib
import numpy as np
import torch
import torch.nn as nn

from .preprocess import normalize, tokenize
from .lexicon import rule_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MAX_LEN = 64
BERT_NAME = "distilbert-base-uncased"

# Dynamic INT8 kernels on Apple Silicon / ARM use the qnnpack engine
if "qnnpack" in torch.backends.quantized.supported_engines and \
        "fbgemm" not in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"


def path(name):
    return os.path.join(MODEL_DIR, name)


# ---------------------------------------------------------------- Area A ---
class SkipGram(nn.Module):
    """word2vec skip-gram with negative sampling."""

    def __init__(self, vocab_size, dim=100):
        super().__init__()
        self.inp = nn.Embedding(vocab_size, dim)   # the word vectors we keep
        self.out = nn.Embedding(vocab_size, dim)   # context vectors
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):
        v = self.inp(center)                                   # (B, d)
        u_pos = self.out(context)                              # (B, d)
        u_neg = self.out(negatives)                            # (B, k, d)
        pos = torch.log(torch.sigmoid((v * u_pos).sum(-1)) + 1e-9)
        neg = torch.log(torch.sigmoid(-(u_neg @ v.unsqueeze(-1)).squeeze(-1)) + 1e-9).sum(-1)
        return -(pos + neg).mean()


class FFN(nn.Module):
    """Feedforward classifier on the averaged word2vec vector of a message."""

    def __init__(self, dim=100, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def average_vector(text, vocab, emb):
    ids = [vocab[t] for t in tokenize(normalize(text)) if t in vocab]
    if not ids:
        return np.zeros(emb.shape[1], dtype=np.float32)
    return emb[ids].mean(0)


# ------------------------------------------------------------ predictors ---
class RulePredictor:
    name = "Rule (lexicon + edit distance)"
    files = []

    def predict(self, text):
        return rule_score(text)


class TfidfPredictor:
    name = "TF-IDF + LogReg"
    files = ["tfidf_lr.joblib"]

    def __init__(self):
        self.pipe = joblib.load(path("tfidf_lr.joblib"))

    def predict(self, text):
        return float(self.pipe.predict_proba([normalize(text)])[0, 1])


class W2VFFNPredictor:
    name = "word2vec + FFN"
    files = ["w2v_ffn.pt"]

    def __init__(self):
        ck = torch.load(path("w2v_ffn.pt"), weights_only=False)
        self.vocab, self.emb = ck["vocab"], ck["emb"]
        self.model = FFN(self.emb.shape[1])
        self.model.load_state_dict(ck["ffn"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        x = torch.from_numpy(average_vector(text, self.vocab, self.emb)).unsqueeze(0)
        return float(torch.softmax(self.model(x), -1)[0, 1])


class BertPredictor:
    """DistilBERT classifier. int8=True loads the dynamically quantized copy."""

    def __init__(self, int8=False):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.int8 = int8
        self.name = "DistilBERT INT8 (quantized)" if int8 else "DistilBERT FP32"
        self.tok = AutoTokenizer.from_pretrained(path("bert_fp32"))
        if int8:
            self.files = ["bert_int8.pt"]
            self.model = torch.load(path("bert_int8.pt"), weights_only=False)
        else:
            self.files = ["bert_fp32/model.safetensors"]
            self.model = AutoModelForSequenceClassification.from_pretrained(path("bert_fp32"))
        self.model.to("cpu").eval()

    @torch.no_grad()
    def predict(self, text):
        enc = self.tok(normalize(text), truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return float(torch.softmax(self.model(**enc).logits, -1)[0, 1])


def load_all():
    return [RulePredictor(), TfidfPredictor(), W2VFFNPredictor(),
            BertPredictor(int8=False), BertPredictor(int8=True)]


def size_mb(pred):
    if not pred.files:   # rule baseline: the lexicon lives in the code
        import sys
        from .lexicon import OFFENSIVE_WORDS
        return sum(sys.getsizeof(w) for w in OFFENSIVE_WORDS) / 1e6
    return sum(os.path.getsize(path(f)) for f in pred.files) / 1e6
