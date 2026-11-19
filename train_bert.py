"""Model 3 (Transformer, Area C): fine-tune DistilBERT (WordPiece tokenizer,
self-attention encoder) with a classification head.
Uses Apple-Silicon GPU (mps) or CUDA if available, otherwise CPU."""
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data import load_splits
from src.preprocess import normalize
from src.models import BERT_NAME, MAX_LEN, path

SEED, EPOCHS, BATCH, LR = 42, 1, 32, 3e-5
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

d = load_splits()
Xtr, ytr = d["train"]
Xva, yva = d["validation"]
tok = AutoTokenizer.from_pretrained(BERT_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    BERT_NAME, num_labels=2, id2label={0: "clean", 1: "offensive"},
    label2id={"clean": 0, "offensive": 1}).to(device)


def batches(X, y, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for k in range(0, len(X), BATCH):
        b = idx[k:k + BATCH]
        enc = tok([normalize(X[i]) for i in b], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        yield {k2: v.to(device) for k2, v in enc.items()}, torch.tensor([y[i] for i in b]).to(device)


@torch.no_grad()
def evaluate(X, y):
    model.eval()
    preds = []
    for enc, _ in batches(X, y, False):
        preds += model(**enc).logits.argmax(-1).cpu().tolist()
    return f1_score(y, preds, average="macro")


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(Xtr) + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
best = -1
for epoch in range(EPOCHS):
    model.train()
    for step, (enc, yb) in enumerate(batches(Xtr, ytr, True)):
        loss = model(**enc, labels=yb).loss
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0:
            print("epoch %d step %4d loss %.4f" % (epoch + 1, step, loss.item()))
    f1 = evaluate(Xva, yva)
    print("epoch %d  validation macro-F1 = %.4f" % (epoch + 1, f1))
    if f1 > best:
        best = f1
        model.save_pretrained(path("bert_fp32"))
        tok.save_pretrained(path("bert_fp32"))
        print("  -> saved models/bert_fp32")
print("best validation macro-F1 = %.4f" % best)
