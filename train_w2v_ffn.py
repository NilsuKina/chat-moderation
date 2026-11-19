"""Model 2 (neural network, Area A):
  step 1 - train our OWN word2vec (skip-gram + negative sampling) on the training messages
  step 2 - average the word vectors of each message
  step 3 - train a feedforward network with SGD + backpropagation
"""
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data import load_splits
from src.preprocess import normalize, tokenize
from src.models import SkipGram, FFN, average_vector, path

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs("models", exist_ok=True)

d = load_splits()
Xtr, ytr = d["train"]
Xva, yva = d["validation"]
sents = [tokenize(normalize(t)) for t in Xtr]

# ---------- vocabulary
counts = Counter(w for s in sents for w in s)
words = [w for w, c in counts.items() if c >= 2]
vocab = {w: i for i, w in enumerate(words)}
print("vocab size:", len(vocab))

# ---------- skip-gram pairs (window = 3)
WINDOW, NEG, DIM = 3, 5, 100
pairs = []
for s in sents:
    ids = [vocab[w] for w in s if w in vocab]
    for i, c in enumerate(ids):
        for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
            if i != j:
                pairs.append((c, ids[j]))
pairs = torch.tensor(pairs)
print("training pairs:", len(pairs))

# negative sampling distribution: unigram^0.75
freq = torch.tensor([counts[w] for w in words], dtype=torch.float) ** 0.75
freq /= freq.sum()

sg = SkipGram(len(vocab), DIM)
opt = torch.optim.Adam(sg.parameters(), lr=0.005)
for epoch in range(5):
    perm = torch.randperm(len(pairs))
    total = 0.0
    for k in range(0, len(pairs), 1024):
        b = pairs[perm[k:k + 1024]]
        negs = torch.multinomial(freq, len(b) * NEG, replacement=True).view(len(b), NEG)
        loss = sg(b[:, 0], b[:, 1], negs)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(b)
    print("word2vec epoch %d  loss %.4f" % (epoch + 1, total / len(pairs)))
emb = sg.inp.weight.detach().numpy().astype(np.float32)

# sanity check: nearest neighbours
def neighbours(w, k=5):
    if w not in vocab:
        return []
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = e @ e[vocab[w]]
    return [words[i] for i in np.argsort(-sims)[1:k + 1]]
for w in ["idiot", "liberals", "gun", "love"]:
    print("  nearest to %-9s" % w, neighbours(w))

# ---------- feedforward classifier, SGD + backprop
# standardize the word vectors so FFN inputs have a sensible scale
mu, sd = emb.mean(0), emb.std(0) + 1e-6
emb_n = ((emb - mu) / sd).astype(np.float32)
Xtr_v = torch.tensor(np.stack([average_vector(t, vocab, emb_n) for t in Xtr]))
Xva_v = torch.tensor(np.stack([average_vector(t, vocab, emb_n) for t in Xva]))
ytr_t = torch.tensor(ytr)

w = torch.tensor([1.0, (len(ytr) - sum(ytr)) / sum(ytr)])   # class weights (imbalance)
ffn = FFN(DIM)
opt = torch.optim.SGD(ffn.parameters(), lr=0.05, momentum=0.9)
lossf = nn.CrossEntropyLoss(weight=w)
best_f1, best_state = -1, None
for epoch in range(40):
    ffn.train()
    perm = torch.randperm(len(Xtr_v))
    for k in range(0, len(perm), 64):
        idx = perm[k:k + 64]
        loss = lossf(ffn(Xtr_v[idx]), ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()     # backpropagation + SGD step
    ffn.eval()
    with torch.no_grad():
        f1 = f1_score(yva, ffn(Xva_v).argmax(-1).numpy(), average="macro")
    if f1 > best_f1:
        best_f1, best_state = f1, {k: v.clone() for k, v in ffn.state_dict().items()}
    if (epoch + 1) % 5 == 0:
        print("FFN epoch %2d  val macro-F1 %.4f" % (epoch + 1, f1))
print("best validation macro-F1 = %.4f" % best_f1)

torch.save({"vocab": vocab, "emb": emb_n, "ffn": best_state}, path("w2v_ffn.pt"))
print("saved models/w2v_ffn.pt")
