"""Evaluate all models on the SAME held-out test set (860 tweets).
Metrics: macro-F1, accuracy, model size (MB), latency per message
(batch size 1, CPU, mean over N_RUNS messages after warm-up).
Writes results/results.csv, results/results.md and results/errors.md."""
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.data import load_splits
from src.models import load_all, size_mb

N_RUNS = 200
torch.set_num_threads(4)
os.makedirs("results", exist_ok=True)

Xte, yte = load_splits()["test"]
hardware = f"{platform.machine()} {platform.processor() or ''} {platform.system()} - CPU, {torch.get_num_threads()} threads, batch size 1"
print("hardware:", hardware)

rows, all_preds = [], {}
for m in load_all():
    probs = np.array([m.predict(t) for t in Xte])
    preds = (probs >= 0.5).astype(int)
    all_preds[m.name] = (preds, probs)

    for t in Xte[:10]:                       # warm-up
        m.predict(t)
    times = []
    for t in Xte[:N_RUNS]:
        t0 = time.perf_counter(); m.predict(t); times.append((time.perf_counter() - t0) * 1000)

    row = {"model": m.name,
           "macro_F1": round(f1_score(yte, preds, average="macro"), 4),
           "accuracy": round(accuracy_score(yte, preds), 4),
           "size_MB": round(size_mb(m), 2),
           "latency_ms_mean": round(float(np.mean(times)), 2),
           "latency_ms_p95": round(float(np.percentile(times, 95)), 2)}
    rows.append(row)
    print(row, "\n  confusion matrix [[TN FP][FN TP]] =", confusion_matrix(yte, preds).tolist())

df = pd.DataFrame(rows)
df.to_csv("results/results.csv", index=False)
with open("results/results.md", "w") as f:
    f.write(f"Test set: OLID/OffensEval-2019 test, n={len(yte)}. Latency over {N_RUNS} messages.\n\n")
    f.write(f"Hardware: {hardware}\n\n")
    f.write(df.to_markdown(index=False))
print("\n", df.to_markdown(index=False))

# failure examples of the deployed (quantized) model, for the Conclusion slide
name = [n for n in all_preds if "INT8" in n][0]
preds, probs = all_preds[name]
wrong = [(Xte[i], yte[i], probs[i]) for i in range(len(yte)) if preds[i] != yte[i]]
with open("results/errors.md", "w") as f:
    f.write(f"# {name}: {len(wrong)} errors on {len(yte)} test messages\n\n")
    for text, y, p in wrong[:25]:
        f.write(f"- true={'offensive' if y else 'clean'}  p(offensive)={p:.2f}  :: {text}\n")
print("wrote results/results.csv, results/results.md, results/errors.md")
