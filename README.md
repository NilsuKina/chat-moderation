# Real-Time Chat Moderation (SWE015 – Use Case 4)

Flags offensive English chat messages live on a CPU, with four models of increasing
sophistication evaluated on the same held-out test set, and a Gradio app that runs the
**quantized Transformer (DistilBERT, dynamic INT8)** by default.

| # | Model | Area |
|---|-------|------|
| 0 | Rule baseline: regex normalization + offensive lexicon matched with minimum edit distance | B |
| 1 | TF-IDF (1–2 grams) + logistic regression | B |
| 2 | Own word2vec (skip-gram, negative sampling) → averaged vectors → feedforward net trained with SGD + backprop | A |
| 3 | DistilBERT fine-tuned for classification (WordPiece, self-attention) | C |
| 4 | DistilBERT, post-training dynamic INT8 quantization (CPU) | D |

**Data:** OLID / OffensEval-2019 English tweets (Zampieri et al., 2019), via Hugging Face
`cardiffnlp/tweet_eval` (config `offensive`, Barbieri et al., 2020). Splits: train 11,916 /
validation 1,324 / test 860. Labels: clean (NOT) / offensive (OFF).

## Setup

Requires Python 3.10–3.12 (tested on macOS, Apple Silicon).

```bash
git clone https://github.com/NilsuKina/chat-moderation.git && cd chat-moderation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (single command)

```bash
bash run_all.sh && python app.py
```

`run_all.sh` downloads the dataset and `distilbert-base-uncased`, trains all models, quantizes
DistilBERT and evaluates every model on the same test set (≈10–15 min on a laptop; the laptop is
slow while DistilBERT trains). `app.py` then starts the Gradio app at http://127.0.0.1:7860.
Model weights are written to `models/` and are not stored in the repository (too large);
they are re-created by `run_all.sh`. Next time, `python app.py` alone starts the app.

## Files

```
src/preprocess.py   regex normalization (leetspeak, repeats, URLs, @user, Turkish chars), tokenizer, edit distance
src/lexicon.py      rule baseline: lexicon + edit-distance matcher
src/data.py         dataset loading (same test split for all models)
src/models.py       model classes + common predictor interface
train_classical.py  TF-IDF + logistic regression
train_w2v_ffn.py    own word2vec + feedforward network (SGD)
train_bert.py       DistilBERT fine-tuning
quantize.py         dynamic INT8 quantization
evaluate.py         macro-F1, size, latency (≥100 runs) -> results/
app.py              Gradio application
```

## Results

Same held-out test set (860 tweets). MacBook Air M2, CPU only, 4 threads, batch size 1,
latency = mean over 200 messages after warm-up. Produced by `evaluate.py` (`results/results.md`).

| Model | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | p95 (ms) |
|---|---|---|---|---|---|
| Rule (lexicon + edit distance) | 0.7086 | 0.7884 | <0.01 | 0.95 | 2.03 |
| TF-IDF + LogReg | 0.7179 | 0.7733 | 0.98 | 0.25 | 0.29 |
| word2vec + FFN | 0.6672 | 0.7093 | 5.36 | 0.06 | 0.10 |
| DistilBERT FP32 | 0.7862 | 0.8372 | 267.83 | 13.24 | 16.37 |
| **DistilBERT INT8 (used in app)** | 0.7819 | 0.8395 | 138.72 | 12.69 | 21.30 |

DistilBERT was fine-tuned for 1 epoch on the M2 GPU (MPS); numbers can vary slightly between runs.
