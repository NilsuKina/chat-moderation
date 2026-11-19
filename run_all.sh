#!/usr/bin/env bash
# Trains every model, quantizes the Transformer and evaluates all models on the same test set.
set -e
python train_classical.py
python train_w2v_ffn.py
python train_bert.py
python quantize.py
python evaluate.py
