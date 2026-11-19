Test set: OLID/OffensEval-2019 test, n=860. Latency over 200 messages.

Hardware: arm64 arm Darwin - CPU, 4 threads, batch size 1

| model                          |   macro_F1 |   accuracy |   size_MB |   latency_ms_mean |   latency_ms_p95 |
|:-------------------------------|-----------:|-----------:|----------:|------------------:|-----------------:|
| Rule (lexicon + edit distance) |     0.7086 |     0.7884 |      0    |              0.95 |             2.03 |
| TF-IDF + LogReg                |     0.7179 |     0.7733 |      0.98 |              0.25 |             0.29 |
| word2vec + FFN                 |     0.6672 |     0.7093 |      5.36 |              0.06 |             0.1  |
| DistilBERT FP32                |     0.7862 |     0.8372 |    267.83 |             13.24 |            16.37 |
| DistilBERT INT8 (quantized)    |     0.7819 |     0.8395 |    138.72 |             12.69 |            21.3  |