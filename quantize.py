"""Model 4 (Area D): post-training DYNAMIC INT8 quantization of the fine-tuned
DistilBERT. All nn.Linear layers (attention Q/K/V/out, feed-forward, classifier)
get int8 weights; activations are quantized on the fly at inference. CPU only."""
import os
import torch
from transformers import AutoModelForSequenceClassification

from src.models import path  # also selects the qnnpack engine on Apple Silicon

print("quantization engine:", torch.backends.quantized.engine)
fp32 = AutoModelForSequenceClassification.from_pretrained(path("bert_fp32")).to("cpu").eval()
int8 = torch.ao.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(int8, path("bert_int8.pt"))

mb = lambda p: os.path.getsize(p) / 1e6
print("FP32 size: %.1f MB" % mb(path("bert_fp32/model.safetensors")))
print("INT8 size: %.1f MB" % mb(path("bert_int8.pt")))
