"""Real-Time Chat Moderation - Gradio app.
Each message is labeled live (clean / offensive) with a confidence score,
offensive messages are masked, and a side panel logs the response time of every
message. Default model = quantized DistilBERT (INT8, CPU). Run:  python app.py"""
import html
import time

import gradio as gr
import pandas as pd
import torch

from src.models import load_all
from src.lexicon import match_tokens
from src.preprocess import normalize, tokenize

torch.set_num_threads(4)
BUDGET_MS = 50
THRESHOLD = 0.5

MODELS = {m.name: m for m in load_all()}
NAMES = list(MODELS)
DEFAULT = [n for n in NAMES if "INT8" in n][0]

# warm-up: the first PyTorch calls are slow, so run every model a few times at start-up
for _m in MODELS.values():
    for _ in range(5):
        _m.predict("warm up message for the moderation models")


def mask(text):
    """Mask lexicon words with asterisks; if the model flagged the message but no
    lexicon word is found, mask the whole message."""
    hits = set(match_tokens(text))
    out, masked = [], False
    for w in text.split():
        if hits and set(tokenize(normalize(w))) & hits:
            out.append(w[0] + "*" * (len(w) - 1)); masked = True
        else:
            out.append(w)
    return " ".join(out) if masked else "*" * min(len(text), 30) + "  (hidden)"


def bubble(text, label, p, ms, model):
    color = "#fde2e2" if label == "offensive" else "#e3f6e8"
    shown = mask(text) if label == "offensive" else text
    return (f"<div style='background:{color};color:#111;border-radius:10px;padding:8px 12px;margin:6px 0'>"
            f"<b>{html.escape(shown)}</b><br><small>{label.upper()} · confidence {max(p, 1-p):.2f} · "
            f"{ms:.1f} ms · {html.escape(model)}</small></div>")


def send(text, model_name, chat, log):
    if not text.strip():
        return "", chat, log, None
    m = MODELS[model_name]
    t0 = time.perf_counter()
    p = m.predict(text)
    ms = (time.perf_counter() - t0) * 1000
    label = "offensive" if p >= THRESHOLD else "clean"
    chat = chat + bubble(text, label, p, ms, model_name)
    log = pd.concat([pd.DataFrame([{"#": len(log) + 1, "model": model_name, "label": label,
                                    "p(offensive)": round(p, 3), "latency_ms": round(ms, 1),
                                    f"<{BUDGET_MS}ms": "yes" if ms < BUDGET_MS else "NO"}]), log],
                    ignore_index=True)
    # side-by-side comparison of all models for this message
    comp = []
    for n, mm in MODELS.items():
        t0 = time.perf_counter(); pp = mm.predict(text); tt = (time.perf_counter() - t0) * 1000
        comp.append({"model": n, "label": "offensive" if pp >= THRESHOLD else "clean",
                     "p(offensive)": round(pp, 3), "latency_ms": round(tt, 1)})
    return "", chat, log, pd.DataFrame(comp)


EMPTY_LOG = pd.DataFrame(columns=["#", "model", "label", "p(offensive)", "latency_ms", f"<{BUDGET_MS}ms"])

with gr.Blocks(title="Real-Time Chat Moderation") as demo:
    gr.Markdown("## 🎮 Real-Time Chat Moderation\nEnglish gaming-chat moderation on CPU · "
                f"latency budget **{BUDGET_MS} ms / message** · default model: quantized DistilBERT (INT8)")
    chat_state = gr.State("")
    log_state = gr.State(EMPTY_LOG)
    with gr.Row():
        with gr.Column(scale=3):
            chat_html = gr.HTML("<i>No messages yet.</i>")
            with gr.Row():
                box = gr.Textbox(placeholder="Type a chat message and press Enter…", show_label=False, scale=5)
                btn = gr.Button("Send", variant="primary", scale=1)
            gr.Markdown("**All models on the last message**")
            comp = gr.Dataframe(interactive=False)
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(NAMES, value=DEFAULT, label="Moderation model")
            gr.Markdown("**Response-time log**")
            log_df = gr.Dataframe(EMPTY_LOG, interactive=False)

    def _send(text, model_name, chat, log):
        _, chat, log, c = send(text, model_name, chat, log)
        return "", chat, chat, log, log, c

    outs = [box, chat_state, chat_html, log_state, log_df, comp]
    box.submit(_send, [box, model_dd, chat_state, log_state], outs)
    btn.click(_send, [box, model_dd, chat_state, log_state], outs)

if __name__ == "__main__":
    demo.launch()
