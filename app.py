"""
P719 - Customer Review Sentiment Analyzer (Streamlit app)

Run locally:   streamlit run app.py

Where the model is loaded from (first match wins):
  1. Streamlit secret  MODEL_SOURCE
  2. Environment var   MODEL_SOURCE
  3. "model_path" in deployment_config.json (default: ./model next to this file)
MODEL_SOURCE can be a local folder or a Hugging Face Hub repo id ("username/repo").
For a private Hub repo also set HF_TOKEN.
"""

import json
import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
CONFIG_FILE = APP_DIR / "deployment_config.json"

DEFAULT_CONFIG = {
    "best_model": "BERT",
    "model_path": "model",
    "max_length": 256,
    "labels": ["Negative", "Neutral", "Positive"],
    "metrics": {},
    "neutral_recall": None,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass  # fall back to defaults
    return cfg


CFG = load_config()
LABELS = CFG["labels"]
MAX_LENGTH = int(CFG["max_length"])
MAX_CHARS = 5000        # hard cap on characters per review (model reads max_length tokens)
MAX_BATCH_ROWS = 500    # CPU-friendly limit for CSV scoring
EMOJI = {"Negative": "😠", "Neutral": "😐", "Positive": "😊"}

EXAMPLES = {
    "Positive": "Great phone. The camera quality is excellent and the battery backup is very good.",
    "Negative": "Heating. Major heating problem during charging, very disappointed.",
    "Mixed / Neutral": "Average. Display is fine but the camera is just okay for the price.",
}

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")


def get_model_source() -> str:
    try:
        if "MODEL_SOURCE" in st.secrets:
            return str(st.secrets["MODEL_SOURCE"])
    except Exception:  # no secrets file available
        pass
    env_value = os.getenv("MODEL_SOURCE")
    if env_value:
        return env_value
    return str(APP_DIR / CFG["model_path"])


def is_hub_repo_id(value: str) -> bool:
    return re.fullmatch(r"[\w.-]+/[\w.-]+", value) is not None


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model (first start can take a minute)...")
def load_model(source: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.to(device).eval()
    return tokenizer, model, device


def predict(texts, tokenizer, model, device, batch_size=16):
    """Score a list of texts. Returns a list of dicts: label, confidence, probs."""
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:MAX_CHARS] for t in texts[start : start + batch_size]]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu()
        for p in probs:
            idx = int(p.argmax())
            rows.append(
                {
                    "label": id2label[idx],
                    "confidence": float(p[idx]),
                    "probs": {id2label[j]: float(p[j]) for j in range(p.shape[0])},
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
best = CFG["best_model"]
best_metrics = CFG["metrics"].get(best, {})

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** fine-tuned {best} (multilingual, cased)")
    if "Accuracy" in best_metrics:
        st.metric("Test accuracy", f"{best_metrics['Accuracy']:.1%}")
    if "F1-score (Macro)" in best_metrics:
        st.metric("Test macro F1", f"{best_metrics['F1-score (Macro)']:.3f}")
    st.markdown("---")
    st.markdown(
        "**Known limitations**\n"
        "- *Neutral* (3-star) reviews are the hardest class"
        + (f" (recall ≈ {CFG['neutral_recall']:.0%} on the test set)" if CFG["neutral_recall"] is not None else "")
        + ".\n"
        "- Trained on ~1,400 mobile-phone reviews; other product types may score worse.\n"
        f"- Only the first {MAX_LENGTH} tokens of a review are read."
    )

# --------------------------------------------------------------------------- #
# Load model (with clear error messages)
# --------------------------------------------------------------------------- #
source = get_model_source()
if not Path(source).exists() and not is_hub_repo_id(source):
    st.error(
        f"Model not found at `{source}`.\n\n"
        "Put the exported `model/` folder next to `app.py`, or set `MODEL_SOURCE` "
        "to a Hugging Face repo id such as `username/bert-sentiment`."
    )
    st.stop()

try:
    tokenizer, model, device = load_model(source)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model from `{source}`.\n\n{exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Main UI
# --------------------------------------------------------------------------- #
st.title("💬 Customer Review Sentiment Analyzer")
st.caption("Classifies a review as Negative, Neutral or Positive.")

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV)"])

with tab_single:
    example = st.selectbox("Load an example (optional)", ["—"] + list(EXAMPLES))
    review = st.text_area(
        "Customer review",
        value=EXAMPLES.get(example, ""),
        height=160,
        key=f"review_{example}",  # refreshes the box when the example changes
        placeholder="Type or paste a review here...",
    )

    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter some review text.")
        else:
            result = predict([review.strip()], tokenizer, model, device)[0]
            label, conf = result["label"], result["confidence"]

            st.subheader(f"{EMOJI.get(label, '')} {label}")
            st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf:.1%}")
            st.bar_chart(pd.DataFrame({"Probability": [result["probs"].get(l, 0.0) for l in LABELS]}, index=LABELS))

            if len(review) > MAX_CHARS:
                st.info(f"Review was cut to the first {MAX_CHARS} characters.")
            if conf < 0.60:
                st.info("Low confidence: the review may be mixed or ambiguous.")
            if label == "Neutral":
                st.info("Neutral is the model's weakest class, so double-check this result.")

with tab_batch:
    upload = st.file_uploader("Upload a CSV file that has a review-text column", type=["csv"])
    if upload is not None:
        try:
            data = pd.read_csv(upload, encoding_errors="replace")
        except Exception as exc:  # noqa: BLE001
            data = None
            st.error(f"Could not read the CSV file: {exc}")

        if data is not None and data.empty:
            st.warning("The uploaded CSV has no rows.")
        elif data is not None:
            text_col = st.selectbox("Column that contains the review text", list(data.columns))
            if len(data) > MAX_BATCH_ROWS:
                st.warning(f"Only the first {MAX_BATCH_ROWS} rows will be scored.")
                data = data.head(MAX_BATCH_ROWS).copy()

            if st.button("Run batch prediction"):
                texts = data[text_col].fillna("").astype(str).tolist()
                with st.spinner("Scoring reviews..."):
                    scored = predict(texts, tokenizer, model, device)
                data["predicted_sentiment"] = [r["label"] for r in scored]
                data["confidence"] = [round(r["confidence"], 4) for r in scored]

                st.dataframe(data)
                st.bar_chart(data["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0))
                st.download_button(
                    "Download results (CSV)",
                    data.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption("P719 · NLP · Fine-tuned on mobile-phone customer reviews (Samsung M12).")
