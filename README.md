# P719 – Customer Review Sentiment Analyzer

Fine-tuned multilingual BERT classifier (Negative / Neutral / Positive) served with Streamlit.

## Folder layout
```
deployment/
├── app.py                    # Streamlit app
├── deployment_config.json    # best model name, labels, test metrics (written by the notebook)
├── requirements.txt
└── model/                    # exported model + tokenizer (created by the notebook, section 13)
```

## 1. Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 2. Host the model on the Hugging Face Hub (recommended)
GitHub rejects files over 100 MB and the model is ~700 MB, so upload it to the Hub:
```python
from huggingface_hub import HfApi
api = HfApi()                                   # run `huggingface-cli login` first
api.create_repo("YOUR_USERNAME/bert-sentiment", exist_ok=True)
api.upload_folder(folder_path="model", repo_id="YOUR_USERNAME/bert-sentiment")
```

## 3. Deploy
**Streamlit Community Cloud** – push `app.py`, `requirements.txt`, `deployment_config.json` to GitHub, create the app,
then add this in *Settings → Secrets*:
```toml
MODEL_SOURCE = "YOUR_USERNAME/bert-sentiment"
```
Community Cloud has ~1 GB RAM; if the app runs out of memory while loading BERT, use **Hugging Face Spaces**
(Streamlit SDK, more RAM) with the same files, or deploy DistilBERT instead.

**Hugging Face Spaces** – create a Streamlit Space, upload the same files, and put this at the top of `README.md`:
```yaml
---
title: Review Sentiment Analyzer
sdk: streamlit
app_file: app.py
---
```
Set `MODEL_SOURCE` as a Space variable/secret.

## Notes
- The first request after a cold start is slow (model loading); later requests are fast.
- Neutral (3-star) reviews are the weakest class; see section 12 of the notebook.
