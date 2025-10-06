# AI Sentiment Text Generator (Remote ML Internship Assessment)

## Summary
This project detects the sentiment of a user prompt and generates a paragraph aligned with that sentiment. Built with Hugging Face Transformers and Streamlit for the frontend.

## Files
- `app.py`: Streamlit app (frontend + glue).
- `sentiment_gen.py`: sentiment classification and generation functions.
- `requirements.txt`: Python dependencies.
- `README.md`: this file.
- (Optional) `generated_paragraph.txt`: sample outputs you may create and share.

## Methodology
1. **Sentiment Classification**
   - Model: `distilbert-base-uncased-finetuned-sst-2-english` (pretrained on SST-2).
   - Approach: use HF `pipeline("sentiment-analysis")`. Low-confidence results (score < 0.65) are mapped to `NEUTRAL`.

2. **Text Generation**
   - Model: `gpt2` (default) or `distilgpt2`. Uses conditional prompt engineering:
     - Example conditioning prompt: `"Write a positive paragraph about {user_prompt}\n\n"`
   - Generation parameters: sampling (top-p), temperature; word count approx via token estimates.
   - Post-processing truncates text to requested approximate word limit.

3. **Frontend**
   - Streamlit app (`app.py`) with:
     - Prompt box
     - Auto-detect sentiment OR manual override
     - Word count slider
     - Model choice and sampling toggle
     - Download feature for generated text

## Dataset(s) used
- **SST-2**: the sentiment classifier is from a `distilbert` model fine-tuned on SST-2.
- The generation model (`gpt2`) is pretrained on WebText-like data from OpenAI; no additional fine-tuning performed.

## How to run (local)
1. Create a virtualenv and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
