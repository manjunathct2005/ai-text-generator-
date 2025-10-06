from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import math


_sentiment_analyzer = None
_text_model = None
_text_tokenizer = None


def _load_sentiment_model(device=-1):
    """Load or return the cached sentiment-analysis pipeline."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
    return _sentiment_analyzer


def analyze_sentiment(text, device=-1):
    """
    Detects sentiment of the input text.
    Returns a dictionary: {"label": "POSITIVE"/"NEGATIVE"/"NEUTRAL", "score": confidence_score}
    """
    model = _load_sentiment_model(device=device)
    result = model(text[:1000])[0]  

    label = result["label"]
    confidence = float(result["score"])


    if confidence < 0.65:
        label = "NEUTRAL"

    return {"label": label, "score": confidence}


def _load_text_model(model_name="gpt2", device=-1):
    """Load or return the cached GPT-like model and tokenizer for generation."""
    global _text_model, _text_tokenizer

    if _text_model is None or _text_tokenizer is None:
        _text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _text_tokenizer.pad_token is None:
            _text_tokenizer.pad_token = _text_tokenizer.eos_token

        _text_model = AutoModelForCausalLM.from_pretrained(model_name)

        if device >= 0 and torch.cuda.is_available():
            _text_model = _text_model.to(f"cuda:{device}")

    return _text_model, _text_tokenizer


def generate_text(
    prompt,
    sentiment=None,
    max_words=150,
    model_name="gpt2",
    device=-1,
    seed=None,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
):

    if seed is not None:
        torch.manual_seed(seed)

    device = 0 if (device >= 0 and torch.cuda.is_available()) else -1


    if sentiment is None:
        sentiment = analyze_sentiment(prompt, device=device)["label"]

    model, tokenizer = _load_text_model(model_name=model_name, device=device)


    sentiment_prefix = {
        "POSITIVE": "Write a positive paragraph about",
        "NEGATIVE": "Write a negative paragraph about",
        "NEUTRAL": "Write an informative and neutral paragraph about"
    }.get(sentiment.upper(), "Write a neutral paragraph about")

    conditioned_prompt = f"{sentiment_prefix} {prompt.strip()}:\n\n"


    token_limit = max(30, int(max_words * 1.5))
    input_ids = tokenizer.encode(conditioned_prompt, return_tensors="pt")

    if torch.cuda.is_available() and device >= 0:
        input_ids = input_ids.to(model.device)

    output = model.generate(
        input_ids=input_ids,
        max_length=min(1024, input_ids.shape[1] + token_limit),
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    if text.startswith(conditioned_prompt):
        text = text[len(conditioned_prompt):]

    # Limit to desired word count
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        if not text.endswith((".", "!", "?")):
            text += "."

    return text.strip()
