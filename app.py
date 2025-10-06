import streamlit as st
from sentiment_gen import analyze_sentiment, generate_text
import torch


st.set_page_config(page_title="AI Sentiment Text Generator", layout="centered")

st.title("🧠 AI Text Generator (Sentiment-Based)")
st.write(
    "Enter a topic or sentence below. The system will detect its sentiment and generate "
    "a paragraph matching that tone."
)


prompt = st.text_area("✍️ Enter your topic or prompt:", height=150, placeholder="e.g., The new park in my city looks beautiful...")

col1, col2 = st.columns(2)
with col1:
    detect_btn = st.button("🔍 Detect Sentiment")
with col2:
    generate_btn = st.button("⚙️ Generate Text")


manual_choice = st.selectbox(
    "Override Sentiment (Optional):",
    ["Auto-detect", "POSITIVE", "NEGATIVE", "NEUTRAL"],
    index=0
)
max_words = st.slider("Approx. word count:", min_value=50, max_value=400, value=120, step=10)
model_choice = st.selectbox("Choose model:", ["gpt2", "distilgpt2"], index=0)
creative_mode = st.checkbox("Enable creative mode (sampling)", value=True)

st.caption(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'} device")


if detect_btn and prompt.strip():
    with st.spinner("Analyzing sentiment..."):
        device = 0 if torch.cuda.is_available() else -1
        result = analyze_sentiment(prompt, device=device)
    st.success(f"Detected Sentiment: **{result['label']}** (Confidence: {result['score']:.2f})")


if generate_btn and prompt.strip():
    with st.spinner("Generating sentiment-aligned text..."):
        device = 0 if torch.cuda.is_available() else -1

        if manual_choice != "Auto-detect":
            chosen_sentiment = manual_choice
        else:
            result = analyze_sentiment(prompt, device=device)
            chosen_sentiment = result["label"]

        generated_text = generate_text(
            prompt,
            sentiment=chosen_sentiment,
            max_words=max_words,
            model_name=model_choice,
            device=device,
            do_sample=creative_mode
        )

    st.subheader(f"Generated ({chosen_sentiment}):")
    st.write(generated_text)
    st.download_button(
        "💾 Download Paragraph",
        data=generated_text,
        file_name="generated_text.txt",
        mime="text/plain"
    )

elif generate_btn and not prompt.strip():
    st.warning("Please enter a prompt before generating.")

with st.expander("💡 Example Prompts"):
    st.markdown("""
    - *The impact of remote work on productivity.*
    - *A bad experience with online shopping.*
    - *My favorite travel memory in the mountains.*
    """)

st.markdown("---")
st.info("Tip: For creative or longer text, enable *creative mode* and increase the word count.")
