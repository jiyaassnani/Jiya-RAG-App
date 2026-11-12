import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Sentiment Analyzer")

st.title("My AI Sentiment Analyzer 🤖")
st.write("Developed by: jiya")
st.write("Roll Number: b040010ja")
st.write("Course: Foundations of Generative AI")
st.write("Foundations of Generative AI")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

def clear_text():
    st.session_state.user_text = ""

text = st.text_area(
    "Enter your sentence:",
    key="user_text"
)

st.button("Clear Text", on_click=clear_text)

if st.button("Analyze Sentiment"):
    if text.strip():
        with st.spinner("Analyzing..."):
            classifier = load_model()
            result = classifier(text, truncation=True)[0]

        st.success(f"Sentiment: {result['label']}")
        st.write(f"Confidence: {result['score']:.2%}")
    else:
        st.warning("Please enter some text.")