import numpy as np
import streamlit as st
import faiss
import PyPDF2
import docx

from sentence_transformers import SentenceTransformer
from groq import Groq

# Page settings
st.set_page_config(
    page_title="Jiya's RAG App",
    page_icon="📚"
)

st.title("📚 Open-Source RAG App with Groq")
st.caption("CS4106 — Foundations of Generative AI")

# Sidebar settings
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    model = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b"]
    )

    top_k = st.slider(
        "Chunks to retrieve (k)",
        1, 5, 3
    )

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Read uploaded files
def read_file(f):
    if f.name.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(f)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if f.name.lower().endswith(".docx"):
        document = docx.Document(f)
        return "\n".join(
            p.text for p in document.paragraphs
        )

    return f.read().decode("utf-8", "ignore")

# Split document into chunks
def chunk_text(text, size=150, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size-overlap):
        chunks.append(" ".join(words[i:i+size]))

        if i + size >= len(words):
            break

    return chunks

# Build FAISS index
def build_index(chunks):
    emb = embedder.encode(chunks)

    emb = emb / np.linalg.norm(
        emb, axis=1, keepdims=True
    )

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))

    return index

# Search relevant chunks
def search(index, chunks, query, k):
    q = embedder.encode([query])

    q = q / np.linalg.norm(
        q, axis=1, keepdims=True
    )

    scores, ids = index.search(
        q.astype("float32"),
        min(k, len(chunks))
    )

    return [
        (chunks[i], float(score))
        for score, i in zip(scores[0], ids[0])
        if 0 <= i < len(chunks)
    ]

# Upload documents
files = st.file_uploader(
    "Upload PDF / DOCX / TXT",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if files and st.button("Build knowledge base"):
    text = "\n".join(read_file(f) for f in files)
    chunks = chunk_text(text)

    if not chunks:
        st.error("No readable text found in the files.")
    else:
        st.session_state.chunks = chunks
        st.session_state.index = build_index(chunks)

        st.success(
            f"Indexed {len(chunks)} chunks "
            f"from {len(files)} file(s)."
        )

# Ask questions
question = st.text_input(
    "Ask a question about your documents:"
)

if st.button("Answer") and question:

    if not api_key:
        st.error("Enter your Groq API key in the sidebar.")

    elif "index" not in st.session_state:
        st.error(
            "Upload a document and click "
            "'Build knowledge base' first."
        )

    else:
        hits = search(
            st.session_state.index,
            st.session_state.chunks,
            question,
            top_k
        )

        context = "\n\n".join(
            doc for doc, _ in hits
        )

        prompt = (
            "Answer using ONLY the context below. "
            "If the answer is not there, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            st.subheader("Answer")
            st.write(response.choices[0].message.content)

            with st.expander("Sources (retrieved chunks)"):
                for i, (doc, score) in enumerate(hits, 1):
                    st.markdown(
                        f"**Chunk {i}** — Score: {score:.2f}"
                    )
                    st.write(doc)

        except Exception as e:
            st.error(f"Error: {e}")
