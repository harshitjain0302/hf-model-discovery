import streamlit as st
import chromadb
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="HF Model Discovery",
    page_icon="🤗",
    layout="wide"
)

st.title("🤗 HuggingFace Model Discovery")
st.caption("Find the right ML model using natural language — powered by RAG + Fine-tuned LLM")

# ── Load models (cached so they don't reload on every query)
@st.cache_resource
def load_rag():
    client = chromadb.PersistentClient(path="../data/chroma_db")
    collection = client.get_collection("model_cards")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedder

@st.cache_resource
def load_finetuned():
    base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base)
    model = PeftModel.from_pretrained(model, "../models/hf-model-advisor-lora")
    model.eval()
    return model, tokenizer

# ── Query functions ────────────────────────────────────────
def rag_answer(query, collection, embedder):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)

    context = ""
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\nModel: {meta['model_id']} (task: {meta['pipeline_tag']})\n{doc}\n"
        retrieved.append(meta["model_id"])

    prompt = f"""You are an AI model recommendation assistant.
Based only on the context below, recommend the most suitable model and explain why.
Be specific — mention model names and relevant details.

Query: {query}
Context: {context}
Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip(), retrieved

def finetuned_answer(query, model, tokenizer):
    prompt = f"""### Instruction:
You are a HuggingFace model recommendation assistant.

### Question:
{query}

### Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Answer:")[-1].strip()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Answer mode",
        ["RAG (recommended)", "Fine-tuned model", "Compare both"],
        index=0
    )
    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "lightweight model for sentiment analysis",
        "best english speech recognition model",
        "multilingual sentence embedding model",
        "small model for named entity recognition",
        "image classification model for medical imaging",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

# ── Main input ─────────────────────────────────────────────
query = st.text_input(
    "What kind of model are you looking for?",
    value=st.session_state.get("query", ""),
    placeholder="e.g. lightweight model for text classification in Spanish",
)

if st.button("Search", type="primary") and query:
    collection, embedder = load_rag()

    if mode in ["RAG (recommended)", "Compare both"]:
        with st.spinner("Searching model cards..."):
            rag_ans, retrieved = rag_answer(query, collection, embedder)

    if mode in ["Fine-tuned model", "Compare both"]:
        with st.spinner("Running fine-tuned model..."):
            ft_model, tokenizer = load_finetuned()
            ft_ans = finetuned_answer(query, ft_model, tokenizer)

    # ── Display results ────────────────────────────────────
    if mode == "RAG (recommended)":
        st.subheader("Recommendation")
        st.markdown(rag_ans)
        with st.expander("Retrieved from these model cards"):
            for m in retrieved:
                st.markdown(f"- [{m}](https://huggingface.co/{m})")

    elif mode == "Fine-tuned model":
        st.subheader("Recommendation")
        st.markdown(ft_ans)

    elif mode == "Compare both":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RAG answer")
            st.markdown(rag_ans)
            with st.expander("Retrieved from"):
                for m in retrieved:
                    st.markdown(f"- [{m}](https://huggingface.co/{m})")
        with col2:
            st.subheader("Fine-tuned answer")
            st.markdown(ft_ans)

        st.info("RAG retrieves answers from real model cards. Fine-tuned learned from 206 training examples. RAG wins on factual accuracy; fine-tuning would improve with more data.")