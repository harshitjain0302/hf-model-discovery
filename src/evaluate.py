import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import json

# ── Test queries (unseen during fine-tuning) ───────────────
TEST_QUERIES = [
    "lightweight model for sentiment analysis on social media",
    "best english speech recognition model",
    "multilingual sentence embedding model",
    "image classification model for medical imaging",
    "small model for named entity recognition",
]

# ── Load RAG components ────────────────────────────────────
def load_rag():
    client = chromadb.PersistentClient(path="../data/chroma_db")
    collection = client.get_collection("model_cards")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedder

def rag_answer(query, collection, embedder):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    
    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\nModel: {meta['model_id']} (task: {meta['pipeline_tag']})\n{doc}\n"
    
    prompt = f"""You are an AI model recommendation assistant.
Based only on the context below, recommend the most suitable model and explain why.

Query: {query}
Context: {context}
Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()

# ── Load fine-tuned model ──────────────────────────────────
def load_finetuned():
    base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base)
    model = PeftModel.from_pretrained(model, "../models/hf-model-advisor-lora")
    model.eval()
    return model, tokenizer

def finetuned_answer(query, model, tokenizer):
    prompt = f"""### Instruction:
You are a HuggingFace model recommendation assistant.
Answer the following question about ML models.

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

def base_answer(query, tokenizer, base_model):
    prompt = f"Question: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()

# ── Run evaluation ─────────────────────────────────────────
if __name__ == "__main__":
    print("Loading RAG...")
    collection, embedder = load_rag()

    print("Loading fine-tuned model...")
    ft_model, tokenizer = load_finetuned()

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    base_model.eval()

    results = []

    for query in tqdm(TEST_QUERIES):
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        rag = rag_answer(query, collection, embedder)
        ft  = finetuned_answer(query, ft_model, tokenizer)
        base = base_answer(query, tokenizer, base_model)

        print(f"\nRAG:\n{rag[:300]}")
        print(f"\nFine-tuned:\n{ft[:300]}")
        print(f"\nBase:\n{base[:300]}")

        results.append({
            "query": query,
            "rag_answer": rag,
            "finetuned_answer": ft,
            "base_answer": base,
        })

    df = pd.DataFrame(results)
    df.to_parquet("../data/eval_results.parquet", index=False)
    df.to_csv("../data/eval_results.csv", index=False)
    print("\nEval results saved.")