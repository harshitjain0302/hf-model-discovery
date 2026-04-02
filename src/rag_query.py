import requests
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_vectorstore():
    client = chromadb.PersistentClient(path="../data/chroma_db")
    collection = client.get_collection("model_cards")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    return collection, embedder

def retrieve(query, collection, embedder, n_results=5):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "model_id": meta["model_id"],
            "pipeline_tag": meta["pipeline_tag"],
            "downloads": meta["downloads"],
            "text": doc
        })
    return chunks

def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Model: {chunk['model_id']} (task: {chunk['pipeline_tag']}) ---\n"
        context += chunk["text"] + "\n"

    return f"""You are an AI model recommendation assistant. 
A user is looking for a machine learning model from HuggingFace.
Based only on the context below, recommend the most suitable model(s) and explain why.
Be specific — mention model names, their task type, and any relevant details from the context.

User query: {query}

Context from HuggingFace model cards:
{context}

Answer:"""

def ask(query, collection, embedder):
    chunks = retrieve(query, collection, embedder)
    prompt = build_prompt(query, chunks)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )
    answer = response.json()["response"]
    return answer, chunks

if __name__ == "__main__":
    collection, embedder = load_vectorstore()

    test_queries = [
        "lightweight model for sentiment analysis",
        "best model for speech recognition in english",
        "multilingual text embedding model",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        answer, chunks = ask(query, collection, embedder)
        print(f"\nRetrieved from: {[c['model_id'] for c in chunks]}")
        print(f"\nAnswer:\n{answer}")


# Then add your Anthropic API key to your `.env` file:
# ANTHROPIC_API_KEY=your_key_here