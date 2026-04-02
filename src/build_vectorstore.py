import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 256

def build_vectorstore(chunks_path):
    df = pd.read_parquet(chunks_path)

    # Drop chunks that are too short
    df = df[df["chunk_text"].str.len() > 20].reset_index(drop=True)
    print(f"Embedding {len(df)} chunks...")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up Chroma (persisted locally)
    client = chromadb.PersistentClient(path="../data/chroma_db")

    # Delete collection if rebuilding
    try:
        client.delete_collection("model_cards")
    except:
        pass
    collection = client.create_collection("model_cards")

    # Embed and insert in batches
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[i:i+BATCH_SIZE]
        embeddings = model.encode(batch["chunk_text"].tolist()).tolist()

        collection.add(
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=batch["chunk_text"].tolist(),
            metadatas=batch[["model_id", "pipeline_tag", "downloads"]].fillna("unknown").astype(str).to_dict("records"),
        )

    print(f"\nVector store built. Total vectors: {collection.count()}")
    return collection

if __name__ == "__main__":
    collection = build_vectorstore("../data/model_chunks.parquet")

    # Quick sanity check query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "lightweight model for text classification"
    query_embedding = model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=3)

    print(f"\nTest query: '{query}'")
    print("-" * 40)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\nResult {i+1}: {meta['model_id']} ({meta['pipeline_tag']})")
        print(f"Snippet: {doc[:150]}...")