import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_cards(df):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for _, row in df.iterrows():
        splits = splitter.split_text(row["card_text"])
        for i, chunk in enumerate(splits):
            chunks.append({
                "model_id": row["model_id"],
                "pipeline_tag": row["pipeline_tag"],
                "downloads": row["downloads"],
                "chunk_index": i,
                "chunk_text": chunk,
            })

    return pd.DataFrame(chunks)

if __name__ == "__main__":
    df = pd.read_parquet("../data/model_cards_clean.parquet")
    chunks_df = chunk_cards(df)

    print(f"Total chunks: {len(chunks_df)}")
    print(f"Avg chunks per model: {len(chunks_df)/len(df):.1f}")
    print(f"\nSample chunk:\n{'-'*40}")
    print(chunks_df.iloc[5]['chunk_text'])
    print(f"\nChunk length stats:")
    print(chunks_df['chunk_text'].str.len().describe().round(0))

    chunks_df.to_parquet("../data/model_chunks.parquet", index=False)
    print(f"\nSaved. Shape: {chunks_df.shape}")