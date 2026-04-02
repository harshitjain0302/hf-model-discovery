from huggingface_hub import ModelCard
import pandas as pd
from tqdm import tqdm
import time

def fetch_model_cards(parquet_path, limit=500):
    df = pd.read_parquet(parquet_path)
    df = df.head(limit)

    cards = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            card = ModelCard.load(row["model_id"])
            cards.append({
                "model_id": row["model_id"],
                "pipeline_tag": row["pipeline_tag"],
                "downloads": row["downloads"],
                "tags": row["tags"],
                "card_text": card.text,
            })
            time.sleep(0.1)  # gentle rate limiting
        except Exception as e:
            failed.append(row["model_id"])

    print(f"\nFetched: {len(cards)} | Failed: {len(failed)}")
    if failed:
        print("Failed models:", failed[:5])

    return pd.DataFrame(cards)

if __name__ == "__main__":
    df = fetch_model_cards("../data/models_metadata.parquet", limit=500)
    df.to_parquet("../data/model_cards.parquet", index=False)
    print(df[["model_id", "card_text"]].head(3))