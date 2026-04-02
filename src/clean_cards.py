import pandas as pd
import re

def clean_card_text(text):
    if not isinstance(text, str):
        return None
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # Remove very short cards (likely empty/placeholder)
    if len(text.strip()) < 100:
        return None
    return text.strip()

def explore(df):
    print(f"Total cards: {len(df)}")
    print(f"Null card_text: {df['card_text'].isna().sum()}")
    print(f"\nPipeline tag distribution:")
    print(df['pipeline_tag'].value_counts().head(10))
    print(f"\nAvg card length (chars): {df['card_text'].dropna().str.len().mean():.0f}")
    print(f"Min: {df['card_text'].dropna().str.len().min()}")
    print(f"Max: {df['card_text'].dropna().str.len().max()}")

if __name__ == "__main__":
    df = pd.read_parquet("../data/model_cards.parquet")

    df["card_text"] = df["card_text"].apply(clean_card_text)
    df = df.dropna(subset=["card_text"]).reset_index(drop=True)

    explore(df)

    df.to_parquet("../data/model_cards_clean.parquet", index=False)
    print(f"\nCleaned dataset saved. Shape: {df.shape}")