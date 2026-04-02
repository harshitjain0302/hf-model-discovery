import pandas as pd
import requests
import json
from tqdm import tqdm

def generate_qa(model_id, pipeline_tag, card_text, num_pairs=3):
    prompt = f"""You are creating training data for a model recommendation system.
Given this HuggingFace model card, generate {num_pairs} question-answer pairs.
Questions should be things a data scientist might ask when searching for a model.
Answers should recommend this specific model and explain why based on the card.

Model: {model_id}
Task: {pipeline_tag}
Model card excerpt:
{card_text[:1000]}

Respond ONLY with a JSON array like this, no other text:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    
    raw = response.json()["response"].strip()
    # Extract JSON array from response
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    pairs = json.loads(raw[start:end])
    for p in pairs:
        p["model_id"] = model_id
        p["pipeline_tag"] = pipeline_tag
    return pairs

if __name__ == "__main__":
    df = pd.read_parquet("../data/model_cards_clean.parquet")
    
    # Use top 100 models — enough for fine-tuning, fast to generate
    df = df.head(100)
    
    all_pairs = []
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            pairs = generate_qa(
                row["model_id"],
                str(row["pipeline_tag"]),
                row["card_text"]
            )
            all_pairs.extend(pairs)
        except Exception as e:
            failed.append(row["model_id"])

    print(f"\nGenerated {len(all_pairs)} Q&A pairs | Failed: {len(failed)}")
    
    qa_df = pd.DataFrame(all_pairs)
    qa_df.to_parquet("../data/qa_pairs.parquet", index=False)
    
    print("\nSample pairs:")
    print(qa_df[["question", "answer"]].head(3).to_string())