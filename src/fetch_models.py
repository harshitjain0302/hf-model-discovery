from huggingface_hub import list_models
import pandas as pd
from tqdm import tqdm

def fetch_model_metadata(limit=5000):
    models = []
    for model in tqdm(list_models(limit=limit, sort="downloads", direction=-1)):
        models.append({
            "model_id": model.id,
            "author": model.author,
            "downloads": model.downloads,
            "likes": model.likes,
            "tags": model.tags,
            "pipeline_tag": model.pipeline_tag,
            "last_modified": model.last_modified,
        })
    return pd.DataFrame(models)

if __name__ == "__main__":
    df = fetch_model_metadata(limit=5000)
    df.to_parquet("../data/models_metadata.parquet", index=False)
    print(df.head())
    print(f"\nShape: {df.shape}")