# Run this on Google Colab (free T4 GPU)
# Upload your qa_pairs.parquet to Colab first

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ── 1. Load data ───────────────────────────────────────────
df = pd.read_parquet("qa_pairs.parquet")
df = df[["question", "answer"]].dropna()

def format_prompt(row):
    return f"""### Instruction:
You are a HuggingFace model recommendation assistant.
Answer the following question about ML models.

### Question:
{row['question']}

### Answer:
{row['answer']}"""

df["text"] = df.apply(format_prompt, axis=1)
dataset = Dataset.from_pandas(df[["text"]])
print(f"Training on {len(dataset)} examples")

# ── 2. Load base model ─────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # small enough for free Colab

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

# ── 3. LoRA config ─────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 4. Train ───────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./hf-model-advisor-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

# ── 5. Save ────────────────────────────────────────────────
model.save_pretrained("./hf-model-advisor-lora")
tokenizer.save_pretrained("./hf-model-advisor-lora")
print("Fine-tuned model saved.")