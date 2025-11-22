# fine_tune_cpu_friendly_debug.py
print("Script started successfully!")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# -----------------------------
# Load your new dataset
# -----------------------------
print("[DEBUG] Loading dataset...")
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
print(f"[DEBUG] Dataset loaded. Number of examples: {len(dataset)}")
print(f"[DEBUG] First example: {dataset[0]}")

# Format each example
def format_examples(example):
    return {"text": f"User: {example['instruction']}\nAI: {example['response']}"}

print("[DEBUG] Formatting examples...")
dataset = dataset.map(format_examples)
print(f"[DEBUG] First formatted example: {dataset[0]}")

# -----------------------------
# Load tokenizer and model
# -----------------------------
model_name = "distilgpt2"
print(f"[DEBUG] Loading tokenizer and model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Important for causal LM
model = AutoModelForCausalLM.from_pretrained(model_name)
print("[DEBUG] Tokenizer and model loaded.")

# -----------------------------
# Tokenize dataset
# -----------------------------
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=256)

print("[DEBUG] Tokenizing dataset...")
tokenized = dataset.map(tokenize, batched=True)
print(f"[DEBUG] First tokenized example: {tokenized[0]}")

# -----------------------------
# Training setup
# -----------------------------
training_args = TrainingArguments(
    output_dir="./finetuned_bot",
    per_device_train_batch_size=1,  # CPU friendly
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=5e-5,
    report_to="none",  # avoids missing logger issues
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# -----------------------------
# Train the model
# -----------------------------
print("[DEBUG] Starting training...")
trainer.train()
print("[DEBUG] Training finished.")

# -----------------------------
# Save the fine-tuned model
# -----------------------------
print("[DEBUG] Saving model...")
trainer.save_model("./finetuned_bot")
tokenizer.save_pretrained("./finetuned_bot")
print("[DEBUG] Model saved.")

# -----------------------------
# Test the bot
# -----------------------------
from transformers import pipeline

print("[DEBUG] Initializing generation pipeline...")
bot = pipeline("text-generation", model="./finetuned_bot", tokenizer=tokenizer)

prompt = "User: Explain NLP\nAI:"
print(f"[DEBUG] Generating output for prompt: {prompt}")
output = bot(prompt, max_new_tokens=100)
print("[DEBUG] Output generated:")
print(output[0]["generated_text"])
