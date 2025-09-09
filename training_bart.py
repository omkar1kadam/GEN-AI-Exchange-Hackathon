# Importing necessary libraries
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Step 1: Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
print(dataset)

# Step 2: Load the tokenizer for BART
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Step 3: Preprocess the data (tokenize inputs and summaries)
def preprocess_function(examples):
    inputs = examples["article"]  # The article column
    targets = examples["highlights"]  # The summary column
    
    # Tokenize the inputs and summaries (truncating and padding them to fixed lengths)
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")
    
    # The labels are the tokenized summaries
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the training and validation datasets
train_dataset = dataset["train"].map(preprocess_function, batched=True)
val_dataset = dataset["validation"].map(preprocess_function, batched=True)

# Step 4: Load the pre-trained BART model for summarization
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Step 5: Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Where to save the model and logs
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=5e-5,  # Learning rate
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    num_train_epochs=3,  # Number of epochs
    weight_decay=0.01,  # Weight decay
    logging_dir="./logs",  # Where to save logs
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,  # Save only the last 2 models
    predict_with_generate=True,  # Use model’s generate method for predictions
)

# Step 6: Set up the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Step 7: Train the model
trainer.train()

# Step 8: Evaluate the model on the validation set
trainer.evaluate()

# Step 9: Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_bart_cnn")
tokenizer.save_pretrained("./fine_tuned_bart_cnn")

# Step 10: Summarize new text with the trained model
article = """
Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
At its core, AI is defined as the simulation of human intelligence in machines that are programmed to think, learn, and problem-solve autonomously. Early AI systems were limited to narrow tasks, such as playing chess or performing basic data analysis. However, modern AI, powered by advancements in machine learning (ML) and deep learning (DL), is much more sophisticated, capable of recognizing patterns, making predictions, and even generating human-like text, images, and sounds.
AI in Healthcare
One of the most promising areas of AI application is in healthcare. AI algorithms are already being used to diagnose diseases more accurately and faster than human doctors. Machine learning models are trained on vast amounts of medical data—ranging from patient histories to diagnostic images—and can spot patterns that might be invisible to the human eye. For example, AI systems have been shown to outperform radiologists in detecting tumors in medical imaging, such as X-rays and MRIs.
Moreover, AI-driven personalized medicine is becoming a reality. By analyzing genetic information, lifestyle data, and medical history, AI can help doctors tailor treatments to the individual needs of patients, potentially improving outcomes and reducing side effects. In the future, AI could also assist in drug discovery, identifying potential therapies for diseases like cancer, Alzheimer's, and rare genetic disorders faster and more efficiently than traditional methods.
"""

# Tokenize the article
inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)

# Generate a summary
summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=50, max_length=150, early_stopping=True)

# Decode the generated summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("Generated Summary:", summary)
