import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch

# 1. Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 2. Prepare the dataset
system_prompt = "You are a helpful assistant trained to provide detailed, structured disaster analysis reports without internal reasoning. Use the following structure: <disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>."
dataset_path = "disaster_data.jsonl"

# Read and format the JSONL data with error handling
formatted_data = []
with open(dataset_path, "r") as f:
    for i, line in enumerate(f, 1):
        try:
            sample = json.loads(line.strip())
            full_text = f"<|user|>{system_prompt} {sample['instruction']}<|assistant|>{sample['output']}"
            formatted_data.append({"text": full_text})
        except json.JSONDecodeError as e:
            print(f"Error in line {i}: {line.strip()}")
            print(f"Error message: {e}")
            raise

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(formatted_data)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Set up LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    learning_rate=2e-4,
    warmup_steps=5,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    gradient_accumulation_steps=4,
)

# 5. Set up data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Initialize and run the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
print("Training completed.")

# 7. Save the LoRA adapter and tokenizer
model.save_pretrained("./finetuned_model_lora")  # Save LoRA adapter separately
tokenizer.save_pretrained("./finetuned_model_lora")
print("LoRA adapter saved to './finetuned_model_lora'")

# Optionally, merge and save the model (for reference)
model = model.merge_and_unload()
model.save_pretrained("./finetuned_model_merged")
tokenizer.save_pretrained("./finetuned_model_merged")
print("Merged model saved to './finetuned_model_merged'")