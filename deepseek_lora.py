import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Reduced rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files="disaster_data.jsonl")
print(f"Dataset size: {len(dataset['train'])} examples")

# Tokenize with output-only labels
def tokenize_function(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    input_encodings = tokenizer(
        instructions,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    output_encodings = tokenizer(
        outputs,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    input_ids = input_encodings["input_ids"]
    attention_mask = input_encodings["attention_mask"]
    labels = output_encodings["input_ids"]
    labels[output_encodings["attention_mask"] == 0] = -100
    print(f"Sample input: {instructions[0]}")
    print(f"Sample output: {outputs[0]}")
    print(f"Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")
    return {
        "input_ids": input_ids.squeeze(),
        "attention_mask": attention_mask.squeeze(),
        "labels": labels.squeeze()
    }

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "output"]
)

# Training setup with adjusted hyperparameters
training_args = TrainingArguments(
    output_dir="./lora_qwen_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Accumulate to stabilize gradients
    num_train_epochs=20,            # More epochs for small data
    learning_rate=2e-5,             # Back to moderate LR
    fp16=True,
    logging_steps=1,
    save_steps=5,
    save_total_limit=2,
    remove_unused_columns=False,
    optim="adamw_torch",
    warmup_steps=5                 # Longer warmup
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

total_samples = len(tokenized_dataset["train"])
batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
total_steps = (total_samples // batch_size) * training_args.num_train_epochs
print(f"Total training steps: {total_steps}")  # Now 30 with 6 samples, 5 epochs

# Fine-tune
trainer.train()

# Save
model.save_pretrained("./deepseek_lora_output")
tokenizer.save_pretrained("./deepseek_lora_output")