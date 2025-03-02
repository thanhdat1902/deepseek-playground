import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the base model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the disaster dataset
import json

def load_dataset(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

dataset = load_dataset("disaster_data.jsonl")

# Create a few-shot prompt
def create_few_shot_prompt(dataset, new_instruction):
    prompt = "You are a disaster response assistant. Analyze the following situations and provide appropriate responses.\n\n"
    for example in dataset:  # Use only the first 2 examples
        prompt += f"Instruction: {example['instruction']}\nOutput: {example['output']}\n\n"
    prompt += "---\n\n"  # Add a separator
    prompt += f"Instruction: {new_instruction}\nOutput:"
    return prompt

# Generate a response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_beams=2,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only the generated response
def extract_generated_response(full_output, prompt):
    # Remove the prompt from the full output
    generated_text = full_output[len(prompt):].strip()
    # Stop at the first newline or double newline
    stop_index = generated_text.find("\n\n")
    if stop_index != -1:
        generated_text = generated_text[:stop_index]
    return generated_text

# Example usage
new_instruction = "Predict flood risk: 1in rain forecast for Los Angeles"
few_shot_prompt = create_few_shot_prompt(dataset, new_instruction)

response = generate_response(few_shot_prompt)
generated_response = extract_generated_response(response, few_shot_prompt)
print("\nGenerated response:\n", generated_response)