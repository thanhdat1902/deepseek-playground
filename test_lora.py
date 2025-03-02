from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
model_path = "./deepseek_lora_output"  # Updated path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test prompt
prompt = "Analyze seismic data: 6.0 magnitude near Los Angeles"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))