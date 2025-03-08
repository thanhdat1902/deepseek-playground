from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1. Load the base model and tokenizer
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 2. Load the LoRA adapter
lora_model_path = "./finetuned_model_lora"
model = PeftModel.from_pretrained(model, lora_model_path)
model = model.to("cuda")  # Ensure model is on GPU

# 3. Define the system prompt (same as training)
system_prompt = "You are a helpful assistant trained to provide detailed, structured disaster analysis reports without internal reasoning. Use the following structure: <disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>."

# 4. Define a new instruction for testing
new_instruction = (
    "event_title: M 4.8 - 15 km SE of Lima, Peru\n"
    "event_description: A minor earthquake occurred near Lima, Peru, with slight shaking reported.\n"
    "disaster_type: Earthquake\n"
    "event_date_time: 2025-03-20 09:15:30 UTC\n"
    "event_location: 15 km SE of Lima, Peru\n"
    "event_coordinates: [-76.9876, -12.1234, 10.5]\n"
    "disaster_details: magnitude: 4.8, alert_level: Green, tsunami_risk: 0, significance_level: 280, felt_reports: 800, damage_reports: No significant damage.\n"
    "climate_data: temperature: 25.0°C, windspeed: 10.5 km/h, winddirection: 180°, humidity: 65%, precipitation_probability: 10%, cloud_cover: 30%, pressure_sea_level: 1015.0 hPa\n"
    "Provide a detailed analysis with the following structure: <disaster_analysis>, <event_summary>, <detailed_analysis>, <predictions>, <impacts>, <mitigation_strategies>. Start with <disaster_analysis>."
)

# 5. Prepare the input with a structure cue
input_text = f"<|user|>{system_prompt} {new_instruction}<|assistant|><disaster_analysis>"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 6. Generate the output
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.5,  # Reduced for more deterministic output
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# 7. Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\nInference Output:")
print(generated_text.split("<|assistant|>")[1])