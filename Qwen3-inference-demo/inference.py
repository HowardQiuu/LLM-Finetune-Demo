import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    offload_folder="./offload",  # specify a folder to offload weights if needed  
    offload_state_dict=True # offload the model weights to CPU when not in use
).eval()

print("✅ Model loaded. Ready for inference!")

query = "Hello, please introduce yourself in one sentence."
inputs = tokenizer(query, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🤖 Model response:")
print(response)
