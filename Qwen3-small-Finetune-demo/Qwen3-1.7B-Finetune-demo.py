import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen3-1.7B"

# quantization config (bnb 4bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",        
    bnb_4bit_compute_dtype=torch.float16,
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# load 4bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,   
    device_map="auto",
)

# LoRA arguments
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Fine tune attention projection layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# example dataset (IMDB)
dataset = load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(200))  # 200 for training
test_data = dataset["test"].shuffle(seed=42).select(range(50))     # 50 for validation

def tokenize(batch):
    tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# trainning arguments
training_args = TrainingArguments(
    output_dir="./qwen3_lora_demo",
    per_device_train_batch_size=1,   # batch=1 to avoid OOM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,   # equivalent batch=4
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=1,
    fp16=True,                       # use half precision
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer
)

# start training
trainer.train()

# inference test
query = "I really enjoyed this movie because"
inputs = tokenizer(query, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=150)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🤖 Model response:")
print(response)

# Step Training Loss Validation Loss
# 50   3.876800       3.610077

# 🤖 Model response:I really enjoyed this movie because I have a lot of respect for the director and the crew, and I think this movie is one of the best examples of a great director, a great cast, and a great script.
# It's not just a good movie, it's a movie that you can watch over and over again, and it's not the kind of movie that you can just watch once. 
# The acting was great, the direction was great, the cinematography was great, and the music was also great. It's a movie that's a bit different from what you'd expect, but that's a good thing. 
# I was really impressed with the way the movie was made. 
# It was a very well thought out movie, and I think the