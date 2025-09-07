from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "THUDM/chatglm3-6b"

    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",         # auto choose GPU/CPU
        torch_dtype="auto"         # auto choose data type (fp16/bf16)
    ).eval()

    print("✅ Model loaded. Ready for inference!")

    # Example inference
    text = "Hello, please introduce yourself in one sentence."
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)

    print("🤖 ChatGLM3 output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
