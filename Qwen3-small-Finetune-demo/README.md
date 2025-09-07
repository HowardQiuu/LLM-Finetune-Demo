# Qwen3-1.7B LoRA 微调 Demo

本项目展示了如何使用 [PEFT (LoRA)](https://huggingface.co/docs/peft/index)  
对 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 模型进行 **低资源微调**。

通过结合 **4-bit 量化 (bitsandbytes)** 与 **LoRA 参数高效微调**，  
我们可以在仅 15GB 左右显存的环境下运行 Qwen3-1.7B 的微调。

---

## ✨ 功能亮点
- ✅ 使用 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 进行 **4bit 量化**，减少显存占用  
- ✅ 使用 [LoRA](https://arxiv.org/abs/2106.09685) 进行 **高效微调**  
- ✅ 在 Colab 免费 GPU 上可运行  
- ✅ 提供完整训练 + 推理代码示例  

---

## 📦 安装依赖

```bash
pip install -U bitsandbytes transformers accelerate peft datasets
```

🚀 训练

运行以下脚本会：

- 加载 Qwen3-1.7B（4bit 量化）

- 使用 LoRA 插入可训练参数

- 在 IMDB 数据集上进行 情感分类文本建模 微调
```bash
python Qwen3-1.7B-Finetune-demo.py
```

主要训练参数：
- batch_size=1（避免 OOM）

- gradient_accumulation_steps=4（等效 batch=4）

- learning_rate=2e-4

- num_train_epochs=1

训练完成后，LoRA 适配器会保存在 ./qwen3_lora_demo/。

💬 推理测试

训练完成后可以进行推理

```python
query = "I really enjoyed this movie because"
inputs = tokenizer(query, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
示例输出（因随机性而不同）：
```kotlin
🤖 Model response:
I really enjoyed this movie because the acting was great and the story was touching...
```
