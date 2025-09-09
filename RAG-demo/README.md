# 🧩 01 Basic RAG Demo
This project is a **minimal Retrieval-Augmented Generation (RAG) demo**, built with:

- **[HuggingFace Transformers](https://huggingface.co/transformers/)**  
- **[sentence-transformers](https://www.sbert.net/)** for text embeddings  
- **[FAISS](https://github.com/facebookresearch/faiss)** for similarity search    
- **GPT-2 (distilgpt2)** as a lightweight language model   
  
---

## 📖 What is RAG?

**RAG (Retrieval-Augmented Generation)** combines **information retrieval** with **text generation**.  

- **Before querying**:  
  - Split documents into smaller chunks  
  - Encode chunks into embeddings  
  - Store embeddings in a vector database (FAISS)  

- **At query time**:  
  1. Encode the user query into an embedding  
  2. Retrieve the top-k most relevant chunks from FAISS  
  3. Build a prompt by combining retrieved context with the query  
  4. Pass the prompt to a language model for answer generation  

This approach allows the model to generate answers grounded in external knowledge.

---

## 🚀 Quick start

### 1. Install dependencies
```bash
pip install faiss-cpu torch transformers sentence-transformers
```

### 2. Run Demo
```bash
python 01_basic_rag.py
```

### 3. Example Output
```text
📄 Total chunks: 12
User Query:
 What does RAG stand for?
Assistant's Answer:
 Answer the question using the following context:
 Retrieval-Augmented Generation (RAG) is a method ...
 Question: What does RAG stand for?
 Answer: RAG stands for Retrieval-Augmented Generation.
```

# 🏗️ Code Workflow

- Chunking: split long documents into smaller passages
- Embedding: encode chunks with all-MiniLM-L6-v2 into dense vectors
- Vector search: retrieve relevant chunks using FAISS similarity search
- Generation: use distilgpt2 to produce a final answer based on context

# 📂 Future Extensions
This demo is a minimal RAG pipeline. Possible improvements include:
- Replace the embedding model with stronger ones (e.g. m3e-base,text-embedding-ada-002)
- Swap the generator with larger LLMs (LLaMA, ChatGLM, Qwen)
- Add re-ranking for better retrieval quality
- Integrate with frameworks like LangChain or LlamaIndex
  
# ✨ Acknowledgments

HuggingFace

FAISS

SentenceTransformers