# RAG Demo Series

This repository contains a step-by-step implementation of **Retrieval-Augmented Generation (RAG)** using Python, FAISS, and Hugging Face models.  
Each script builds upon the previous one to introduce additional functionality:

- **01_basic_rag.py** – A minimal RAG pipeline with dense retrieval and generation.  
- **02_rag_with_rerank.py** – Adds a cross-encoder reranker to improve retrieval quality.  
- **03_multi_doc_rag.py** – Supports multiple document collections, allowing cross-source retrieval.  

---

## 📂 Files Overview

### 1. `01_basic_rag.py`
- Splits documents into chunks.
- Uses a sentence transformer (`all-MiniLM-L6-v2`) for embeddings.
- Stores embeddings in a FAISS index.
- Retrieves top-k chunks and feeds them to a lightweight generator (`distilgpt2`).
- Demonstrates the core RAG pipeline.

**Key Feature:** First working RAG prototype (retrieval + generation).  

---

### 2. `02_rag_with_rerank.py`
- Inherits the basic pipeline from `01_basic_rag.py`.
- After FAISS retrieval, adds a **CrossEncoder reranker** (`ms-marco-MiniLM-L-6-v2`).
- Reranker scores query–document pairs to select the most relevant chunks.
- Generator answers questions based on the reranked context.

**Key Feature:** Higher-quality retrieval results through reranking.  

---

### 3. `03_multi_doc_rag.py`
- Extends RAG to handle **multiple document sets** (e.g., Transformers, RAG, Tools).
- Builds a separate FAISS index for each set.
- Retrieves from all indexes, optionally reranks, and feeds the top results to the generator.
- Can run with or without reranking.

**Key Feature:** Multi-source retrieval + optional reranking.  

---

## ⚙️ Installation

```bash
# Create a new environment (optional but recommended)
conda create -n rag-demo python=3.10 -y
conda activate rag-demo

# Install dependencies
pip install torch transformers sentence-transformers faiss-cpu
```

🚀 Usage

Run any script directly:
```bash
# Basic RAG
python 01_basic_rag.py

# RAG with reranking
python 02_rag_with_rerank.py

# Multi-document RAG
python 03_multi_doc_rag.py
```

Each script will:

- Build document chunks.
- Compute embeddings and build FAISS indexes.
- Perform retrieval (and reranking if enabled).
- Generate an answer with the retrieved context.

📊 Comparison of Features

| Script                   | Retrieval | Reranking    | Multi-doc Support |
| ------------------------ | --------- | ------------ | ----------------- |
| 01\_basic\_rag.py        | ✅         | ❌            | ❌                 |
| 02\_rag\_with\_rerank.py | ✅         | ✅            | ❌                 |
| 03\_multi\_doc\_rag.py   | ✅         | ✅ (optional) | ✅                 |

🧠 Notes

- The generator model (distilgpt2) is small and easy to run, but answers may be repetitive or limited.
- You can replace it with a stronger model (e.g., gpt2, tiiuae/falcon-7b-instruct, or Qwen models) if you have more resources.
- FAISS supports both CPU and GPU backends. To use GPU, install faiss-gpu instead of faiss-cpu.

📌 Next Steps

- Experiment with larger embedding and generator models.
- Connect to external document sources (e.g., PDFs, web pages).
- Add caching and streaming for more efficient pipelines.