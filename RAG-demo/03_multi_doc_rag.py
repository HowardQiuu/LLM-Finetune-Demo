import faiss
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import CrossEncoder

# 1. Text chunking function
def chunk_text(text, chunk_size=200, overlap=50):
    """
    Split long documents into smaller overlapping chunks.
    - chunk_size: maximum length of each chunk
    - overlap: number of overlapping characters between chunks
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 2. Example document sets (multiple sources)
doc_sets = {
    "Transformers": [
        "The Transformer architecture was introduced in the paper Attention is All You Need. "
        "It relies on self-attention mechanisms to model long-range dependencies in text."
    ],
    "RAG": [
        "Retrieval-Augmented Generation (RAG) combines dense retrieval with language models. "
        "Relevant documents are retrieved from a knowledge base and used as additional context."
    ],
    "Tools": [
        "FAISS is a library for efficient similarity search in high-dimensional spaces.",
        "LoRA is a parameter-efficient fine-tuning method for large language models."
    ]
}

# 3. Load embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def embedding(texts):
    """Convert a list of texts into embeddings (using mean pooling)."""
    inputs = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = embedding_model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1)  # mean pooling
    return embeddings.numpy()

# 4. Build FAISS index (one per document set)
indexes = {}
doc_chunks = {}
for name, docs in doc_sets.items():
    chunks = []
    for d in docs:
        chunks.extend(chunk_text(d, chunk_size=120, overlap=30))
    doc_chunks[name] = chunks
    
    embeddings = embedding(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    indexes[name] = index

print(f"✅ Built indexes for {len(indexes)} document sets")

# 5. load generator LLM and reranker model
gen_model_name = "distilgpt2"
generator = pipeline("text-generation", model=gen_model_name)

reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(reranker_model_name)

# 6. RAG query function
def rag_query(query, top_k=3, use_rerank=True, rerank_k=2):
    """
    Execute a RAG query:
    1. Retrieve top_k chunks from each document set using FAISS
    2. (Optional) Re-rank retrieved chunks with a cross-encoder, keep top_m
    3. Concatenate context + query and feed into the generator
    """
    # (1) Multi-source retrieval
    retrieved = []
    query_vec = embedding([query])
    for name, index in indexes.items():
        scores, ids = index.search(query_vec, top_k)
        retrieved.extend([doc_chunks[name][i] for i in ids[0]])
        
    # (2) reranking with cross-encoder
    if use_rerank:
        pairs = [(query, d) for d in retrieved]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
        retrieved = [d for d, s in ranked[:rerank_k]]
    
    context = "\n".join(retrieved)
    prompt = f"Answer the question using the following context:\n{context}\nQuestion: {query}\nAnswer:"
    
    response = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
    return response

# 7. test RAG
if __name__ == "__main__":
    q = "What is FAISS and what is it used for?"
    print("\nUser Query:", q)

    ans1 = rag_query(q, use_rerank=False)
    print("\n🔹 Without rerank:\n", ans1)

    ans2 = rag_query(q, use_rerank=True)
    print("\n🔹 With rerank:\n", ans2)
