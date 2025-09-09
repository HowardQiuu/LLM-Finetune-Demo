import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline

# 1 prepare data
def chunk_text(text, chunk_size=200, overlap=50):
    """
    chunk long text into pieces of short texts
    - chunk_size: The maximum number of characters in each segment
    - overlap: The number of overlapping characters between adjacent segments
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

raw_documents = [
    "The Transformer architecture was introduced in the paper Attention is All You Need. "
    "It relies on self-attention mechanisms to model long-range dependencies in text, "
    "replacing traditional recurrent and convolutional networks.",

    "Retrieval-Augmented Generation (RAG) is a method that combines dense retrieval with language models. "
    "In RAG, relevant documents are retrieved from a knowledge base and used as additional context for text generation, "
    "allowing models to access up-to-date and domain-specific information.",

    "FAISS is a library developed by Facebook AI Research for efficient similarity search. "
    "It supports large-scale nearest neighbor search in high-dimensional spaces and is widely used for building vector databases. "
    "FAISS can run on both CPUs and GPUs.",

    "LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning method for large language models. "
    "Instead of updating all model weights, LoRA injects trainable rank-decomposition matrices into attention layers, "
    "drastically reducing the number of trainable parameters."
]

documents = []
for doc in raw_documents:
    documents.extend(chunk_text(doc, chunk_size=120, overlap=30))

print(f"📄 Total chunks: {len(documents)}")


# 2. text embedding
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def embedding(texts):
    inputs = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = embedding_model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1) # average pooling
    return embeddings.numpy()

doc_embeddings = embedding(documents)

# 3. build vector store
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# 4. load LLM model
gen_model_name = "distilgpt2"
generator = pipeline("text-generation", model=gen_model_name)

# 5. RAG query function
def rag_query(query, top_k=2):
    # (1) query -> embedding
    query_vec = embedding([query])
    # (2) similarity search
    scores, indices = index.search(query_vec, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    # (3) construct prompt
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question using the following context:\n{context}\nQuestion: {query}\nAnswer:"
    # (4) generate answer
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return response

# 6. test RAG
if __name__ == "__main__":
    query = "What does RAG stand for?"
    answer = rag_query(query)
    print("User Query:\n", query)
    print("Assistant's Answer:\n", answer)