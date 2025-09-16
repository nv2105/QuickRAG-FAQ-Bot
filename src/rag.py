# src/rag.py
from embeddings import Embedder
from db_qdrant import QdrantDB
from config import QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY
from typing import List
import numpy as np

# Groq client
from groq import Groq

class RAG:
    def __init__(self, qdrant_collection: str = "quickrag_collection"):
        # Embedder on CPU
        self.embedder = Embedder(device="cpu")
        
        # Qdrant DB (mandatory)
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("Qdrant keys not found. Put QDRANT_URL and QDRANT_API_KEY in .env")
        self.db = QdrantDB(url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=qdrant_collection)
        
        # Groq client (mandatory)
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found; cannot call Groq.")
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    def index(self, docs: List[str], batch_size: int = 64):
        print(f"[RAG] Indexing {len(docs)} docs with batch_size={batch_size}")
        embeddings = self.embedder.embed(docs, batch_size=batch_size)
        self.db.upsert(docs, embeddings)
        print("[RAG] Upsert complete.")

    def retrieve(self, query: str, top_k: int = 5):
        print(f"[RAG] Retrieving top {top_k} docs for query: {query}")
        q_vec = self.embedder.embed([query])[0]
        results = self.db.search(q_vec, top_k=top_k)
        print(f"[RAG] Retrieved {len(results)} docs.")
        return results

    def generate_with_groq(self, prompt: str, model_name: str = "llama-3.1-8b-instant"):
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Access content via attribute
        return response.choices[0].message.content

    def answer(self, query: str, top_k: int = 5, model_name: str = "llama-3.1-8b-instant"):
        print(f"[RAG] Answering query: {query}")
        retrieved = self.retrieve(query, top_k=top_k)
        for i, doc in enumerate(retrieved):
            print(f"  Doc {i+1}: {doc[:100]} ...")
        context = "\n\n".join(retrieved)
        prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer concisely and cite which retrieved docs you used."
        print("[RAG] Using Groq for generation.")
        return self.generate_with_groq(prompt, model_name=model_name)
