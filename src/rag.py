# src/rag.py
from embeddings import Embedder
from db_qdrant import QdrantDB
from db_local import InMemoryDB
from config import QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY
from typing import List
import numpy as np
import os

# Groq client
from groq import Groq
from transformers import pipeline

class RAG:
    def __init__(self, use_qdrant: bool = True, qdrant_collection: str = "quickrag_collection"):
        # embedder on CPU (safe). You can change device="cuda" if you want and have VRAM.
        self.embedder = Embedder(device="cpu")
        self.use_qdrant = use_qdrant
        if use_qdrant:
            if not QDRANT_URL or not QDRANT_API_KEY:
                raise ValueError("Qdrant keys not found. Put QDRANT_URL and QDRANT_API_KEY in .env")
            self.db = QdrantDB(url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=qdrant_collection)
        else:
            self.db = InMemoryDB()

        # Groq client (if available)
        self.groq_client = None
        if GROQ_API_KEY:
            self.groq_client = Groq(api_key=GROQ_API_KEY)

        # Local fallback generator (small & CPU-friendly)
        self.local_generator = pipeline("text-generation", model="distilgpt2", device=-1)

    def index(self, docs: List[str], batch_size: int = 64):
        embeddings = self.embedder.embed(docs, batch_size=batch_size)
        self.db.upsert(docs, embeddings)

    def retrieve(self, query: str, top_k: int = 5):
        q_vec = self.embedder.embed([query])[0]
        return self.db.search(q_vec, top_k=top_k)

    def generate_with_groq(self, prompt: str, model_name: str = "mixtral-8x7b-instruct"):
        if not self.groq_client:
            raise RuntimeError("Groq API key not found; cannot call Groq.")
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]

    def generate_local(self, prompt: str, max_new_tokens: int = 150):
        out = self.local_generator(prompt, max_new_tokens=max_new_tokens)
        return out[0]["generated_text"]

    def answer(self, query: str, top_k: int = 5, use_groq_if_available: bool = True):
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(retrieved)
        prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer concisely and cite which retrieved docs you used."

        if use_groq_if_available and self.groq_client:
            return self.generate_with_groq(prompt)
        else:
            return self.generate_local(prompt)
