# src/rag.py
"""
RAG (Retrieval-Augmented Generation) pipeline using Qdrant (Vector DB) + Groq (LLM).
- Embeds and indexes documents
- Retrieves top-k relevant docs
- Generates concise answers with citations
"""

from embeddings import Embedder
from db_qdrant import QdrantDB
from config import QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY
from typing import List
import numpy as np

# Groq client
from groq import Groq


class RAG:
    """Retrieval-Augmented Generation system with Qdrant + Groq."""

    def __init__(self, qdrant_collection: str = "quickrag_collection"):
        # Initialize embedder (CPU by default)
        self.embedder = Embedder(device="cpu")

        # Vector DB (Qdrant Cloud)
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("Qdrant keys not found. Please set QDRANT_URL and QDRANT_API_KEY in .env")
        self.db = QdrantDB(url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=qdrant_collection)

        # Groq client (for LLM generation)
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env")
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    # ------------------------------
    # Indexing
    # ------------------------------
    def index(self, docs: List[str], batch_size: int = 64):
        """Embed and store documents in Qdrant."""
        print(f"[RAG] Indexing {len(docs)} docs with batch_size={batch_size}")
        embeddings = self.embedder.embed(docs, batch_size=batch_size)
        self.db.upsert(docs, embeddings)
        print("[RAG] Upsert complete.")

    # ------------------------------
    # Retrieval
    # ------------------------------
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve top-k most relevant docs from Qdrant for a given query."""
        print(f"[RAG] Retrieving top {top_k} docs for query: {query}")
        q_vec = self.embedder.embed([query])[0]
        results = self.db.search(q_vec, top_k=top_k)
        print(f"[RAG] Retrieved {len(results)} docs.")
        return results

    # ------------------------------
    # Generation with Groq
    # ------------------------------
    def generate_with_groq(self, prompt: str, model_name: str = "llama-3.1-8b-instant") -> str:
        """Generate an answer from Groq LLM using the given prompt + context."""
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------
    # Full RAG Answer
    # ------------------------------
    def answer(self, query: str, top_k: int = 5, model_name: str = "llama-3.1-8b-instant") -> str:
        """Retrieve context and generate an answer to the query."""
        print(f"[RAG] Answering query: {query}")
        retrieved = self.retrieve(query, top_k=top_k)

        for i, doc in enumerate(retrieved):
            print(f"  Doc {i+1}: {doc[:100]} ...")

        context = "\n\n".join(retrieved)
        prompt = (
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            f"Answer concisely and cite which retrieved docs you used."
        )

        print("[RAG] Using Groq for generation.")
        return self.generate_with_groq(prompt, model_name=model_name)
