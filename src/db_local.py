# src/db_local.py
import numpy as np
from typing import List

class InMemoryDB:
    def __init__(self):
        self.vectors = None  # numpy array (N, D)
        self.docs = []

    def upsert(self, docs: List[str], embeddings: np.ndarray):
        print(f"[InMemoryDB] Upserting {len(docs)} docs.")
        if self.vectors is None:
            self.vectors = embeddings.copy()
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        self.docs.extend(docs)
        print(f"[InMemoryDB] Total docs: {len(self.docs)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        print(f"[InMemoryDB] Searching for top {top_k} docs.")
        # compute cosine similarity
        q = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        vecs = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        sims = np.dot(vecs, q)
        top_idx = np.argsort(sims)[-top_k:][::-1]
        print(f"[InMemoryDB] Found {len(top_idx)} docs.")
        return [self.docs[i] for i in top_idx]
