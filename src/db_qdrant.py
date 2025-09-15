# src/db_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List
import numpy as np

class QdrantDB:
    def __init__(self, url: str, api_key: str, collection_name: str = "quickrag_collection", vector_size: int = 384):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = vector_size

        # create or recreate collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
        )

    def upsert(self, docs: List[str], embeddings: np.ndarray, batch_size: int = 64):
        """Upsert docs and embeddings to Qdrant in batches."""
        print(f"[QdrantDB] Upserting {len(docs)} docs in batches of {batch_size}")
        points = []
        for idx, (doc, emb) in enumerate(zip(docs, embeddings)):
            points.append(models.PointStruct(id=idx, vector=emb.tolist(), payload={"text": doc}))
            if len(points) >= batch_size:
                print(f"[QdrantDB] Upserting batch ending at doc {idx+1}")
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []
        if points:
            print(f"[QdrantDB] Upserting final batch of {len(points)} docs")
            self.client.upsert(collection_name=self.collection_name, points=points)
        print("[QdrantDB] Upsert complete.")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        print(f"[QdrantDB] Searching for top {top_k} docs.")
        hits = self.client.search(collection_name=self.collection_name, query_vector=query_vector.tolist(), limit=top_k)
        print(f"[QdrantDB] Found {len(hits)} hits.")
        return [r.payload["text"] for r in hits]
