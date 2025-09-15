# src/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from tqdm import tqdm

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        # device="cpu" recommended for small GPU; you can set device="cuda" if available
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Return numpy array of embeddings. Batches to avoid OOM."""
        print(f"[Embedder] Embedding {len(texts)} texts with batch_size={batch_size}")
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            print(f"[Embedder] Batch {i//batch_size+1}: {len(batch)} texts")
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(emb)
        result = np.vstack(embeddings)
        print(f"[Embedder] Final embeddings shape: {result.shape}")
        return result
