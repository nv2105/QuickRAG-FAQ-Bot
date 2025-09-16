# src/load_data.py
"""
Data loader and indexer for RAG.
- Reads FAQ/knowledge data from CSV or HuggingFace dataset
- Cleans and chunks text
- Indexes into Qdrant using RAG class
"""

import pandas as pd
from rag import RAG
import argparse


def chunk_text(text: str, max_chars=1000):
    """Naive character-based chunking (can be improved with sentence split)."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def main(csv_path="data/rag_optimized_5000.csv", max_docs=None):
    """Load CSV, preprocess text, chunk, and index into Qdrant Cloud."""
    df = pd.read_csv(csv_path)

    # Auto-detect text column
    if "text" in df.columns:
        text_col = "text"
    elif "content" in df.columns:
        text_col = "content"
    elif "title" in df.columns and "content" in df.columns:
        df["text"] = df["title"].fillna("") + "\n\n" + df["content"].fillna("")
        text_col = "text"
    elif "question" in df.columns and "answer" in df.columns:
        df["text"] = df["question"].fillna("") + "\n\n" + df["answer"].fillna("")
        text_col = "text"
    else:
        raise ValueError("No suitable text column found (need 'text', 'content', 'title+content', or 'question+answer').")

    docs = df[text_col].astype(str).tolist()
    if max_docs:
        docs = docs[:max_docs]

    # Chunk large documents
    chunked_docs = []
    for d in docs:
        if len(d) > 1200:
            chunked_docs.extend(chunk_text(d, max_chars=1000))
        else:
            chunked_docs.append(d)

    print(f"[LOG] Indexing {len(chunked_docs)} document chunks...")

    rag = RAG()
    rag.index(chunked_docs, batch_size=64)
    print("[LOG] Indexing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/rag_optimized_5000.csv")
    parser.add_argument("--max", type=int, default=None, help="Limit number of docs for quick testing")
    args = parser.parse_args()

    main(csv_path=args.csv, max_docs=args.max)
