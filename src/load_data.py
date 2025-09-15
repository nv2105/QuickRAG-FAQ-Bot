# src/load_data.py
import pandas as pd
from rag import RAG
import argparse
import math

def chunk_text(text, max_chars=1000):
    # naive chunker by characters (you can improve by sentence)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

def main(csv_path="data/rag_optimized_5000.csv", use_qdrant=True, max_docs=None):
    df = pd.read_csv(csv_path)
    # assume text column name can be 'text' or 'content' or 'title+content'
    if "text" in df.columns:
        text_col = "text"
    elif "content" in df.columns:
        text_col = "content"
    else:
        # fallback: try to build from title+content
        if "title" in df.columns and "content" in df.columns:
            df["text"] = df["title"].fillna("") + "\n\n" + df["content"].fillna("")
            text_col = "text"
        else:
            raise ValueError("No suitable text column found in CSV (need 'text' or 'content' or both 'title' and 'content').")

    docs = df[text_col].astype(str).tolist()
    if max_docs:
        docs = docs[:max_docs]

    # Optionally chunk long docs into smaller chunks to improve retrieval granularity
    chunked_docs = []
    for d in docs:
        if len(d) > 1200:
            chunked_docs.extend(chunk_text(d, max_chars=1000))
        else:
            chunked_docs.append(d)

    print(f"Indexing {len(chunked_docs)} document chunks...")

    rag = RAG(use_qdrant=use_qdrant)
    rag.index(chunked_docs, batch_size=64)
    print("Indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/rag_optimized_5000.csv")
    parser.add_argument("--local", action="store_true", help="Use local in-memory DB instead of Qdrant Cloud")
    parser.add_argument("--max", type=int, default=None, help="Limit number of docs for quick testing")
    args = parser.parse_args()
    main(csv_path=args.csv, use_qdrant=not args.local, max_docs=args.max)
