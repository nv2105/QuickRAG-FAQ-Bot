# src/load_data.py
import pandas as pd
from rag import RAG
import argparse

def chunk_text(text, max_chars=1000):
    """
    Split long text into smaller chunks for better retrieval granularity.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

def main(csv_path="data/rag_optimized_5000.csv", max_docs=None):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Determine which column to use as text
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
        raise ValueError(
            "No suitable text column found in CSV. Required: 'text', 'content', 'title+content', or 'question+answer'."
        )

    # Convert to list
    docs = df[text_col].astype(str).tolist()
    if max_docs:
        docs = docs[:max_docs]

    # Chunk long docs
    chunked_docs = []
    for d in docs:
        if len(d) > 1200:
            chunked_docs.extend(chunk_text(d, max_chars=1000))
        else:
            chunked_docs.append(d)

    print(f"[LOG] Indexing {len(chunked_docs)} document chunks...")

    # Initialize RAG and index docs
    rag = RAG(qdrant_collection="quickrag_collection")
    rag.index(chunked_docs, batch_size=64)

    print("[LOG] Indexing complete. All docs should now be in Qdrant Cloud.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/rag_optimized_5000.csv", help="Path to CSV file")
    parser.add_argument("--max", type=int, default=None, help="Limit number of docs for quick testing")
    args = parser.parse_args()

    main(csv_path=args.csv, max_docs=args.max)
