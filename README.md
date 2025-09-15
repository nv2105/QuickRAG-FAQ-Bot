# QuickRAG-FAQ-Bot

A Retrieval-Augmented Generation (RAG) pipeline using **SentenceTransformers**, **Gorq/Qdrant**, and **Transformers**.

## Features
- Embeds documents using MiniLM
- Stores vectors in Gorq or local Qdrant
- Retrieves top-k documents
- Generates answers with DistilGPT-2 (lightweight)

## Setup
```bash
git clone <repo-url>
cd QuickRAG-FAQ-Bot
pip install -r requirements.txt
