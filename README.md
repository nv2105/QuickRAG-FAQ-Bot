# 🤖 QuickRAG - FAQ Bot

An intelligent **Retrieval-Augmented Generation (RAG) chatbot** for answering FAQs using **domain-specific datasets**. The bot indexes FAQs into **Qdrant Cloud (vector DB)** and retrieves relevant context to generate accurate answers with **Groq LLM**.  

Built with modular Python code, this project demonstrates end-to-end **RAG pipelines** — from embeddings to vector search to response generation.

---

## 🏗️ Architecture

![Architecture](assets/architecture.png)

---

## 📊 Key Features

- 🔎 **Semantic Search**: Retrieves relevant FAQ context from Qdrant Cloud  
- 📂 **Domain-Specific Indexing**: Import datasets (CSV/HuggingFace) and index easily  
- ⚡ **Groq-powered LLM**: Fast and concise answer generation  
- 🛠️ **Local & Cloud Support**: Choose between in-memory DB (quick test) or Qdrant Cloud  
- 🖥️ **Notebook Demo**: End-to-end workflow in Jupyter Notebook  
- 📸 **Screenshots included** for quick visualization  

---

## 🧰 Tech Stack

- **Python** 🐍  
- **SentenceTransformers** (MiniLM embeddings)  
- **Qdrant Cloud** (vector DB)  
- **Groq API** (answer generation)  
- **Pandas**, **NumPy**  
- **Jupyter Notebook** for demo  

---

## 📷 Screenshots

| Indexing Data | Query Answering | Multi-Query Demo |
|---------------|-----------------|------------------|
| ![Indexing](assets/indexing.png) | ![Single Query](assets/single_query.png) | ![Multi Query](assets/multi_query.png) |

> All screenshots are stored in the `assets/` folder.

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/nv2105/QuickRAG-FAQ-Bot.git
cd QuickRAG-FAQ-Bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API Keys

Create a .env file in the project root:
```bash
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key   # optional but recommended
```

### 4. Run the notebook demo
``` bash
jupyter notebook QuickRAG_Demo.ipynb
```

### 📁 Project Structure

```bash

QuickRAG-FAQ-Bot/
│── assets/                 # Screenshots & diagrams
│── data/                   # FAQ CSV datasets
│── src/
│   ├── rag.py              # Main RAG pipeline
│   ├── embeddings.py       # Embedding generator
│   ├── db_qdrant.py        # Qdrant Cloud integration
│   ├── db_local.py         # In-memory DB (testing)
│   ├── load_data.py        # Indexing script
│   ├── config.py           # Env key loader
│   └── __init__.py
│── QuickRAG_Demo.ipynb     # Jupyter Notebook demo
│── requirements.txt
│── README.md

```

### 👨‍💻 Author
#### Naman Vora

Final Year CSE Student | Aspiring Data Analyst
📫 [LinkedIn↗](www.linkedin.com/in/namanvora21) • [GitHub↗](https://github.com/nv2105)

### 📄 License

This project is open source and available under the [MIT License.↗](https://mit-license.org/)