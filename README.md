# 🔍 RAG First Project

**A Retrieval-Augmented Generation (RAG) pipeline built from scratch — chunk, embed, index, retrieve, and summarize research papers using FAISS.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-00A67E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Learning%20Project-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

---

## 📖 Overview

This is my **first hands-on Retrieval-Augmented Generation (RAG) system** — built to understand how modern AI search pipelines actually work under the hood, from raw PDFs to a grounded, summarized answer.

Instead of relying on a pre-built framework end-to-end, this project implements the core RAG loop manually:

```
PDF Documents → Chunking → Embeddings → FAISS Index → Similarity Search → Contextual Summary
```

The knowledge base is built from a small but meaningful collection of **foundational deep learning research papers**, making this a mini "ask-a-question-about-deep-learning-history" engine.

---

## 🧠 How It Works

1. **`src/data_loader.py`** — Loads and parses all PDF documents from the `data/` directory.
2. **`src/vectorstore.py`** — Converts document chunks into vector embeddings and builds/loads a **FAISS** index (`faiss_store/`) for fast similarity search.
3. **`src/search.py`** — Runs a semantic query against the FAISS index, retrieves the top-k most relevant chunks, and generates a summarized answer grounded in those chunks.
4. **`app.py`** — Ties it all together: builds the index from documents, then runs an example query end-to-end.

```python
docs = load_all_documents("data")
store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)

rag_search = RAGSearch()
summary = rag_search.search_and_summarize("What is Bert?", top_k=3)
```

---

## 📚 Knowledge Base — Papers Used for Retrieval

The retrieval corpus is built from **four landmark deep learning papers**, each chosen for a specific reason — together they cover the backbone of modern NLP and computer vision architectures:

| # | Paper | Why It's Included |
|---|-------|--------------------|
| 1 | **Attention Is All You Need** (Vaswani et al., 2017) | Introduced the **Transformer** architecture — the origin of self-attention, multi-head attention, and the encoder-decoder design that replaced RNNs. Core reference for understanding *how* attention works. |
| 2 | **Transformer Architecture & Attention Mechanisms** (overview doc) | A distilled, explanatory companion to the original paper — useful for testing retrieval on *conceptual* explanations (scaled dot-product attention, positional encoding, decoder-only models) rather than raw research prose. |
| 3 | **Deep Residual Learning for Image Recognition** (He et al., 2015) | Introduced **ResNets** and residual/skip connections — the paper that made training very deep networks (100+ layers) possible. Included to test retrieval across a *different domain* (computer vision vs. NLP). |
| 4 | **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018) | Extends the Transformer's *encoder* for bidirectional language understanding — used to test whether the system can retrieve and summarize a paper that *builds on* concepts from paper #1, proving the pipeline handles cross-referenced knowledge well. |

> 💡 Using papers that build on one another (Transformer → BERT, plus a CV outlier like ResNet) makes this a great stress-test for retrieval accuracy — the system needs to pull the *right* paper, not just any paper mentioning "attention" or "deep learning."

---

## 🗂️ Project Structure

```
rag_first_project/
├── app.py                 # Entry point — builds index & runs example query
├── src/
│   ├── data_loader.py      # Loads & parses PDFs from data/
│   ├── vectorstore.py       # FAISS index builder + loader
│   └── search.py            # Semantic search + summarization logic
├── faiss_store/            # Persisted FAISS vector index
├── data/                   # Source PDFs (not tracked / add your own)
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repo**
```bash
git clone https://github.com/mayank-1584/rag_first_project.git
cd rag_first_project
```

**2. Install dependencies**

Using `uv` (recommended, since this repo ships a `uv.lock`):
```bash
uv sync
```

**3. Add your documents**

Place your PDF files inside a `data/` folder in the project root.

**4. Run the pipeline**
```bash
python app.py
```

This will:
- Load all PDFs from `data/`
- Build (or load) the FAISS vector index in `faiss_store/`
- Run a sample query and print a summarized, retrieval-grounded answer

---

## 🚀 Example

```python
rag_search = RAGSearch()
summary = rag_search.search_and_summarize("What is BERT?", top_k=3)
print("Summary:", summary)
```

---

## 🛣️ Roadmap / Ideas for Next Steps

- [ ] Add a simple CLI or Streamlit UI for interactive querying
- [ ] Support chunk-level source citations in the summary output
- [ ] Add evaluation metrics (retrieval precision@k)
- [ ] Swap in different embedding models for comparison

---

## 📝 License

This project is licensed under the MIT License — feel free to fork, learn from it, and build your own RAG experiments on top of it.

---

<p align="center">Built as a learning project to understand Retrieval-Augmented Generation from the ground up 🚀</p>