# BioRAG-PubMed

A minimal **Retrieval-Augmented Generation (RAG)** project for bioinformatics using **public PubMed abstracts**.

This repository demonstrates:
- Retrieval of biological knowledge from public literature
- Vector embedding + similarity search
- LLM-based question answering grounded in retrieved evidence

## Why this project
- Bioinformatics-friendly RAG example
- Uses only public data (safe for open GitHub)
- Clean MVP suitable for portfolio / interviews

## What it will do (MVP)
1. Query PubMed using NCBI E-utilities
2. Build a local vector index from abstracts
3. Answer questions with citation-aware responses

## Planned stack
- Python
- PubMed (NCBI Entrez)
- SentenceTransformers (embeddings)
- FAISS (vector search)
- LLM (Ollama local or API-based)

## Repo structure (planned)
```
bio-rag-pubmed/
README.md
requirements.txt
app.py
scripts/
pubmed_fetch.py
build_index.py
src/
rag.py
```
