# AI Interview Question Generator (Hybrid RAG + LLM)

## Overview
An AI-powered interview question generator that extracts technical skills from job descriptions using semantic embeddings and generates structured interview questions using a hybrid Retrieval-Augmented Generation (RAG) pipeline.

## Features
- Semantic skill extraction from Job Descriptions
- Embedding-based similarity ranking
- Hybrid RAG architecture with confidence gating
- FAISS vector search
- Local LLM inference using TinyLlama
- Streamlit web interface

## Architecture
Job Description  
→ Skill Extraction (MiniLM Embeddings)  
→ Top-K Skill Ranking  
→ Hybrid RAG Retrieval  
→ TinyLlama Generation  
→ Structured Question Output  

## Tech Stack
- Python
- Streamlit
- FAISS
- SentenceTransformers (MiniLM)
- HuggingFace Transformers
- TinyLlama (1.1B Chat Model)

## How to Run

1. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

## Why This Project?
This project demonstrates:
- Retrieval-Augmented Generation (RAG)
- Embedding-based semantic search
- Confidence-based gating logic
- Local LLM deployment
- End-to-end AI system design
