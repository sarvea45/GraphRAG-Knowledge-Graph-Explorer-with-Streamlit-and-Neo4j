# GraphRAG Knowledge Graph Explorer

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) application. It extracts entities and relationships from uploaded PDFs using LLMs, stores them in a Neo4j knowledge graph, and allows users to explore the graph visually or chat with it using an interactive Streamlit UI.

## Architecture

- **Frontend:** Streamlit application providing the user interface.
- **Backend:** FastAPI service orchestrating document ingestion and GraphRAG queries.
- **Database:** Neo4j graph database storing the knowledge graph and providing vector search capabilities.
- **LLM:** OpenAI (gpt-4o for extraction and answering, text-embedding-3-small for embeddings).

## Setup

1. Copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY`.
2. Run `docker-compose up --build`.
3. Access the Streamlit application at `http://localhost:8501`.
4. Access the Neo4j browser at `http://localhost:7474`.
