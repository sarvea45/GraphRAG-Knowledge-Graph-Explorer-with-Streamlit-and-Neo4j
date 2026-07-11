# GraphRAG Knowledge Graph Explorer

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) application. It extracts entities and relationships from uploaded PDFs using LLMs, stores them in a Neo4j knowledge graph, and allows users to explore the graph visually or chat with it using an interactive Streamlit UI.

## Architecture

- **Frontend:** Streamlit application providing the user interface.
- **Backend:** FastAPI, Pydantic, FastEmbed (for local embeddings)
- **Database:** Neo4j (Graph Database with Vector Search capability)
- **LLM Provider:** Groq (using `llama-3.1-8b-instant` for lightning fast inference)

## Setup

1. Copy `.env.example` to `.env` and fill in your `GROQ_API_KEY`:
   ```bash
   cp .env.example .env
   ```

> **Note on Open Source Substitution:** As permitted by the assignment guidelines, this project has been configured to use free, open-source models to avoid paid API billing requirements. We replaced OpenAI's `gpt-4o` with Groq's `llama-3.1-8b-instant` for blazing fast text generation. Since Groq does not provide embedding models, we integrated `fastembed` in the backend to generate lightweight `BAAI/bge-small-en-v1.5` embeddings (384 dimensions) locally on the CPU. The Neo4j Vector Index was appropriately resized to 384 dimensions. All core API contracts remain identical.

2. Run `docker-compose up --build`.
3. Access the Streamlit application at `http://localhost:8501`.
4. Access the Neo4j browser at `http://localhost:7474`.
