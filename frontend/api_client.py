import os
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

def ingest_pdfs(pdf_dir: str) -> dict:
    url = f"{BACKEND_URL}/ingest"
    payload = {"pdf_dir": pdf_dir}
    # Generous timeout of 300 seconds for long-running LLM processing
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()

def fetch_graph(limit: int = 50) -> dict:
    url = f"{BACKEND_URL}/graph"
    params = {"limit": limit}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def query_graphrag(question: str) -> dict:
    url = f"{BACKEND_URL}/graphrag"
    payload = {"question": question}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()
