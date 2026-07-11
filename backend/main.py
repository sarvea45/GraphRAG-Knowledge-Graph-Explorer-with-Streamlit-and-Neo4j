from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

from ingest import process_directory, fetch_graph_data
from graphrag import execute_graphrag

app = FastAPI(title="GraphRAG API")

class IngestRequest(BaseModel):
    pdf_dir: str

class GraphRagRequest(BaseModel):
    question: str

@app.post("/ingest")
def run_ingest(request: IngestRequest):
    try:
        result = process_directory(request.pdf_dir)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
def get_graph(limit: int = 50):
    try:
        return fetch_graph_data(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graphrag")
def run_graphrag(request: GraphRagRequest):
    try:
        return execute_graphrag(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
