import streamlit as st
import os
import json
from pyvis.network import Network
import streamlit.components.v1 as components

from api_client import ingest_pdfs, fetch_graph, query_graphrag

# Setup directories
PDF_DIR = "/data/pdfs/"
os.makedirs(PDF_DIR, exist_ok=True)

# Initialize Session State
if "graph_built" not in st.session_state:
    st.session_state["graph_built"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.set_page_config(page_title="GraphRAG Explorer", layout="wide")
st.title("GraphRAG Knowledge Graph Explorer")

# --- Sidebar ---
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Build Graph"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            # Save files
            for file in uploaded_files:
                file_path = os.path.join(PDF_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            with st.spinner("Building knowledge graph... this may take a minute."):
                try:
                    res = ingest_pdfs(PDF_DIR)
                    st.success(f"Graph built successfully!\n\nNodes: {res['nodes_created']}\nEdges: {res['edges_created']}\nFiles: {res['files_processed']}")
                    st.session_state["graph_built"] = True
                except Exception as e:
                    st.error(f"Error building graph: {e}")

# --- Tabs ---
tab1, tab2 = st.tabs(["Knowledge Graph", "Chat with Graph"])

# Color map based on user constraints
COLOR_MAP = {
    "Concept": "#A7C7E7",          # Soft Blue
    "Algorithm": "#9B59B6",        # Vibrant Purple
    "Method": "#9B59B6",
    "Technology": "#2ECC71",       # Emerald Green
    "Tool": "#2ECC71",
    "Organization": "#F39C12",     # Warm Amber
    "Company": "#F39C12",
    "Person": "#FF7F50",           # Coral
    "Location": "#708090",         # Slate Grey
    "Data": "#708090"
}

with tab1:
    if not st.session_state["graph_built"]:
        st.info("Upload PDFs and click 'Build Graph' in the sidebar to populate the knowledge graph.")
    else:
        try:
            graph_data = fetch_graph()
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            # Create PyVis Network
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            net.barnes_hut() # Enforce Barnes-Hut physics
            
            node_types = set()
            
            for node in nodes:
                node_type = node.get("type", "Concept")
                node_types.add(node_type)
                color = COLOR_MAP.get(node_type, "#97C2FC") # Default light blue
                
                title_tooltip = f"Type: {node_type}\nDescription: {node.get('description', '')}"
                
                net.add_node(
                    node["id"],
                    label=node["name"],
                    title=title_tooltip,
                    color=color
                )
                
            for edge in edges:
                net.add_edge(edge["source"], edge["target"], title=edge.get("relation", ""), label=edge.get("relation", ""))
                
            html_path = "/data/graph_render.html"
            net.save_graph(html_path)
            
            with open(html_path, "r", encoding="utf-8") as f:
                html_data = f.read()
                
            components.html(html_data, height=620)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Nodes", len(nodes))
            col2.metric("Total Edges", len(edges))
            col3.metric("Node Types", len(node_types))
            
        except Exception as e:
            st.error(f"Failed to fetch or render graph: {e}")

with tab2:
    for msg in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])
            with st.expander("View reasoning"):
                st.write("**Anchor Nodes:**")
                st.json(msg.get("anchor_nodes", []))
                st.write("**Subgraph Context:**")
                st.code(msg.get("subgraph_context", ""))

    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.spinner("Searching the knowledge graph..."):
            try:
                res = query_graphrag(question)
                st.session_state["chat_history"].append({
                    "question": res["question"],
                    "answer": res["answer"],
                    "anchor_nodes": res["anchor_nodes"],
                    "subgraph_context": res["subgraph_context"]
                })
                st.rerun()
            except Exception as e:
                st.error(f"Failed to query GraphRAG: {e}")
