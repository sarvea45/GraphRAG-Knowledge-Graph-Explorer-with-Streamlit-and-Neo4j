import os
import glob
import json
import uuid
import pdfplumber
from openai import OpenAI
from neo4j import GraphDatabase
from fastembed import TextEmbedding

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_entities_relationships(chunk):
    prompt = """
    Extract entities and relationships from the following text to build a detailed knowledge graph.
    IMPORTANT INSTRUCTION: Do NOT generalize specific numbers, quantities, salaries, or dates. You MUST capture highly specific details (e.g. 'INR 9,41,460', '7 CGPA', '2027') either as their own distinct Entities, or vividly include them in the 'description' field of relevant entities.
    
    Output strictly in JSON format with the following structure:
    {
      "entities": [
        {"name": "Entity Name", "type": "Entity Type (e.g., Concept, Algorithm, Technology, Person, Organization, Location, Metric, Requirement)", "description": "Highly detailed description including specific numbers, values, and constraints if applicable"}
      ],
      "relationships": [
        {"source": "Source Entity Name", "target": "Target Entity Name", "relation": "RELATION_TYPE_IN_UPPERCASE"}
      ]
    }
    Text:
    """ + chunk

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured data for a knowledge graph. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }
    )
    
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}

def generate_embedding(text):
    embedding = list(embedding_model.embed([text]))[0].tolist()
    return embedding

def setup_database():
    driver = get_neo4j_driver()
    with driver.session() as session:
        # Create vector index if it doesn't exist
        try:
            session.run("""
            CALL db.index.vector.createNodeIndex(
                'entity_embeddings',
                'Entity',
                'embedding',
                384,
                'cosine'
            )
            """)
        except Exception:
            pass
        # Create constraint on name to avoid duplicates
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
    driver.close()

def process_directory(pdf_dir):
    setup_database()
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    total_nodes = 0
    total_edges = 0
    
    driver = get_neo4j_driver()
    
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        
        for chunk in chunks:
            data = extract_entities_relationships(chunk)
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            
            # Map of entity names to their data
            entity_map = {e['name']: e for e in entities if 'name' in e}
            
            with driver.session() as session:
                for ent_name, ent_data in entity_map.items():
                    ent_type = ent_data.get("type", "Concept")
                    ent_desc = ent_data.get("description", "")
                    embedding = generate_embedding(ent_desc if ent_desc else ent_name)
                    ent_id = str(uuid.uuid4())
                    
                    query = f"""
                    MERGE (e:Entity {{name: $name}})
                    ON CREATE SET e.id = $id
                    SET e.type = $type, e.description = $desc, e.embedding = $embedding
                    WITH e
                    CALL apoc.create.addLabels(e, [$type]) YIELD node
                    RETURN e
                    """
                    session.run(query, name=ent_name, id=ent_id, type=ent_type, desc=ent_desc, embedding=embedding)
                    total_nodes += 1
                
                for rel in relationships:
                    source = rel.get("source")
                    target = rel.get("target")
                    relation = rel.get("relation", "RELATED_TO").upper().replace(" ", "_")
                    
                    if source and target:
                        query = f"""
                        MATCH (s:Entity {{name: $source}})
                        MATCH (t:Entity {{name: $target}})
                        MERGE (s)-[r:{relation}]->(t)
                        """
                        session.run(query, source=source, target=target)
                        total_edges += 1
                        
    driver.close()
    
    return {
        "status": "success",
        "nodes_created": total_nodes,
        "edges_created": total_edges,
        "files_processed": len(pdf_files)
    }

def fetch_graph_data(limit=50):
    driver = get_neo4j_driver()
    nodes = []
    edges = []
    with driver.session() as session:
        # Get most connected nodes
        node_query = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as connections
        ORDER BY connections DESC
        LIMIT $limit
        RETURN n.id AS id, n.name AS name, n.type AS type, n.description AS description
        """
        result = session.run(node_query, limit=limit)
        node_ids = set()
        for record in result:
            nodes.append(dict(record))
            node_ids.add(record["id"])
            
        # Get edges between these nodes
        edge_query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE s.id IN $node_ids AND t.id IN $node_ids
        RETURN s.id AS source, t.id AS target, type(r) AS relation
        """
        result = session.run(edge_query, node_ids=list(node_ids))
        for record in result:
            edges.append(dict(record))
            
    driver.close()
    return {"nodes": nodes, "edges": edges}
