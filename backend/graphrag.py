import os
from openai import OpenAI
from neo4j import GraphDatabase

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def execute_graphrag(question: str):
    question_embedding = generate_embedding(question)
    driver = get_neo4j_driver()
    
    anchor_nodes = []
    subgraph_context_lines = []
    
    with driver.session() as session:
        # Vector search for anchor nodes
        query = """
        CALL db.index.vector.queryNodes('entity_embeddings', 5, $embedding)
        YIELD node, score
        RETURN node.name AS name, node.type AS type, score
        """
        result = session.run(query, embedding=question_embedding)
        for record in result:
            anchor_nodes.append({
                "name": record["name"],
                "type": record["type"],
                "score": record["score"]
            })
            
        # Subgraph expansion
        if anchor_nodes:
            anchor_names = [n["name"] for n in anchor_nodes]
            expand_query = """
            MATCH (s:Entity)-[r]->(t:Entity)
            WHERE s.name IN $anchor_names OR t.name IN $anchor_names
            RETURN s.name AS source, type(r) AS relation, t.name AS target
            LIMIT 50
            """
            result = session.run(expand_query, anchor_names=anchor_names)
            for record in result:
                subgraph_context_lines.append(f"{record['source']} -[{record['relation']}]-> {record['target']}")
                
    driver.close()
    
    subgraph_context = "\n".join(subgraph_context_lines)
    
    prompt = f"""
    Answer the following question based on the provided knowledge graph context.
    If the context doesn't contain the answer, say "I don't know based on the provided context."
    
    Context:
    {subgraph_context}
    
    Question: {question}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on a provided knowledge graph."},
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    
    return {
        "question": question,
        "anchor_nodes": anchor_nodes,
        "subgraph_context": subgraph_context,
        "answer": answer
    }
