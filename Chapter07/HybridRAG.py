import os
import nltk
import chromadb
import networkx as nx
import re
from chromadb import PersistentClient
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from nltk.corpus import reuters
from dotenv import load_dotenv
import openai

# ----------------------------------------
# Setup
# ----------------------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------
# Step 1: Load Reuters dataset
# ----------------------------------------
nltk.download("reuters")
docs = [reuters.raw(fid) for fid in reuters.fileids()[:100]]
print(f"Loaded {len(docs)} Reuters documents.")

# ----------------------------------------
# Step 2: Generate Embeddings
# ----------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, show_progress_bar=True)

# ----------------------------------------
# Step 3: Store Embeddings in ChromaDB
# ----------------------------------------
print("Setting up ChromaDB...")
chromadb_client = PersistentClient(path=".chromadb_data")
collection = chromadb_client.get_or_create_collection(name="hybridrag_reuters")

ids = [str(i) for i in range(len(docs))]
collection.add(ids=ids, documents=docs, embeddings=embeddings.tolist())
print("Documents indexed in ChromaDB.")

# ----------------------------------------
# Step 4: Extract Triples Using GPT
# ----------------------------------------
def extract_triples(doc_text):
    prompt = f"""
Extract entity-relation-entity triples from the following text:
{doc_text[:1000]}
Format: (Entity1, Relation, Entity2)
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2
    )
    text = response.choices[0].message.content.strip()
    triples = re.findall(r"\(([^,]+), ([^,]+), ([^)]+)\)", text)
    return triples

# ----------------------------------------
# Step 5: Build Knowledge Graph + Insert to Neo4j
# ----------------------------------------
graph = nx.Graph()

print("Building knowledge graph...")
for doc in docs[:10]:  # limit for speed
    triples = extract_triples(doc)
    for subj, rel, obj in triples:
        graph.add_edge(subj.strip(), obj.strip(), relation=rel.strip())

print(f"Knowledge Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "your_neo4j_password"))

def add_to_neo4j(triples):
    with driver.session() as session:
        for subj, rel, obj in triples:
            session.run(
                """
                MERGE (a:Entity {name: $subj})
                MERGE (b:Entity {name: $obj})
                MERGE (a)-[r:RELATION {type: $rel}]->(b)
                """,
                subj=subj.strip(), obj=obj.strip(), rel=rel.strip()
            )

print("Inserting triples into Neo4j...")
for doc in docs[:10]:
    triples = extract_triples(doc)
    add_to_neo4j(triples)

print("Knowledge Graph inserted into Neo4j.")

# ----------------------------------------
# Step 6: Hybrid Retrieval
# ----------------------------------------
def hybrid_retrieve(query_text):
    query_embedding = model.encode([query_text])[0]

    vector_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
        include=["documents", "embeddings"]
    )

    docs_with_scores = []
    for doc_text, doc_emb in zip(vector_results["documents"][0], vector_results["embeddings"][0]):
        score = float(np.dot(query_embedding, doc_emb))
        docs_with_scores.append((doc_text, score))

    reranked_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)[:5]
    top_docs = [doc for doc, _ in reranked_docs]

    with driver.session() as session:
        graph_results = session.run(
            """
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            WHERE toLower(a.name) CONTAINS toLower($query) OR toLower(b.name) CONTAINS toLower($query)
            RETURN a.name AS entity1, r.type AS relation, b.name AS entity2
            """,
            query=query_text
        )
        triples = [f"{record['entity1']} -[{record['relation']}]-> {record['entity2']}" for record in graph_results]

    combined_context = "\n\n".join(top_docs + triples)
    return combined_context

# ----------------------------------------
# Step 7: Final Answer Generation with GPT
# ----------------------------------------
def generate_summary_with_gpt(query_text, context):
    prompt = f"""
We have retrieved the following information based on the query: "{query_text}".

Context:
{context}

Please summarize and explain the causal relationships. Present key insights clearly.
"""

    print("Generating final response using GPT-4o...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API error:", e)
        return ""

# ----------------------------------------
# Step 8: Execution
# ----------------------------------------
query_text = "Analyze how oil price fluctuations have influenced stock markets, currency valuations, and international trade over the past decade. Highlight the causal relationships among these factors. Can you also highlight the points to support the response?"

print(f"Running Hybrid RAG for query: {query_text}")
retrieved_context = hybrid_retrieve(query_text)
final_answer = generate_summary_with_gpt(query_text, retrieved_context)

print("--- Final Answer ---")
print(final_answer)
