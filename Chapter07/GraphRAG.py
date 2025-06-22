import os
import nltk
import numpy as np
import networkx as nx
import re
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from nltk.corpus import reuters
from dotenv import load_dotenv
import openai

# ----------------------------------------
# Setup environment
# ----------------------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------------------
# Step 1: Load Reuters Dataset
# ----------------------------------------
def load_documents(limit=500):
    nltk.download('reuters')
    fileids = reuters.fileids()[:limit]
    docs = [reuters.raw(fid) for fid in fileids]
    print(f"Using {len(docs)} Reuters documents for processing.")
    return docs

# ----------------------------------------
# Step 2: Generate Embeddings
# ----------------------------------------
def embed_documents(docs):
    print("Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, show_progress_bar=True)
    return model, embeddings

# ----------------------------------------
# Step 3: Build Similarity Graph
# ----------------------------------------
def build_similarity_graph(docs, embeddings, k=5):
    print("Constructing similarity graph...")
    G = nx.Graph()
    for i in range(len(docs)):
        G.add_node(i, text=docs[i], embedding=embeddings[i])
    for i in range(len(docs)):
        sim = np.dot(embeddings, embeddings[i])
        similar_indices = np.argsort(sim)[::-1][1:k+1]
        for j in similar_indices:
            G.add_edge(i, j, weight=sim[j])
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# ----------------------------------------
# Step 4: Community Detection
# ----------------------------------------
def detect_communities(G):
    print("Detecting communities...")
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    print(f"Found {len(communities)} communities.")
    for idx, community in enumerate(communities[:3]):
        print(f"Community {idx+1}: {len(community)} nodes")

# ----------------------------------------
# Step 5: Indexing with ChromaDB
# ----------------------------------------
def index_with_chromadb(embeddings, docs):
    print("Setting up ChromaDB...")
    client = PersistentClient(path=".chromadb_data")
    collection = client.get_or_create_collection("graph_rag_reuters")
    ids = [str(i) for i in range(len(docs))]

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings.tolist()
    )
    print("Documents indexed in ChromaDB.")
    return collection

# ----------------------------------------
# Step 6: Query ChromaDB
# ----------------------------------------
def query_chromadb(collection, model, docs, query_text):
    print(f"Running query: '{query_text}'")
    query_embedding = model.encode([query_text]).tolist()[0]
    query_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "distances"]
    )
    retrieved_info = []
    for idx, doc in enumerate(query_results["documents"][0]):
        retrieved_info.append({
            "id": query_results["ids"][0][idx],
            "distance": query_results["distances"][0][idx],
            "text": doc
        })
    print("ChromaDB search results:")
    for info in retrieved_info:
        print(f"ID: {info['id']} - Distance: {info['distance']:.4f}")
    return retrieved_info

# ----------------------------------------
# Step 7: Generate Response with OpenAI GPT
# ----------------------------------------
def generate_response_with_gpt(query_text, retrieved_info):
    context_snippets = "\n\n".join([
        f"Document ID {info['id']}:\n{info['text'][:300]}..." for info in retrieved_info
    ])
    prompt = f"""
We have retrieved a few documents based on the query: "{query_text}".

Below are summaries extracted from the top matching documents:

{context_snippets}

Please provide a clear, concise summary that combines the key insights from these documents regarding financial market trends and investment news.
    """
    print("Generating final response using GPT-4o...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert summarizer for financial topics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        print("OpenAI API error:", e)
        return ""

# ----------------------------------------
# Execution
# ----------------------------------------
def main():
    try:
        docs = load_documents()
        if not docs:
            print("No documents loaded. Exiting.")
            return

        model, embeddings = embed_documents(docs)
        G = build_similarity_graph(docs, embeddings)
        detect_communities(G)

        collection = index_with_chromadb(embeddings, docs)
        if not collection:
            print("Failed to index documents. Exiting.")
            return

        query = "Analyze how oil price fluctuations have influenced stock markets, currency valuations, and international trade over the past decade. Highlight the causal relationships among these factors. Can you also highlight the points to support the response?"
        retrieved_info = query_chromadb(collection, model, docs, query)
        if not retrieved_info:
            print("No documents retrieved. Exiting.")
            return

        final_response = generate_response_with_gpt(query, retrieved_info)
        print("\n--- Final Answer ---\n")
        print(final_response)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
