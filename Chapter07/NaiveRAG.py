import os
import nltk
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from chromadb import PersistentClient
import openai



# Load environment variables (for OpenAI API Key)
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai_api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------------
# Step 1: Load Reuters Dataset
# ----------------------------------------
def load_reuters_docs(limit=500):
    if not nltk.corpus.reuters.fileids():
        nltk.download("reuters")
    fileids = nltk.corpus.reuters.fileids()[:limit]
    docs = [nltk.corpus.reuters.raw(fid) for fid in fileids]
    print(f"Using {len(docs)} Reuters documents for processing.")
    return docs

# ----------------------------------------
# Step 2: Generate Embeddings
# ----------------------------------------
def generate_embeddings(docs):
    print("Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True)
    return model, embeddings

# ----------------------------------------
# Step 3: Setup ChromaDB Vector Store
# ----------------------------------------
def setup_chromadb(docs, embeddings):
    print("Setting up ChromaDB...")
    
    client = PersistentClient(path=".chromadb_data")
    
    collection = client.get_or_create_collection(name="naive_rag_reuters")
    ids = [str(i) for i in range(len(docs))]
    
    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings.tolist()
    )
    
    print("Documents indexed in ChromaDB.")
    return collection

# ----------------------------------------
# Step 4: Retrieve Top-K Documents
# ----------------------------------------
def retrieve_documents(collection, query_text, model, top_k=3):
    print(f"Running query: '{query_text}'")
    query_embedding = model.encode([query_text]).tolist()[0]
    query_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )

    retrieved_info = []
    for idx, doc in enumerate(query_results["documents"][0]):
        retrieved_info.append({
            "id": query_results["ids"][0][idx],
            "distance": query_results["distances"][0][idx],
            "text": doc
        })

    print("Top retrieved documents:")
    for info in retrieved_info:
        print(f"ID: {info['id']} - Distance: {info['distance']:.4f}")
    return retrieved_info

# ----------------------------------------
# Step 5: Generate Final Response Using GPT
# ----------------------------------------
def generate_summary_with_gpt(query_text, retrieved_info):
    context_snippets = "\n\n".join([
        f"Document ID {info['id']}:\n{info['text'][:300]}..." for info in retrieved_info
    ])

    prompt = f"""
    We have retrieved a few documents based on the query: "{query_text}".
    Below are summaries extracted from the top matching documents:

    {context_snippets}

    Please provide a clear, concise summary that combines the key insights about current financial market trends and investment strategies.
    """
    print("Generating final response using ChatGPT API...")
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial assistant that provides concise, insightful answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        answer = response.choices[0].message.content
        print("Response generated successfully.")
        print(f"AI: {answer}")
    except Exception as e:
        answer = f"OpenAI API Error: {e}"

        # Output result
        print(f"\nUser: {query}")
        print(f"AI: {answer}")
        print(f"Sources: {sources}")
# ----------------------------------------
# Execution of Naive RAG
# ----------------------------------------
def main():
    docs = load_reuters_docs()
    model, embeddings = generate_embeddings(docs)
    collection = setup_chromadb(docs, embeddings)

    static_query = "Analyze how oil price fluctuations have influenced stock markets, currency valuations, and international trade over the past decade. Highlight the causal relationships among these factors. Can you also highlight the points to support the response?"
    retrieved_info = retrieve_documents(collection, static_query, model)

    generate_summary_with_gpt(static_query, retrieved_info)


if __name__ == "__main__":
    main()
