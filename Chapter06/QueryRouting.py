from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.indexes import BM25Retriever
from langchain.chat_models import ChatOpenAI

# Sample documents categorized into domains
documents = [
    Document(page_content="Quantum computing leverages qubits for superposition.", metadata={"category": "science"}),
    Document(page_content="AI is transforming industries like finance and healthcare.", metadata={"category": "technology"}),
    Document(page_content="The stock market is volatile due to economic fluctuations.", metadata={"category": "finance"}),
    Document(page_content="Newton's laws describe classical mechanics.", metadata={"category": "science"})
]

# Initialize OpenAI embeddings for FAISS
embedding_model = OpenAIEmbeddings()

# Create FAISS (vector-based retrieval) for Science & Technology
science_docs = [doc for doc in documents if doc.metadata["category"] in ["science", "technology"]]
science_vector_db = FAISS.from_documents(science_docs, embedding_model)
science_retriever = science_vector_db.as_retriever()

# Create BM25 (keyword-based retrieval) for Finance
finance_docs = [doc for doc in documents if doc.metadata["category"] == "finance"]
finance_retriever = BM25Retriever.from_documents(finance_docs)

# Function to route queries based on keywords
def route_query(query):
    if "quantum" in query or "AI" in query or "Newton" in query:
        return science_retriever
    elif "stock" in query or "market" in query or "finance" in query:
        return finance_retriever
    else:
        return None  # Default case

# Query the router			
query = "How does quantum computing work?"
retriever = route_query(query)
retrieved_docs = retriever.get_relevant_documents(query) if retriever else ["No relevant retriever found."]

# Print retrieved results	
for doc in retrieved_docs:
    print("\n--- Retrieved Document ---")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
