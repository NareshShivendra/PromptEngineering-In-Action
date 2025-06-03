from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Define sample documents with metadata
documents = [
    Document(page_content="Quantum computing uses qubits instead of classical bits.", metadata={"category": "Science", "date": "2024-06-01"}),
    Document(page_content="AI is transforming industries like healthcare and finance.", metadata={"category": "Technology", "date": "2024-05-20"}),
    Document(page_content="The latest breakthroughs in nanotechnology were published this month.", metadata={"category": "Science", "date": "2024-06-05"}),
    Document(page_content="Stock markets fluctuate based on economic indicators.", metadata={"category": "Finance", "date": "2024-06-02"})
]

# Initialize OpenAI embedding model
embedding_model = OpenAIEmbeddings()

# Store documents in FAISS with metadata
vector_db = FAISS.from_documents(documents, embedding_model)

# Define retriever with metadata filtering
retriever = vector_db.as_retriever(search_kwargs={"filter": {"category": "Science"}})

# Query with metadata filtering
query = "What is the latest in quantum computing?"
retrieved_docs = retriever.get_relevant_documents(query)

# Print retrieved results
for doc in retrieved_docs:
    print("\n--- Retrieved Document ---")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
