from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample long document
parent_document = """ Quantum computing is a revolutionary field that leverages quantum mechanics
to perform computations at unprecedented speeds. Unlike classical computing, which uses bits,
quantum computers use qubits that can exist in superposition states. This allows quantum
computers to solve problems exponentially faster than classical computers, particularly in areas
like cryptography, optimization, and material science."""

# Define metadata (parent document ID)
parent_id = "doc1"

# Chunk the document into small segments for better indexing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
small_chunks = text_splitter.split_text(parent_document)

# Convert chunks into Document objects with metadata
documents = [
    Document(page_content=chunk, metadata={"parent_id": parent_id}) for chunk in small_chunks
]

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()
# Store small chunks in FAISS for retrieval
vector_db = FAISS.from_documents(documents, embedding_model)

# Simulate Parent Document Storage (Dictionary for simplicity)
parent_docs = {
    "doc1": parent_document  # Full document stored separately
}

# Retrieve the most relevant small chunk	
retriever = vector_db.as_retriever()
query = "How do qubits enable faster computing?"
retrieved_chunk = retriever.get_relevant_documents(query)[0]

# Fetch the full parent document using parent_id
full_context = parent_docs[retrieved_chunk.metadata["parent_id"]]

# Print results
print("\n--- Retrieved Small Chunk ---")
print(retrieved_chunk.page_content)
print("\n--- Full Parent Document ---")
print(full_context)
