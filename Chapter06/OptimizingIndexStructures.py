import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

documents = [						
    Document(page_content="Quantum computing is a breakthrough in computational science.", metadata={"category": "Science"}),
    Document(page_content="AI is transforming the healthcare industry.", metadata={"category": "Technology"}),
    Document(page_content="Nanotechnology is advancing at a rapid pace.", metadata={"category": "Science"}),
    Document(page_content="The stock market is experiencing volatility due to economic shifts.", metadata={"category": "Finance"})
]
# Initialize OpenAI embeddings				
embedding_model = OpenAIEmbeddings()

# Convert documents to embeddings					
texts = [doc.page_content for doc in documents]
embeddings = embedding_model.embed_documents(texts)

# Convert list to FAISS indexable format			
dimension = len(embeddings[0])  # Get embedding vector dimension
# IVF index with 10 centroids
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 10)  

# Train index (important for IVF indices)
index.train(embeddings)
index.add(embeddings)  # Add vectors to the FAISS index

# Store in FAISS vector DB
vector_db = FAISS(index=index, embedding_function=embedding_model.embed_query)

# Query the retriever
retriever = vector_db.as_retriever()
query = "What is new in computational science?"
retrieved_docs = retriever.get_relevant_documents(query)

# Print retrieved results
for doc in retrieved_docs:
    print("\n--- Retrieved Document ---")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
