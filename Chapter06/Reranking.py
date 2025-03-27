import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

nltk.download('punkt')  # Download tokenizer resources
# Sample documents							
documents = [
    Document(page_content="Quantum computing research at MIT has led to major breakthroughs in 2024.", metadata={"category": "science", "date": "2024", "organization": "MIT"}),
    Document(page_content="AI-powered diagnostic tools are revolutionizing medical treatments.", metadata={"category": "healthcare", "date": "2023", "organization": "Harvard Medical School"}),
    Document(page_content="Stock markets are influenced by inflation and global economic shifts.", metadata={"category": "finance", "date": "2024", "organization": "Federal Reserve"}),
    Document(page_content="Recent advances in neuroscience have expanded our understanding of brain function.", metadata={"category": "science", "date": "2023", "organization": "Stanford University"})
]
# Tokenize documents for BM25 Indexing
tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)  # Create BM25 index

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Store documents in FAISS for vector search
vector_db = FAISS.from_documents(documents, embedding_model)

# Load Cross-Encoder Model for Re-Ranking	
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Function to retrieve documents using BM25 & FAISS		
def retrieve_documents(query, top_n=3):
    # BM25 Retrieval
    tokenized_query = word_tokenize(query.lower())  # Tokenize query
    bm25_scores = bm25.get_scores(tokenized_query)  # Compute BM25 scores
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]
    bm25_results = [documents[i] for i in bm25_ranked_indices]
    # Dense Retrieval using FAISS
    retrieved_dense_docs = vector_db.similarity_search(query, k=top_n)
    # Combine retrieved documents
    combined_results = bm25_results + retrieved_dense_docs
    unique_docs = list({doc.page_content: doc for doc in combined_results}.values())  # Remove duplicates
    return unique_docs

# Function to re-rank documents using a cross-encoder
def re_rank_documents(query, retrieved_docs):
    # Prepare query-document pairs for scoring
    query_doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
    # Compute relevance scores
    scores = cross_encoder.predict(query_doc_pairs)
    # Re-rank documents based on cross-encoder scores
    ranked_results = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_results]

# Function to generate response using re-ranked documents
def generate_response(query, retrieved_docs):
    chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")
    # Define a structured prompt template
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="Use the retrieved context below to answer the query accurately.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
    )
    # Combine retrieved documents into context
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    # Generate response using LLM
    final_prompt = prompt_template.format(query=query, context=context)
    response = chat_model(final_prompt)
    return response.content

query = "What are the latest advances in quantum computing?"
retrieved_docs = retrieve_documents(query, top_n=3)
re_ranked_docs = re_rank_documents(query, retrieved_docs)
response = generate_response(query, re_ranked_docs)

# Display Results
print("\n--- Retrieved Documents (After Re-Ranking) ---")
for doc in re_ranked_docs:
    print(f"Content: {doc.page_content}\n")
print("\n--- LLM Response ---")
print(response)
