import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
	
# Download tokenizer resources
nltk.download('punkt')  
documents = [								
    Document(page_content="Quantum computing leverages qubits instead of classical bits.", metadata={"category": "science"}),
    Document(page_content="AI is revolutionizing healthcare with advanced diagnostics.", metadata={"category": "technology"}),
    Document(page_content="Stock markets fluctuate based on economic trends and global policies.", metadata={"category": "finance"}),
    Document(page_content="Newton’s laws of motion form the foundation of classical mechanics.", metadata={"category": "science"})
]

# Tokenize documents for BM25 Indexing
tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in documents]

# Create BM25 index
bm25 = BM25Okapi(tokenized_corpus)  

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Store documents in FAISS for vector search
vector_db = FAISS.from_documents(documents, embedding_model)

# Function to process query for both BM25 and FAISS
def retrieve_hybrid(query, bm25_weight=0.5, dense_weight=0.5, top_n=2):
    # BM25 retrieval
    tokenized_query = word_tokenize(query.lower())  # Query tokenization
    bm25_scores = bm25.get_scores(tokenized_query)  # BM25 term matching
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]
    bm25_results = {documents[i].page_content: bm25_scores[i] * bm25_weight for i in bm25_ranked_indices}
    
    # Dense retrieval using FAISS
    retrieved_dense_docs = vector_db.similarity_search(query, k=top_n)
    dense_results = {doc.page_content: dense_weight for doc in retrieved_dense_docs}
    
    # Merge and re-rank results
    hybrid_results = bm25_results
    for doc, score in dense_results.items():
        hybrid_results[doc] = hybrid_results.get(doc, 0) + score  # Combine scores
    
    # Sort final results by combined score
    ranked_results = sorted(hybrid_results.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_results[:top_n]]  # Return top-N results

# Function to generate response using retrieved documents
def generate_response(query, retrieved_docs):
    chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")
    
    # Define a structured prompt template
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="Use the retrieved context below to answer the query accurately.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
    )
    
    # Combine retrieved documents into context
    context = "\n".join(retrieved_docs)
    
    # Generate response using LLM
    final_prompt = prompt_template.format(query=query, context=context)
    response = chat_model(final_prompt)
    return response.content

query = "How does quantum computing work?"
retrieved_docs = retrieve_hybrid(query, bm25_weight=0.5, dense_weight=0.5, top_n=2)
response = generate_response(query, retrieved_docs)

# Display Results
print("\n--- Retrieved Documents ---")
for doc in retrieved_docs:
    print(f"Content: {doc}\n")
print("\n--- LLM Response ---")
print(response)
