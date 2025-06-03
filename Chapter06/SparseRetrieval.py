import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain.indexes import BM25Retriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Download NLTK tokenizer resources
nltk.download('punkt')
# Sample documents
documents = [
    Document(page_content="Quantum computing leverages qubits for advanced computation.", metadata={"category": "science"}),
    Document(page_content="AI is transforming industries like finance, healthcare, and robotics.", metadata={"category": "technology"}),
    Document(page_content="Stock markets fluctuate based on economic trends and global policies.", metadata={"category": "finance"}),
    Document(page_content="Newton's laws define classical mechanics and motion principles.", metadata={"category": "science"})
]								
# Tokenize documents for BM25 Indexing
tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)  # Create BM25 index

# Function to tokenize user query
def preprocess_query(query):
    return word_tokenize(query.lower())

# Function to retrieve top-N documents
def retrieve_documents(query, top_n=2):		
    tokenized_query = preprocess_query(query)  # Query Tokenization
    scores = bm25.get_scores(tokenized_query)  # Extract Term Matching & Scoring
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    retrieved_docs = [documents[i] for i in ranked_indices]
    return retrieved_docs

# Function to generate response using retrieved documents and LLM
def generate_response(query, retrieved_docs):		
    chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")
    # Prompt Engineering to instruct LLM on how to use retrieved docs
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template="Based on the retrieved context below, answer the question accurately.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    )								
    # Combine retrieved document content
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    # Generate response using LLM
    final_prompt = prompt_template.format(query=query, context=context)
    response = chat_model(final_prompt)
    return response.content

# Example Query
query = "How does quantum computing work?"
retrieved_docs = retrieve_documents(query)
response = generate_response(query, retrieved_docs)

# Display Results
print("\n--- Retrieved Documents ---")
for doc in retrieved_docs:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}")

print("\n--- LLM Response ---")
print(response)
