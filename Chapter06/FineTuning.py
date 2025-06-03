from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Define domain-specific documents
documents = [
    Document(page_content="Quantum computing uses qubits and superposition to perform calculations beyond classical computing.", metadata={"domain": "science", "difficulty": "advanced"}),
    Document(page_content="AI is revolutionizing finance by automating trading algorithms and risk assessment.", metadata={"domain": "finance", "difficulty": "intermediate"}),
    Document(page_content="Newton's laws describe motion and gravity, forming the basis of classical mechanics.", metadata={"domain": "science", "difficulty": "basic"}),
    Document(page_content="Blockchain technology secures transactions in decentralized finance (DeFi).", metadata={"domain": "technology", "difficulty": "intermediate"})
]

# Initialize OpenAI embeddings (fine-tuned for science and technology)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002") # Using a more precise embedding model

# Store documents in FAISS with fine-tuned embeddings
vector_db = FAISS.from_documents(documents, embedding_model)

# Fine-Tune Retrieval by Adjusting Parameters
# Retrieves top-2 docs with similarity above 0.8				
retriever = vector_db.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.8})  

# Function to generate response using fine-tuned retrieval
def generate_response(query):					
    retrieved_docs = retriever.get_relevant_documents(query)
    if not retrieved_docs:
        return "No relevant documents found."
    # Define a structured prompt template
    response_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Use the retrieved context below to answer the query accurately.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
    )

    # Combine retrieved documents into context
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    # Initialize LLM
    chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")
    # Generate response using LLM
    final_prompt = response_prompt.format(query=query, context=context)
    response = chat_model(final_prompt)
    return response.content			

query = "How does quantum computing outperform classical computing?"
response = generate_response(query)
# Display Results
print("\n--- Fine-Tuned Retrieval & LLM Response ---")
print(response)
