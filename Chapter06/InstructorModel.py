from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Sample documents
documents = [
    Document(page_content="Quantum computing research at MIT focuses on improving qubit stability and quantum error correction.", metadata={"domain": "science", "organization": "MIT"}),
    Document(page_content="AI-powered medical diagnostics help doctors detect diseases early.", metadata={"domain": "healthcare", "organization": "Harvard Medical School"}),
    Document(page_content="The stock market reacts to inflation data and policy changes by central banks.", metadata={"domain": "finance", "organization": "Federal Reserve"}),
    Document(page_content="New research on the human brain reveals how neural networks process memory.", metadata={"domain": "science", "organization": "Stanford University"})
]

# Initialize OpenAI embeddings with Instructor Model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Store documents in FAISS with optimized embeddings
vector_db = FAISS.from_documents(documents, embedding_model)

# Initialize Instructor Model (GPT-4o)			
chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")

# Define a prompt for refining the user query
query_refinement_prompt = PromptTemplate(
    input_variables=["query"],
    template="Rephrase the following query to improve search accuracy, ensuring clarity and domain specificity:\n\nQuery: {query}\n\nOptimized Query:"
)
# Create LangChain LLM chain for query refinement	
query_refinement_chain = LLMChain(llm=chat_model, prompt=query_refinement_prompt)

# Function to process query using instructor model and retrieve optimized results
def instructor_retrieval(query, top_n=3):				
    # Optimize the query using the Instructor Model
    optimized_query = query_refinement_chain.run(query=query)

    # Retrieve relevant documents from FAISS
    retriever = vector_db.as_retriever(search_kwargs={"k": top_n})
    retrieved_docs = retriever.get_relevant_documents(optimized_query)
    return retrieved_docs, optimized_query

# Example Query
query = "Tell me about the latest research in quantum computing?"
retrieved_docs, refined_query = instructor_retrieval(query)

# Print Results
print("\n--- Refined Query ---")
print(refined_query)

print("\n--- Retrieved Documents ---")
for doc in retrieved_docs:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}")

# Function to generate structured response using Instructor Model
def generate_response(query, retrieved_docs):
    # Define a structured prompt template
    response_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Use the optimized context below to answer the query factually.\n\nOptimized Context: {context}\n\nQuery: {query}\n\nAnswer:"
    )

    # Combine retrieved documents into structured context
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate response using LLM
    final_prompt = response_prompt.format(query=query, context=context)
    response = chat_model(final_prompt)
    return response.content

# Generate the LLM response
response = generate_response(query, retrieved_docs)

# Display Results
print("\n--- LLM Response ---")
print(response)
