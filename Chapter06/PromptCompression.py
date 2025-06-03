from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Sample documents							
documents = [
    Document(page_content="Quantum computing research at MIT has led to breakthroughs in 2024, including improved qubit coherence and error correction techniques.", metadata={"category": "science", "date": "2024", "organization": "MIT"}),
    Document(page_content="OpenAI's latest model introduces enhanced reasoning and contextual understanding, improving human-like text generation.", metadata={"category": "technology", "date": "2023", "organization": "OpenAI"}),
    Document(page_content="The US stock market surged due to economic policy changes, leading to record gains in the technology sector.", metadata={"category": "finance", "date": "2024", "organization": "Federal Reserve"}),
    Document(page_content="Recent advances in neuroscience have expanded our understanding of brain function, particularly in memory storage and retrieval.", metadata={"category": "science", "date": "2023", "organization": "Stanford University"})]

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()
# Store documents in FAISS for retrieval
vector_db = FAISS.from_documents(documents, embedding_model)

# Initialize LLM (GPT-4o)						
chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")

# Define a retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant docs

# Function to compress retrieved documents			
def compress_context(retrieved_docs):
    # Combine retrieved documents into a single text block
    full_context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Define a prompt for context compression
    compression_prompt = PromptTemplate(
        input_variables=["context"],
        template="Summarize the key facts from the following retrieved context while preserving essential details:\n\n{context}\n\nCompressed Summary:"
    )
    # Create LLM chain for compression
    compression_chain = LLMChain(llm=chat_model, prompt=compression_prompt)
    # Generate compressed context
    compressed_context = compression_chain.run(context=full_context)
    return compressed_context

# Function to generate response using compressed prompt		
def generate_response(query, compressed_context):
    # Define a structured prompt template
    response_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Use the compressed context below to answer the query accurately.\n\nCompressed Context: {context}\n\nQuery: {query}\n\nAnswer:"
    )
    # Format the final prompt
    final_prompt = response_prompt.format(query=query, context=compressed_context)
    # Generate response using LLM
    response = chat_model(final_prompt)

    return response.content

query = "What are the recent advancements in quantum computing?"
retrieved_docs = retriever.get_relevant_documents(query)
compressed_context = compress_context(retrieved_docs)
response = generate_response(query, compressed_context)

# Display Results
print("\n--- Retrieved Documents (Before Compression) ---")
for doc in retrieved_docs:
    print(f"Content: {doc.page_content}\n")

print("\n--- Compressed Context ---")
print(compressed_context)

print("\n--- LLM Response ---")
print(response)
