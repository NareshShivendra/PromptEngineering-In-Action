from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

# Define sample documents with metadata				
documents = [
    Document(page_content="Quantum computing breakthrough achieved in 2024 at MIT.", metadata={"category": "science", "date": "2024", "organization": "MIT"}),
    Document(page_content="OpenAI releases new AI model improving natural language understanding.", metadata={"category": "technology", "date": "2023", "organization": "OpenAI"}),
    Document(page_content="Stock market surges due to new economic policies announced by the US government.", metadata={"category": "finance", "date": "2024", "organization": "US Government"}),
    Document(page_content="New healthcare initiative launched by WHO to combat global pandemics.", metadata={"category": "healthcare", "date": "2022", "organization": "WHO"})
]
# Initialize OpenAI embeddings					
embedding_model = OpenAIEmbeddings()

# Store documents in FAISS for retrieval
vector_db = FAISS.from_documents(documents, embedding_model)

# Initialize LLM (GPT-4o)						
chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")

# Define prompt for extracting metadata from user query
metadata_extraction_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    Extract relevant metadata filters from the following user query.
    Identify key entities such as date, organization, category, or any relevant filters.
    Query: {query}
    Output in JSON format:
    """)

# Create LangChain LLM chain					
metadata_extraction_chain = LLMChain(llm=chat_model, prompt=metadata_extraction_prompt)

# Function to process query, extract metadata, and retrieve relevant documents
def self_query_retrieval(query, top_n=2):
    # Extract metadata filters from query
    metadata_output = metadata_extraction_chain.run(query=query)
    # Convert string output to JSON
    try:
        metadata_filters = json.loads(metadata_output)
    except json.JSONDecodeError:
        metadata_filters = {}
    # Retrieve documents with metadata filtering
    retriever = vector_db.as_retriever(search_kwargs={"filter": metadata_filters})
    retrieved_docs = retriever.get_relevant_documents(query)
    return retrieved_docs, metadata_filters

# Example Query						
query = "Tell me about quantum computing research at MIT in 2024"
retrieved_docs, extracted_metadata = self_query_retrieval(query)
# Print Results

print("\n--- Extracted Metadata Filters ---")
print(extracted_metadata)
print("\n--- Retrieved Documents ---")
for doc in retrieved_docs:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}")
