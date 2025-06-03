from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

documents = [						
    Document(page_content="Quantum computing uses qubits instead of classical bits.", metadata={"category": "science"}),
    Document(page_content="AI is revolutionizing healthcare with advanced diagnostics.", metadata={"category": "technology"}),
    Document(page_content="Stock markets are influenced by global economic policies.", metadata={"category": "finance"}),
    Document(page_content="Einstein’s theory of relativity changed our understanding of physics.", metadata={"category": "science"})
]
# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()
		
vector_db = FAISS.from_documents(documents, embedding_model)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})  
query = "How does quantum computing work?"
query_embedding = embedding_model.embed_query(query)  # Convert query to vector
retrieved_docs = retriever.get_relevant_documents(query)

# Print retrieved results
print("\n--- Retrieved Documents ---")
for doc in retrieved_docs:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}")

# Initialize LLM
chat_model = ChatOpenAI(model="gpt-4o", 	openai_api_key="<<YOUR_OPENAI_API_KEY>>")				

# Define a structured prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="Use the context below to answer the query accurately.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
)

# Format retrieved documents into context		
context = "\n".join([doc.page_content for doc in retrieved_docs])

# Generate response using LLM							#C
final_prompt = prompt_template.format(query=query, context=context)
response = chat_model(final_prompt)

# Print LLM-generated response
print("\n--- LLM Response ---")
print(response.content)
