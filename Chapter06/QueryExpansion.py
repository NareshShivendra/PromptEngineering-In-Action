from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM (GPT-4o)					
chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")

# Define prompt templates for different query expansion techniques	
synonym_expansion_prompt = PromptTemplate(
    input_variables=["query"],
    template="Expand the following query by adding common synonyms and alternative terms to improve search recall:\nQuery: {query}\nExpanded Query:"
)
conceptual_expansion_prompt = PromptTemplate(
    input_variables=["query"],
    template="Enhance the following query by adding domain-specific terminology and related concepts for better retrieval:\nQuery: {query}\nExpanded Query:"
)

# Create LangChain LLM chains						
synonym_expansion_chain = LLMChain(llm=chat_model, prompt=synonym_expansion_prompt)
conceptual_expansion_chain = LLMChain(llm=chat_model, prompt=conceptual_expansion_prompt)

# Sample Queries								
query1 = "What are common diseases?"
query2 = "How does climate change affect the planet?"

# Perform query expansion
expanded_query1 = synonym_expansion_chain.run(query=query1)
expanded_query2 = conceptual_expansion_chain.run(query=query2)

# Print Results
print("\n--- Synonym Expanded Query ---")
print(f"Original: {query1}")
print(f"Expanded: {expanded_query1}")
print("\n--- Conceptually Expanded Query ---")
print(f"Original: {query2}")
print(f"Expanded: {expanded_query2}")
