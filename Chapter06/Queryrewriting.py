from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM (GPT-4o)
chat_model = ChatOpenAI(model="gpt-4o", openai_api_key="<<YOUR_OPENAI_API_KEY>>")

# Define prompt templates for different query rewriting techniques			
paraphrasing_prompt = PromptTemplate(
    input_variables=["query"],
    template="Reword the following query while keeping its meaning:\nQuery: {query}\nReworded Query:"
)
synonym_substitution_prompt = PromptTemplate(
    input_variables=["query"],
    template="Expand the following query by replacing key terms with synonyms to improve search accuracy:\nQuery: {query}\nExpanded Query:"
)
query_decomposition_prompt = PromptTemplate(
    input_variables=["query"],
    template="Break down the following complex query into smaller, specific sub-queries:\nQuery: {query}\nDecomposed Queries:"
)

# Create LangChain LLM chains						
paraphrasing_chain = LLMChain(llm=chat_model, prompt=paraphrasing_prompt)
synonym_substitution_chain = LLMChain(llm=chat_model, prompt=synonym_substitution_prompt)
query_decomposition_chain = LLMChain(llm=chat_model, prompt=query_decomposition_prompt)

# Sample Queries								
query1 = "How does exercise impact mental health?"
query2 = "Job opportunities in the tech industry?"
query3 = "What are the environmental, economic, and social impacts of deforestation?"

# Perform query rewriting							
rewritten_query1 = paraphrasing_chain.run(query=query1)
rewritten_query2 = synonym_substitution_chain.run(query=query2)
rewritten_query3 = query_decomposition_chain.run(query=query3)

# Print Results
print("\n--- Paraphrased Query ---")
print(f"Original: {query1}")
print(f"Rewritten: {rewritten_query1}")

print("\n--- Synonym Substituted Query ---")
print(f"Original: {query2}")
print(f"Rewritten: {rewritten_query2}")

print("\n--- Decomposed Queries ---")
print(f"Original: {query3}")
print(f"Decomposed:\n{rewritten_query3}")
