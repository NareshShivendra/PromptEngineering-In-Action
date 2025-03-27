import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain

# setting the openai api key as environment variable
os.environ['OPENAI_API_KEY']= “<<YOUR_OPENAI>>” # replace with your openai api key

question = "I have a 12 litre jug and a 5 litre jug and a 3 litre jug. How can I measure exactly 7 litres of water?"

prompt = PromptTemplate.from_template(question)
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=3, verbose=True)  
chain.invoke({})
