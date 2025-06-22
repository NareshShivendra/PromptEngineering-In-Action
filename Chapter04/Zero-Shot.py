from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import ShellTool

# Create the shell tool
shell_tool = ShellTool()

# Instantiate the chat LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key="YOUR_KEY_HERE"
)

# Build the agent
agent = initialize_agent(
    tools=[shell_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run your query
agent.run("Name the tallest mountain in the world and in North America")
