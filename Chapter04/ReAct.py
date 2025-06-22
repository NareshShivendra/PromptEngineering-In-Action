import os
import openai

# Configure your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the ReAct prompt as a single user message
react_prompt = """
You are a helpful assistant using the ReAct framework.  
For each question, alternate between Thought, Action, and Observation steps, and finish with a final answer.

Question 1: How deep is the Mariana Trench?
Thought: I need to search for the Mariana Trench, then identify the deepest point recorded in this trench.  
Action: Search[Mariana Trench depth]  
Observation: The Mariana Trench is the deepest part of the world's oceans.  
Thought: I need more specific data about the Challenger Deep.  
Action: Lookup[Challenger Deep depth]  
Observation: The Challenger Deep reaches approximately 36,070 feet (10,994 meters).  
Thought: I have the specific depth.  
Action: Finish[36,070 feet (10,994 meters)]

Question 2: What is the temperature range for the area that the Great Barrier Reef extends into?
"""

# Build the message list
messages = [
    {"role": "system", "content": "You are a reasoning assistant. Think step by step using ReAct."},
    {"role": "user",   "content": react_prompt},
]

# Call the chat completion endpoint
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.0,     # deterministic reasoning
    max_tokens=500,      # adjust as needed
)

# Print out the full ReAct trace and final answer
print(response.choices[0].message.content)
