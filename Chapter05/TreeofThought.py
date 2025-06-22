import os
import openai

# Configure your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the problem
question = (
    "I have a 12 litre jug and a 5 litre jug and a 3 litre jug. "
    "How can I measure exactly 7 litres of water?"
)

# Build the messages for Tree-of-Thought
system_message = {
    "role": "system",
    "content": (
        "You are a reasoning assistant using the Tree of Thought method. "
        "Generate 3 distinct step-by-step solution paths for the following problem."
    )
}
user_message = {
    "role": "user",
    "content": f"Problem: {question}"
}

# Call the Chat Completion endpoint, requesting 3 separate “thought” branches
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[system_message, user_message],
    temperature=0.0,   # deterministic reasoning
    n=3,               # three distinct reasoning paths
    max_tokens=400
)

# Print each reasoning path
for i, choice in enumerate(response.choices, 1):
    print(f"--- Thought path {i} ---\n{choice.message.content}\n")
