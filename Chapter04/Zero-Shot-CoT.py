import os
import openai

# Configure your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define your messages
messages = [
    {"role": "system", "content": "You are a helpful assistant. Let's think step by step."},
    {"role": "user", "content": (
        "Mrs. Smith currently has 10 students in her class. "
        "She receives 3 new groups of students over the week. "
        "Each group contains 4 students. "
        "How many students does Mrs. Smith have in her class now?"
    )},
]

# Call the Chat Completion endpoint
response = openai.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    messages=messages
)

# Extract and print the assistant’s answer
answer = response.choices[0].message.content
print(answer)
