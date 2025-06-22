import os
import openai
from collections import Counter

# Configure your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the system and user messages
system_message = {
    "role": "system",
    "content": "You are a helpful assistant. Let's think step by step."
}
user_prompt = """Q: A farmer has 20 apple trees. Each tree produces 10 apples per day.
The farmer eats 4 apples himself and uses 6 apples to make pies every day.
He sells the rest for $0.50 per apple.
How much money does he make every day?
A: Let's think step by step."""

# Generate multiple reasoning paths
responses = []
for _ in range(5):
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[system_message, {"role": "user", "content": user_prompt}],
        temperature=0.7,   # sampling for diversity
        n=1
    )
    text = resp.choices[0].message.content.strip()
    responses.append(text)

# Extract the final numeric answer from each path
#    (assumes the last line contains the answer, e.g. "Therefore, he makes $92.00 per day.")
final_answers = [r.splitlines()[-1] for r in responses]

# Find the most common answer
most_common = Counter(final_answers).most_common(1)[0][0]

# Display everything
print("All reasoning paths:\n")
for i, r in enumerate(responses, 1):
    print(f"--- Response {i} ---\n{r}\n")
print(f"Most consistent answer: {most_common}")
