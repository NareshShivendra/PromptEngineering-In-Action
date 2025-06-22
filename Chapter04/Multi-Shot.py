import os
import openai

# Configure your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Prepare your few‐shot examples
examples = [
    {
        "Input": "I love this movie! It's fantastic.",
        "Sentiment": "Positive"
    },
    {
        "Input": "This is the worst experience I have ever had.",
        "Sentiment": "Negative"
    },
    {
        "Input": "The movie was okay, not the best but not the worst",
        "Sentiment": "Neutral"
    },
]

# Build the multi‐shot prompt
few_shot_prompt = ""
for ex in examples:
    few_shot_prompt += f"Prompt: {ex['Input']}\nSentiment: {ex['Sentiment']}\n\n"

# Add your new input
user_input = (
    "Despite the occasional setbacks and challenges, the team’s progress has been remarkable. "
    "There were moments of frustration, but the overall journey has been incredibly rewarding. "
    "It’s not always easy, but the sense of accomplishment makes it all worthwhile."
)
few_shot_prompt += f"Prompt: {user_input}\nSentiment:"

# Call the Chat API directly
response = openai.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a sentiment analysis assistant. "
                "Classify each user Prompt as exactly one of: Positive, Negative, or Neutral."
            )
        },
        {"role": "user", "content": few_shot_prompt},
    ],
    max_tokens=10,   # enough for a single label
)

# Extract and print the label
sentiment_label = response.choices[0].message.content.strip()
print("Sentiment:", sentiment_label)
