import os
import openai

#1. CONFIG 
# Make sure your OPENAI_API_KEY is set in the environment
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"
TEMPERATURE = 0.0  # deterministic chains

#2. SAMPLE QUESTIONS (COLLATION)
sample_questions = [
    "A chef can peel a potato in 1 minute. Three thousand potatoes need to be peeled within 2 hours. How many people does the chef need to peel all the potatoes?",
    "$300 is divided amongst John, Elena, and James so that John gets seven more than Elena, and Elena receives eight more than James. What is the ratio of their shares?",
    "Find the principal if compound interest is charged on the principal at the rate of 16.5% per year for two years and the sum becomes $200.",
    "250 students took a mock test. 54% of the boys and 73% of the girls cleared the cutoff. If the total percentage of all students clearing the cutoff is 83%, how many girls appeared for the mock test?"
]

#3. AUTO-CoT GENERATION FUNCTION
def generate_cot_chain(question: str) -> str:
    """Generate a single chain-of-thought for `question`."""
    prompt = f"Q: {question}\nA: Let's think step by step."
    resp = openai.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a reasoning assistant. Show your full chain of thought."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

#4. GENERATE CoT FOR ALL SAMPLES (CONTEXT GENERATION)
print("Generating chains for sample questions…")
sample_chains = [generate_cot_chain(q) for q in sample_questions]
for i, chain in enumerate(sample_chains, 1):
    print(f"\n--- Chain {i} ---\n{chain}")

#5. CONTEXTUAL PROMPTING FOR A NEW QUESTION
def solve_with_autocot(new_question: str) -> str:
    """Prompt the model on `new_question`, seeding it with the sample CoTs."""
    # Build a single user prompt that includes all sample Q&A chains
    context = ""
    for q, chain in zip(sample_questions, sample_chains):
        context += f"Q: {q}\nA: Let's think step by step.\n{chain}\n\n"
    context += f"Q: {new_question}\nA: Let's think step by step."
    
    resp = openai.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a reasoning assistant. Use the examples above to guide your chain of thought."},
            {"role": "user",   "content": context},
        ],
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()

#6. EXAMPLE USAGE
if __name__ == "__main__":
    new_q = ("$1550 is lent out in two parts, one at 8% and the other at 6%. "
             "If the total annual income is $106, find the money lent out at each rate.")
    print("\n\n===== SOLVING NEW QUESTION WITH AUTO-CoT =====\n")
    print(solve_with_autocot(new_q))
