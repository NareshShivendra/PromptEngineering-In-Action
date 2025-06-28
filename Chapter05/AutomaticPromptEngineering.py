import os
import openai

# 0. Configure your API key
os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_API_KEY>"
openai.api_key = os.environ["OPENAI_API_KEY"]

# 1. Your candidate prompts
prompt_templates = [
    "Summarize Quantum Computing in simple terms.",
    "Can you explain the key points about Quantum Computing for beginners?",
    "What are the main ideas behind Quantum Computing? Can you summarize it in an easy-to-understand language?",
    "Provide a simplified overview of Quantum Computing, highlighting the essential points.",
    "Write a summary of Quantum Computing suitable for someone without a technical background."
]

# 2. Scoring setup
criteria_weights = {
    "clarity":    1.0,
    "accuracy":   1.5,
    "conciseness":1.0,
    "relevance":  1.5
}
ideal_summary_keywords = [
    "quantum", "computing", "qubits", "superposition", "entanglement", "faster"
]

def evaluate_summary(text: str) -> float:
    """Return a weighted score [0–1] for clarity, accuracy, conciseness, relevance."""
    words = text.split()
    lower = text.lower()
    # Accuracy & relevance: fraction of ideal keywords mentioned
    hit_rate = sum(kw in lower for kw in ideal_summary_keywords) / len(ideal_summary_keywords)
    # Conciseness: ideal length ~<150 words
    length_score = max(0.0, 1 - (len(words) / 150))
    # Clarity: simple proxy → shorter average sentence length
    sentences = text.split(".")
    avg_sent_len = sum(len(s.split()) for s in sentences if s) / max(1, len(sentences))
    clarity_score = max(0.0, 1 - (avg_sent_len / 25))
    # Weighted aggregate
    total_w = sum(criteria_weights.values())
    return (
        clarity_score   * criteria_weights["clarity"] +
        hit_rate        * criteria_weights["accuracy"] +
        length_score    * criteria_weights["conciseness"] +
        hit_rate        * criteria_weights["relevance"]
    ) / total_w

# 3. Try each prompt, generate, score
results = []
for prompt in prompt_templates:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": "You are an expert explainer. Summarize complex topics clearly and concisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    summary = resp.choices[0].message.content.strip()
    score = evaluate_summary(summary)
    results.append({
        "prompt":  prompt,
        "summary": summary,
        "score":   score
    })

# 4. Pick the best
best = max(results, key=lambda r: r["score"])

# 5. Report
print("=== All Summaries & Scores ===")
for i,r in enumerate(results,1):
    print(f"{i}. Score: {r['score']:.3f}\n   Prompt: {r['prompt']}\n   Summary: {r['summary']}\n")

print("=== Best Summary ===")
print(f"Prompt: {best['prompt']}\nScore: {best['score']:.3f}\nSummary: {best['summary']}")
