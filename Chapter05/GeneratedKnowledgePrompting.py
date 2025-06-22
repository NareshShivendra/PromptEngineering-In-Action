import os
import openai
from collections import Counter

# 1. Configure your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# 2. Patient data and knowledge prompts
patient = {
    "age": 55,
    "gender": "male",
    "history": "high blood pressure",
    "symptoms": ["shortness of breath", "chest pain", "fatigue", "irregular heartbeat"]
}

knowledge_1 = """
General medical knowledge related to the symptoms: in patients with shortness of breath,
chest pain, and hypertension, consider angina, heart failure, or arrhythmia.
"""
knowledge_2 = """
For patients over 50 with these findings, also evaluate risks of coronary artery
disease and hypertensive heart disease.
"""

# 3. Build the combined prompt
final_prompt = f"""
A {patient['age']}-year-old {patient['gender']} with a history of {patient['history']} is
experiencing {', '.join(patient['symptoms'])}.

{knowledge_1}
{knowledge_2}

Question: Considering cardiovascular conditions like angina, arrhythmia, heart failure,
coronary artery disease, and hypertensive heart disease, what are the most likely
diagnoses? Please list each with a confidence score (e.g., "Angina – Confidence: 0.85").
"""

# 4. Send to the Chat Completion endpoint
resp = openai.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    messages=[
        {"role": "system", "content": "You are a knowledgeable medical assistant."},
        {"role": "user",   "content": final_prompt}
    ]
)

raw = resp.choices[0].message.content.strip()
print("Model response:\n", raw)

# 5. Parse diagnoses and scores
def parse_diagnoses(text):
    diagnoses = []
    for line in text.splitlines():
        if "– Confidence:" in line:
            diag, score = line.split("– Confidence:")
            diagnoses.append({
                "diagnosis": diag.strip(),
                "confidence": float(score.strip())
            })
    return diagnoses

predictions = parse_diagnoses(raw)

# 6. Find the highest‐confidence diagnosis
best = max(predictions, key=lambda x: x["confidence"])
print("\nParsed predictions:", predictions)
print(f"\nHighest‐confidence diagnosis: {best['diagnosis']} ({best['confidence']:.2f})")
