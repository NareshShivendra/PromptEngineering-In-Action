from langchain import PromptTemplate, LLMChain
from langchain_openai import ChatOpenAI
from typing import List, Dict
import os

# setting the openai api key as environment variable
os.environ['OPENAI_API_KEY']= "<OpenAI API Key>>" # replace with your openai api key
# Initialize the OpenAI LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.environ['OPENAI_API_KEY'])

# Step 1: Basic Patient Input Prompt
patient_input = {
    "age": 55,
    "gender": "male",
    "history": "high blood pressure",
    "symptoms": ["shortness of breath", "chest pain", "fatigue", "irregular heartbeat"]
}

# Step 2: Generated Knowledge Prompts
knowledge_prompt_1 = """
General medical knowledge related to the symptoms listed for the patient. In patients experiencing shortness of breath, chest pain, and a history of high blood pressure, possible cardiovascular conditions include angina, heart failure, or arrhythmia.
"""
knowledge_prompt_2 = """
General knowledge for patients over 50 years with these symptoms should also be evaluated for risks related to coronary artery disease and hypertensive heart disease.
"""
# Combine all data into the final prompt
final_prompt_template = """
A {age}-year-old {gender} with a history of {history} is experiencing {symptoms}.
Considering cardiovascular conditions like angina, arrhythmia, heart failure, coronary artery disease, or hypertensive heart disease, what are the most likely diagnoses? 
Please provide multiple diagnoses with confidence scores for each.
"""
# Prepare the prompt template
prompt_template = PromptTemplate(input_variables=["age", "gender", "history", "symptoms"], template=final_prompt_template)
# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Step 3: Run the prompt through the LLM
prompt_input = {
    "age": patient_input["age"],
    "gender": patient_input["gender"],
    "history": patient_input["history"],
    "symptoms": ", ".join(patient_input["symptoms"])
}
# Get the diagnosis predictions with confidence scores
response = llm_chain.run(prompt_input)
#print(response)

#Step 4: Parse the model's response to extract diagnoses and confidence scores
def parse_diagnoses(response_text: str):
    diagnoses = []
    lines = response_text.strip().split('\n')
    for line in lines:
        parts = line.split(" - Confidence Score: ")
        print(parts)
        if len(parts) == 2:
            diagnosis = parts[0].strip()
            confidence = parts[1]
            diagnoses.append({"diagnosis": diagnosis, "confidence": confidence})
    return diagnoses

# Extract diagnoses from the LLM's response
diagnoses_with_confidence = parse_diagnoses(response)
print(diagnoses_with_confidence)

# Step 5: Select the highest confidence diagnosis
def get_highest_confidence_diagnosis(predictions):
    return max(predictions, key=lambda x: x['confidence'])
best_diagnosis = get_highest_confidence_diagnosis(diagnoses_with_confidence)

# Display the final result
print(f"Highest confidence diagnosis: {best_diagnosis['diagnosis']} with confidence {best_diagnosis['confidence']}")
