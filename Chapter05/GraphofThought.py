import os
import openai

# 1. Configure your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# 2. Define each node’s prompt
crime_scene_prompt = """
You are an investigator analyzing a robbery at the First Republic Bank.
Given the crime scene, evaluate the following:
- What physical evidence can be gathered?
- Are there any witness statements?
- How did the robbers enter and exit the bank?
Respond with a list of potential evidence and actions to take.
"""

cctv_prompt = """
You have access to CCTV footage from both the bank and nearby areas. Focus on the following:
- Check for suspicious individuals entering or leaving the bank two hours before and after the robbery.
- Identify any vehicles used for the getaway.
- Use facial recognition if possible.
Last saved: 5/23/2025
What do you find? Respond with key observations and suspects.
"""

license_plate_prompt = """
You have a list of license plates from vehicles in the vicinity of the bank two hours before and after the robbery.
- Cross-reference these plates with known offenders.
- Identify any stolen or out-of-place vehicles.
Which vehicles seem suspicious? Respond with details of vehicles and their owners.
"""

suspect_profile_prompt = """
You are building a profile of potential suspects based on the robbery.
- Cross-reference with known offenders in the database.
- Use witness descriptions and CCTV footage to match with existing criminal profiles.
What is the profile of the most likely suspect(s)?
"""

# 3. Helper to call the OpenAI Chat API
def call_node(prompt_text: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a skilled investigative assistant."},
            {"role": "user",   "content": prompt_text}
        ],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

# 4. Execute the Graph-of-Thought sequence
def solve_bank_robbery():
    print("=== Crime Scene Analysis ===")
    print(call_node(crime_scene_prompt), "\n")
    
    print("=== CCTV Footage Analysis ===")
    print(call_node(cctv_prompt), "\n")
    
    print("=== License Plate Analysis ===")
    print(call_node(license_plate_prompt), "\n")
    
    print("=== Suspect Profile Generation ===")
    print(call_node(suspect_profile_prompt), "\n")

if __name__ == "__main__":
    solve_bank_robbery()
