import requests
import json
import random
import time

OLLAMA_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
MODEL_NAME = "gemma3:1b"  # Use your downloaded model name

topics = [
    "Cloud Computing", "DevOps", "Cybersecurity", "Mobile App Development", 
    "Embedded Systems", "Big Data", "Quantum Computing", "Computer Vision", 
    "Natural Language Processing", "Robotics", "Game Development", "IoT", 
    "AR/VR", "Software Testing", "Microservices", "UI/UX Design"
]


def generate_from_model(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        print(f"[!] Error: {response.status_code} {response.text}")
        return None

dataset = []

for i in range(20):  # 20 batches × 5 prompts = 100 samples
    topic = random.choice(topics)
    print(f"[+] Generating batch {i+1}/20 for topic: {topic}")

    for j in range(5):
        instruction = f"Explain something about {topic}."
        response_text = generate_from_model(instruction)
        if response_text:
            dataset.append({
                "instruction": instruction,
                "response": response_text
            })
        time.sleep(1)

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Saved {len(dataset)} instruction-response pairs to dataset.jsonl")
