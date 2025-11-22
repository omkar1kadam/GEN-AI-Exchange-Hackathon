import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
MODEL_NAME = "gemma3:1b"  # Use your downloaded model name

topics = [
    "Java OOP", "Java Inheritance", "Java Constructors", "Polymorphism in Java", "Encapsulation",
    "Abstraction", "Interfaces", "Exception Handling", "Threads", "File Handling",
    "DSA Arrays", "Linked Lists", "Stacks", "Queues", "Trees", "Graphs",
    "Sorting Algorithms", "Searching Algorithms", "Dynamic Programming", "Greedy Algorithms",
    "ESP32 Basics", "WiFi Connection", "MQ135 Sensor", "DHT11 Sensor", "Rain Sensor",
    "Soil Moisture", "GPS Module", "I2C Communication", "ADC in ESP32", "PWM control",
    "Blockchain Basics", "Smart Contracts", "Polygon Network", "Web3.py Integration", "Minting Tokens",
    "Machine Learning", "Neural Networks", "CNN", "LSTM", "Model Evaluation",
    "Python Flask", "APIs", "JSON Handling", "React Frontend", "Node.js Backend",
    "Motivational Coding Quotes", "Debugging", "Error Handling", "Version Control (Git)", "Linux Commands"
]

def generate_chat_pairs(topic):
    prompt = f"Generate 5 casual chatbot-style Q&A pairs between two bros about {topic}. " \
             f"Each response should be technical, beginner-friendly, and use a chill tone like 'bro'. " \
             f"Output in JSONL format with keys: instruction, response."

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )

    data = response.json()
    return data.get("response", "").strip()

for i, topic in enumerate(topics, 1):
    print(f"🔥 Generating data for topic {i}/{len(topics)} → {topic}")
    content = generate_chat_pairs(topic)
    filename = f"dataset_{topic.replace(' ', '_').lower()}.jsonl"

    with open(filename, "w") as f:
        f.write(content)
    print(f"✅ Saved: {filename}")
    time.sleep(2)  # prevent rate limit or overloading

print("\n🚀 Done bro! All 50 topics generated.")
