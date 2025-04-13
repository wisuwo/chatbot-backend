import requests
import os
from dotenv import load_dotenv
from club_info import CLUB_CONTEXT

load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

def get_response(user_message): 

    full_context = f"""
{CLUB_CONTEXT}
User: {user_message}
Assistant:"""

    payload = {
        "inputs": full_context,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }
    
    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        generated = response.json()[0]["generated_text"]
        return generated.strip()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Sorry, I couldn't process your request at the moment."
