import os
import requests
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
headers = {
    "Authorization": f"Bearer {gemini_api_key}",
    "Content-Type": "application/json",
}

user_text = input("Enter the text you want to translate to English: ")

payload = {
    "model": "gemini-2.0-flash",
    "messages": [
        {
            "role": "user",
            "content": f"Translate the following text to English: '{user_text}'"
        }
    ],
    "temperature": 0.7,
}
resp = requests.post(url, headers=headers, json=payload)
resp.raise_for_status()
data = resp.json()
try:
    print("\n Translated Text:")
    print(data["choices"][0]["message"]["content"])
except Exception:
    import json as _json
    print("\n Full Response (for debugging):")
    print(_json.dumps(data, indent=2))
