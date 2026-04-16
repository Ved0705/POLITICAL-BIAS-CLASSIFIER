import os
import requests
import logging
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

TIMEOUT = 15  # seconds


# ── Core Function ─────────────────────────────────────────────────────────────
def translate_to_english(text: str, temperature: float = 0.3) -> str:
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            {
                "role": "user",
                "content": f"Translate the following text to English:\n{text}"
            }
        ],
        "temperature": temperature,
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return "Error: Request timed out"

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return "Error: API request failed"

    except (KeyError, IndexError):
        logger.error("Unexpected response format")
        return f"Error: Unexpected API response → {data}"


# ── CLI Interface ─────────────────────────────────────────────────────────────
def main():
    user_text = input("Enter text to translate to English: ").strip()

    if not user_text:
        print("⚠️ Please enter some text.")
        return

    print("\n Translating...")
    result = translate_to_english(user_text)

    print("\n Translated Text:")
    print(result)


if __name__ == "__main__":
    main()
