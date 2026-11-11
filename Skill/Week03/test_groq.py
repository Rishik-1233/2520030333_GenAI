
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}"
}

try:
    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    print("Status Code:", response.status_code)

    if response.ok:
        models = response.json()["data"]

        print("\nAvailable Groq models:")

        for model in models:
            print(model["id"])

    else:
        print("Groq Error:", response.text)

except requests.exceptions.RequestException as error:
    print("Connection Error:", error)
