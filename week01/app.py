
import os
from dotenv import load_dotenv
from groq import Groq

# Load variables from the .env file
load_dotenv()

# Read the API key securely from the environment
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check that .env exists "
        "and contains GROQ_API_KEY=your_key."
    )

# Create the Groq client
client = Groq(api_key=api_key)

# Send a prompt to the model
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "What is Artificial Intelligence? Explain in simple words."
        }
    ]
)

# Display the generated answer
print("\nLLM Response:")
print(response.choices[0].message.content)
