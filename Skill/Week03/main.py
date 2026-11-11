
import os

from dotenv import load_dotenv

from llm_client import GroqProvider, LLMClient
from usage_logger import UsageLogger


load_dotenv(override=True)


def main():

    usage_logger = UsageLogger()

    try:

        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile"
        )

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        provider = GroqProvider(api_key, model)

        client = LLMClient(provider, usage_logger)

        print("\n===== GROQ LLM MONITORING =====")

        prompt = input("\nEnter your question: ").strip()

        if not prompt:
            print("Please enter a question.")
            return

        answer = client.generate(prompt)

        print("\n===== LLM RESPONSE =====")
        print(answer)

    finally:
        usage_logger.close()


if __name__ == "__main__":
    main()
