
import os
from transformers import AutoTokenizer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pretrained tokenizers
TOKENIZERS = {
    "BPE - GPT-2": "gpt2",
    "WordPiece - Multilingual BERT": "bert-base-multilingual-cased",
    "SentencePiece - XLM-R": "xlm-roberta-base"
}

# Input sentences
TEXTS = {
    "English": "Artificial intelligence is changing the world.",
    "Telugu": "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
    "Hindi": "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।"
}


# Load tokenizers
def load_tokenizers():

    loaded = {}

    for name, model in TOKENIZERS.items():

        print(f"\nLoading {name}...")

        loaded[name] = AutoTokenizer.from_pretrained(model)

    return loaded


# Compare tokenizers
def compare_tokenizers(tokenizers):

    for language, text in TEXTS.items():

        print("\n" + "=" * 70)
        print("LANGUAGE:", language)
        print("INPUT TEXT:", text)
        print("=" * 70)

        for name, tokenizer in tokenizers.items():

            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )

            token_ids = encoded["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            print("\n", name)
            print("Tokens:", tokens)
            print("Token IDs:", token_ids)
            print("Token count:", len(token_ids))


# Groq API integration
def groq_demo():

    print("\n" + "=" * 70)
    print("GROQ API DEMONSTRATION")
    print("=" * 70)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("API key not found. Check your .env file.")
        return

    try:

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "Explain BPE, WordPiece and SentencePiece simply."
                }
            ],
            max_completion_tokens=200
        )

        print("\nGroq Response:")
        print(response.choices[0].message.content)

        usage = response.usage

        if usage:
            print("\nToken Usage:")
            print("Prompt tokens:", usage.prompt_tokens)
            print("Completion tokens:", usage.completion_tokens)
            print("Total tokens:", usage.total_tokens)

    except Exception as error:
        print("Groq API Error:", error)


# Main function
def main():

    print("\nWEEK 4 TOKENIZER LAB")
    print("BPE | WordPiece | SentencePiece")

    tokenizers = load_tokenizers()

    compare_tokenizers(tokenizers)

    groq_demo()


if __name__ == "__main__":
    main()
