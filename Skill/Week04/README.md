
# WEEK04 - Tokenizer Lab

## Student Details

- **Name:** Ramagiri Rishik
- **Project:** WEEK04
- **Subject:** GenAI

## Project Title
Comparison of BPE, WordPiece and SentencePiece Tokenizers.

## Aim
To implement and compare three subword tokenization approaches using English, Telugu and Hindi text.

## Technologies Used
- Python
- Hugging Face Transformers
- SentencePiece
- Groq API
- Visual Studio Code

## Tokenizers Implemented

1. BPE - GPT-2
2. WordPiece - Multilingual BERT
3. SentencePiece - XLM-R

## Languages Tested
- English
- Telugu
- Hindi

## Features
- Multilingual text tokenization.
- Display token pieces.
- Display token IDs.
- Calculate token counts.
- Compare three pretrained tokenizers.
- Groq API integration.
- Display Groq prompt and completion token usage.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Configure your `.env` file:

```env
GROQ_API_KEY=your_actual_api_key
```

## Execution

Run the following command:

```bash
python tokenizer_comparison.py
```

## Project Structure

```text
Week04/
│
├── tokenizer_comparison.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

## Learning Outcomes

- Understand BPE tokenization.
- Understand WordPiece tokenization.
- Understand SentencePiece tokenization.
- Compare token counts across languages.
- Integrate Groq API with Python.

---

**Name:** Ramagiri Rishik  
**Project:** WEEK04  
**Subject:** GenAI
