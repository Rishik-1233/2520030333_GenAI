# Week 2 - One LLM client, two providers (Groq + Ollama)

| | |
|---|---|
| **Name** | Ramagiri Rishik |
| **Roll No.** | 2520030333 |
| **Subject** | GenAI |
| **Project** | Week02 |

A single provider-agnostic LLM client that talks to **Groq** (hosted API) or
**Ollama** (local model on your machine) behind the same `.chat()` interface.
The CLI lets you pick the provider interactively or from `.env`, and it starts
Ollama for you if it is not already running.

## Files

| File | Purpose |
|---|---|
| `llm-client.py` | The `LLMClient` wrapper, `LLMError`, and the Ollama helpers. |
| `main.py` | Command-line app: prompt in, answer out, provider selection. |
| `test_client.py` | Offline pytest suite (plus opt-in live tests). |
| `requirements.txt` | `groq`, `ollama`, `python-dotenv`, `pytest`. |

> **Note on the filename:** `llm-client.py` keeps the hyphenated name from the
> lab sheet, so `import llm_client` does not work. `main.py` and
> `test_client.py` load it by path with `importlib` (see `load_llm_module()`).

## Providers

- **Groq** - hosted, OpenAI-style API. Needs `GROQ_API_KEY`.
  Default model: `openai/gpt-oss-20b`.
- **Ollama** - local models served at `http://localhost:11434`.
  Default model: `llama3.2`.

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in this folder (it is git-ignored, never commit it):

```
LLM_PROVIDER=groq

GROQ_API_KEY=your_real_groq_key
GROQ_MODEL=openai/gpt-oss-20b

OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
```

For the local provider, install [Ollama](https://ollama.com) and pull the model
once:

```bash
ollama pull llama3.2
```

## Run

```bash
# Interactive: asks which provider should answer
python main.py

# Default prompt ("What is Artificial Intelligence? ...")
python main.py "Explain embeddings in one paragraph."

# Force a provider / model, or add a system prompt
python main.py --provider ollama --model llama3.2 "What is RAG?"
python main.py --provider groq --system "Answer in one sentence." "What is a token?"
```

Piped input (non-interactive) skips the provider question and uses
`LLM_PROVIDER` from `.env`, so the script never blocks in automation.

## What it does
## Execution

### Step 1: Activate Virtual Environment (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run Using Groq API

```powershell
python main.py
```

Select `groq` when prompted and enter your question.

### Step 4: Run Using Ollama (Llama 3.2)

Make sure Ollama is installed and the model is downloaded.

```powershell
ollama pull llama3.2
ollama run llama3.2
```

Exit the interactive Ollama session using `/bye`.

Then run:

```powershell
python main.py
```

Select `ollama` and enter your question.

### Step 5: Run Automated Tests

```powershell
python -m pytest -v
```

### Step 6: Run Live Ollama Integration Test

```powershell
$env:RUN_OLLAMA_TESTS="1"
python -m pytest -v
```

### Expected Test Results

```text
21 passed, 0 skipped
```

Both Groq and Ollama integration tests should pass.

## Tests

```bash
python -m pytest -v          # offline tests, no API key or model needed
```

The offline suite uses fake Groq/Ollama clients, so it covers request shaping,
system-prompt handling, model defaults, error wrapping, provider prompts and the
Ollama auto-start logic without any network calls.

Live tests are opt-in:

```bash
python -m pytest -v -k groq_live

$env:RUN_OLLAMA_TESTS="1"
python -m pytest -v -k ollama_live        
```

## Status

- Total tests: 21
- Passed: 21
- Failed: 0
- Skipped: 0
- Live Groq integration: PASSED
- Live Ollama integration: PASSED

Last verified using:

`python -m pytest -v`
