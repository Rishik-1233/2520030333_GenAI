# Week 3 - LLM Monitoring with MongoDB (Grok + Ollama)

A provider-agnostic LLM client that logs token usage, latency and estimated cost
for every call into MongoDB, with request timeouts and limited retries.

## Providers

- **Grok** (xAI hosted API, OpenAI-compatible, `https://api.x.ai/v1`)
- **Ollama** (local, `llama3.2`)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the values into `.env` (never commit this file):

```
XAI_API_KEY=your-key
GROK_MODEL=grok-4.7
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/...
GROK_INPUT_PRICE_PER_MILLION=2.00
GROK_OUTPUT_PRICE_PER_MILLION=6.00
```

## Run

```bash
python -m unittest -v test_client
python main.py
```

Each call is stored in database `llm_monitoring`, collection `api_usage`
(fields: provider, model, tokens, latency_ms, estimated_cost_usd, status,
attempts, error, timestamp).
