
import os
import time
import requests

from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class GroqProvider:

    provider_name = "Groq"

    def __init__(self, api_key, model):

        self.api_key = api_key
        self.model = model

        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=(5, 60)
        )

        response.raise_for_status()

        data = response.json()
        usage = data.get("usage") or {}

        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0
        )


class LLMClient:

    def __init__(self, provider, usage_logger):

        self.provider = provider
        self.usage_logger = usage_logger

    def generate(self, prompt):

        start = time.perf_counter()

        for attempt in range(3):

            try:

                result = self.provider.generate(prompt)

                latency = round(
                    (time.perf_counter() - start) * 1000, 2
                )

                input_rate = float(
                    os.getenv("GROQ_INPUT_PRICE_PER_MILLION", "0")
                )

                output_rate = float(
                    os.getenv("GROQ_OUTPUT_PRICE_PER_MILLION", "0")
                )

                cost = (
                    result.input_tokens * input_rate
                    + result.output_tokens * output_rate
                ) / 1_000_000

                self.usage_logger.log_call({

                    "provider": self.provider.provider_name,
                    "model": self.provider.model,

                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,

                    "total_tokens": (
                        result.input_tokens + result.output_tokens
                    ),

                    "latency_ms": latency,
                    "estimated_cost_usd": round(cost, 8),

                    "cost_type": "estimated_hosted_api_cost",
                    "status": "success",
                    "attempts": attempt + 1

                })

                print("\n----- MONITORING -----")
                print(f"Latency: {latency} ms")
                print(f"Input tokens: {result.input_tokens}")
                print(f"Output tokens: {result.output_tokens}")
                print(f"Total tokens: {result.input_tokens + result.output_tokens}")
                print(f"Estimated cost: ${cost} USD")

                return result.text

            except Exception as error:

                status = getattr(
                    getattr(error, "response", None),
                    "status_code",
                    None
                )

                retryable = (
                    isinstance(
                        error,
                        (
                            requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError
                        )
                    )
                    or status in (408, 429, 500, 502, 503, 504)
                )

                if not retryable or attempt == 2:

                    latency = round(
                        (time.perf_counter() - start) * 1000, 2
                    )

                    self.usage_logger.log_call({

                        "provider": self.provider.provider_name,
                        "model": self.provider.model,

                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,

                        "latency_ms": latency,
                        "estimated_cost_usd": None,

                        "status": "failed",
                        "attempts": attempt + 1,
                        "error": str(error)

                    })

                    raise

                delay = 2 ** attempt

                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
