"""Meta's open-weights Llama 3.3 70B on Amazon Bedrock (Converse API), in ELYND's own AWS account.

Bedrock keeps no prompt or answer and shares none with the model's maker (AWS, "Data protection" in the Amazon
Bedrock user guide), so a listener's words stay with ELYND's cloud provider. Nothing here logs what is sent or
answered: an error names its kind only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MODEL = "us.meta.llama3-3-70b-instruct-v1:0"   # a US cross-region profile: the EU regions only serve Llama 3.2 1B/3B
REGION = "us-east-1"
# Bedrock on-demand, Llama 3.3 70B Instruct, USD per million tokens (published price, us-east-1).
PRICE_IN = 0.72
PRICE_OUT = 0.72


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def usd(self) -> float:
        return (self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT) / 1e6


class LlamaError(Exception):
    """The model could not be reached or answered nothing usable. Its message never holds a listener's words."""


class Llama:
    """One Llama 3.3 70B chat at a time. `client`: a boto3 `bedrock-runtime` client (made on first use from the
    environment's AWS credentials when None); tests pass a fake with a `converse(**kwargs)` method."""

    def __init__(self, client: Any = None, model: str = MODEL, region: str = REGION, timeout: float = 60):
        self._client = client
        self.model, self.region, self.timeout = model, region, timeout
        self.spent = 0.0

    def _bedrock(self):
        if self._client is None:
            import boto3
            from botocore.config import Config
            self._client = boto3.client("bedrock-runtime", region_name=self.region,
                                        config=Config(read_timeout=self.timeout, retries={"max_attempts": 2}))
        return self._client

    def chat(self, system: str, user: str, max_tokens: int = 600, temperature: float = 0.6) -> Reply:
        """One question, one answer: the system prompt and a single user message."""
        try:
            response = self._bedrock().converse(
                modelId=self.model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
        except Exception as exc:  # the kind only: an SDK message could quote the request
            raise LlamaError(type(exc).__name__) from None
        content = ((response.get("output") or {}).get("message") or {}).get("content") or []
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        usage = response.get("usage") or {}
        reply = Reply(text, int(usage.get("inputTokens") or 0), int(usage.get("outputTokens") or 0))
        self.spent += reply.usd
        if not text:
            raise LlamaError("empty answer")
        return reply
