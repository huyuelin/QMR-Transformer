#!/usr/bin/env python3
"""
Simplified LLM client that uses OpenAI-compatible API endpoints directly.
No dependency on hunyuan_api or qwen_chat_api modules.
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

LOGGER = logging.getLogger("llm_client")


@dataclass
class RequestMetrics:
    success: bool
    latency_s: float
    attempt: int
    status_code: Optional[int]
    error: Optional[str]
    provider: str = ""


# Provider configs (OpenAI-compatible endpoints)
PROVIDERS = [
    {
        "name": "qwen",
        "api_key": "sk-d5a16bb38a7646039f5715973761dd3f",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus-latest",
    },
    {
        "name": "hunyuan",
        "api_key": "dMaIDnRH4iT7Tc0u8Ua8nBiv2yhNanl9",
        "base_url": "http://hunyuanapi.woa.com",
        "model": "hunyuan-2.0-instruct-20251111",
    },
]


class SimpleLLMClient:
    """Simple OpenAI-compatible LLM client with failover."""

    def __init__(self, max_retries: int = 3, timeout: float = 120.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.total_calls = 0

    def chat(self, messages: List[Dict[str, Any]],
             temperature: float = 0.7,
             max_tokens: int = 4096) -> Tuple[Dict[str, Any], RequestMetrics]:
        """Send chat completion request, with provider failover."""
        self.total_calls += 1
        start = time.monotonic()
        last_error = None

        for provider in PROVIDERS:
            for attempt in range(self.max_retries):
                try:
                    url = f"{provider['base_url']}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {provider['api_key']}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": provider["model"],
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    resp = self.client.post(url, headers=headers, json=payload)

                    if resp.status_code == 200:
                        data = resp.json()
                        metrics = RequestMetrics(
                            success=True,
                            latency_s=time.monotonic() - start,
                            attempt=attempt + 1,
                            status_code=200,
                            error=None,
                            provider=provider["name"],
                        )
                        return data, metrics
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        LOGGER.warning(f"{provider['name']} attempt {attempt+1} failed: {last_error}")
                        time.sleep(1)

                except Exception as e:
                    last_error = str(e)
                    LOGGER.warning(f"{provider['name']} attempt {attempt+1} exception: {last_error}")
                    time.sleep(1)

            LOGGER.info(f"Switching away from {provider['name']}")

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        """Convenience: single prompt → text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp, _ = self.chat(messages, **kwargs)
        return resp["choices"][0]["message"]["content"]


# Compatibility alias
ResilientLLMClient = SimpleLLMClient


if __name__ == "__main__":
    client = SimpleLLMClient()
    resp, metrics = client.chat([{"role": "user", "content": "Say hello in one sentence."}])
    print(f"Provider: {metrics.provider}")
    print(f"Latency: {metrics.latency_s:.2f}s")
    print(f"Response: {resp['choices'][0]['message']['content']}")
