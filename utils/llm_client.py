import httpx
import asyncio
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, use_hf=True):
        self.use_hf = use_hf
        if use_hf:
            self.token = os.getenv("HF_TOKEN")
            self.base_url = "https://router.huggingface.co/v1"
            self.model = os.getenv("HF_MODEL", "openai/gpt-oss-120b:fastest")
        else:
            self.token = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return f"ERROR: {response.status_code}"
        except Exception as e:
            logger.exception(e)
            return f"ERROR: {str(e)}"

    async def generate_batch(self, prompts: List[str], temperature=0.7) -> List[str]:
        tasks = [self.generate(p, temperature) for p in prompts]
        return await asyncio.gather(*tasks)

    async def close(self):
        await self.client.aclose()