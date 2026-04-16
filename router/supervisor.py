import json
import logging
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class Router:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        with open("prompts/router.md", "r") as f:
            self.system_prompt = f.read()

    async def route(self, question: str) -> dict:
        prompt = f"{self.system_prompt}\n\nUser question: {question}"
        response = await self.llm.generate(prompt, temperature=0.2)
        try:
            # Extract JSON from response (may be wrapped)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1:
                raise ValueError("No JSON found")
            data = json.loads(response[start:end])
            category = data.get("category", "Fallback/Out-of-scope")
            confidence = data.get("confidence", 0.5)
            reasoning = data.get("reasoning", "")
            return {"category": category, "confidence": confidence, "reasoning": reasoning}
        except Exception as e:
            logger.error(f"Router parse error: {e}\nResponse: {response}")
            return {"category": "Fallback/Out-of-scope", "confidence": 0.0, "reasoning": "Parsing failed"}