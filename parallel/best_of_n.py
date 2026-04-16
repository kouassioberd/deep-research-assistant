import asyncio
import json
import logging
from typing import List, Dict
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class BestOfNSelector:
    def __init__(self, llm_client: LLMClient, n=3):
        self.llm = llm_client
        self.n = n
        with open("prompts/judge.md", "r") as f:
            self.judge_prompt = f.read()

    async def generate_candidates(self, sub_question: str) -> List[str]:
        """Generate N candidate answers in parallel."""
        prompts = [f"Answer this sub-question concisely and factually: {sub_question}" for _ in range(self.n)]
        return await self.llm.generate_batch(prompts, temperature=0.8)

    async def score_candidate(self, candidate: str, sub_question: str) -> Dict:
        prompt = f"{self.judge_prompt}\n\nSub-question: {sub_question}\nCandidate answer: {candidate}"
        resp = await self.llm.generate(prompt, temperature=0.2)
        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            scores = json.loads(resp[start:end])
            total = scores.get("total_score", (scores.get("correctness",0)+scores.get("specificity",0)+scores.get("hedging",0))/3)
            return {"answer": candidate, "scores": scores, "total": total}
        except:
            return {"answer": candidate, "scores": {}, "total": 0.0}

    async def select_best(self, sub_question: str) -> str:
        candidates = await self.generate_candidates(sub_question)
        scored = await asyncio.gather(*[self.score_candidate(c, sub_question) for c in candidates])
        best = max(scored, key=lambda x: x["total"])
        logger.info(f"Best-of-{self.n} selected score {best['total']:.1f}")
        return best["answer"]