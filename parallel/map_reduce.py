import asyncio
import json
import time
import logging
from typing import List
from utils.llm_client import LLMClient
from parallel.best_of_n import BestOfNSelector

logger = logging.getLogger(__name__)

class MapReduce:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.selector = BestOfNSelector(llm_client)
        with open("prompts/map.md", "r") as f:
            self.map_prompt = f.read()

    async def decompose(self, question: str) -> List[str]:
        prompt = f"{self.map_prompt}\n\nQuestion: {question}"
        resp = await self.llm.generate(prompt, temperature=0.4)
        try:
            start = resp.find('[')
            end = resp.rfind(']') + 1
            sub_qs = json.loads(resp[start:end])
            if isinstance(sub_qs, list) and len(sub_qs) >= 3:
                return sub_qs[:5]
        except:
            logger.warning("Decomposition failed, using fallback sub-questions")
        return ["What are the key facts?", "What are the main arguments?", "What is the conclusion?"]

    async def map_step(self, sub_questions: List[str]) -> List[str]:
        """Fan-out: answer each sub-question with best-of-N."""
        tasks = [self.selector.select_best(q) for q in sub_questions]
        return await asyncio.gather(*tasks)

    async def reduce_step(self, original_question: str, sub_answers: List[str]) -> str:
        synthesis_prompt = f"""
        You are a research synthesizer. Based on the following answers to sub-questions, produce a coherent, well-structured research brief that answers the original question.
        Original question: {original_question}
        Sub-answers:
        {chr(10).join(f"- {a}" for a in sub_answers)}
        Write a detailed, factual brief (300-600 words).
        """
        return await self.llm.generate(synthesis_prompt, temperature=0.5)

    async def run(self, question: str, domain: str) -> str:
        start = time.time()
        sub_qs = await self.decompose(question)
        logger.info(f"Decomposed into {len(sub_qs)} sub-questions: {sub_qs}")
        map_start = time.time()
        sub_answers = await self.map_step(sub_qs)
        map_time = time.time() - map_start
        logger.info(f"Map step completed in {map_time:.2f}s")
        brief = await self.reduce_step(question, sub_answers)
        total_time = time.time() - start
        logger.info(f"Total map-reduce time: {total_time:.2f}s")
        return brief