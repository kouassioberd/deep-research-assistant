import asyncio
import json
import difflib
import logging
from typing import Dict, List
from utils.llm_client import LLMClient
from config import MAX_REFLECTION_ITERATIONS, QUALITY_THRESHOLD, PLATEAU_WINDOW

logger = logging.getLogger(__name__)

class ProducerCriticLoop:
    def __init__(self, producer_client: LLMClient, critic_client: LLMClient = None):
        self.producer = producer_client
        self.critic = critic_client if critic_client else producer_client  # use same if no separate
        with open("prompts/producer.md", "r") as f:
            self.producer_prompt = f.read()
        with open("prompts/critic.md", "r") as f:
            self.critic_prompt = f.read()

    async def critique(self, brief: str, original_question: str, domain: str) -> Dict:
        prompt = f"{self.critic_prompt}\n\nDomain: {domain}\nOriginal question: {original_question}\nBrief: {brief}"
        resp = await self.critic.generate(prompt, temperature=0.3)
        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            data = json.loads(resp[start:end])
            # Ensure overall_score is present
            if "overall_score" not in data:
                scores = [data.get(k, 0) for k in ["factual_grounding","completeness","internal_consistency","tone_appropriateness"]]
                data["overall_score"] = sum(scores)/4
            return data
        except Exception as e:
            logger.error(f"Critic parse error: {e}")
            return {"overall_score": 5.0, "improvement_instructions": ["Please provide more specific information."]}

    async def produce(self, draft: str, feedback: str, original_question: str) -> str:
        prompt = f"{self.producer_prompt}\n\nOriginal question: {original_question}\nCurrent draft: {draft}\nCritic feedback: {feedback}\n\nGenerate improved brief:"
        return await self.producer.generate(prompt, temperature=0.6)

    async def run(self, initial_brief: str, original_question: str, domain: str) -> Dict:
        history = []
        current = initial_brief
        prev_scores = []
        for iteration in range(1, MAX_REFLECTION_ITERATIONS+1):
            logger.info(f"Reflection iteration {iteration}")
            critique_result = await self.critique(current, original_question, domain)
            score = critique_result.get("overall_score", 5.0)
            feedback = critique_result.get("improvement_instructions", ["No specific feedback."])
            feedback_text = "\n".join(f"- {f}" for f in feedback)
            history.append({
                "iteration": iteration,
                "score": score,
                "feedback": feedback,
                "brief": current
            })
            print(f"  Iteration {iteration} score: {score:.1f}")
            if score >= QUALITY_THRESHOLD:
                logger.info(f"Quality threshold reached at iteration {iteration}")
                break
            # Plateau detection
            prev_scores.append(score)
            if len(prev_scores) >= PLATEAU_WINDOW+1:
                if all(abs(prev_scores[-i] - prev_scores[-i-1]) < 0.3 for i in range(1, PLATEAU_WINDOW+1)):
                    logger.info("Score plateau detected – stopping")
                    break
            # Generate improved brief
            current = await self.produce(current, feedback_text, original_question)
        # Compute diff between first and last brief
        if len(history) > 1:
            diff = difflib.unified_diff(
                history[0]["brief"].splitlines(),
                history[-1]["brief"].splitlines(),
                fromfile='initial',
                tofile='final',
                lineterm=''
            )
            diff_text = '\n'.join(diff)
        else:
            diff_text = "No changes made."
        return {
            "final_brief": current,
            "history": history,
            "diff": diff_text
        }