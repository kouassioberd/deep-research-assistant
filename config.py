import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider
USE_HUGGINGFACE = False          # Set False to use OpenRouter
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "openai/gpt-oss-120b:fastest"   # free

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Reflection
MAX_REFLECTION_ITERATIONS = 3
QUALITY_THRESHOLD = 8.0          # out of 10
PLATEAU_WINDOW = 2

# Parallel
MAX_SUB_QUESTIONS = 5
MIN_SUB_QUESTIONS = 3
BEST_OF_N = 3
TIMEOUT_PER_CALL = 30

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/research.log"