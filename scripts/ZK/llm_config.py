import os

# Single source of truth for LLM access settings.
# You can edit values here once, or override via env vars.
LLM_API_KEY = os.getenv("LARES_LLM_API_KEY", "sk-M0DLvTc54yt7gqpbCYbUXSIBjT9pwR4mBvUfjZGt4MMRIuFt")
LLM_BASE_URL = os.getenv("LARES_LLM_BASE_URL", "https://api.vectorengine.ai/v1")
LLM_MODEL_NAME = os.getenv("LARES_LLM_MODEL_NAME", "gpt-5.4-mini")


def get_llm_settings():
    return {
        "api_key": LLM_API_KEY,
        "base_url": LLM_BASE_URL,
        "model_name": LLM_MODEL_NAME,
    }
