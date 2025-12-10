import requests
from typing import List, Optional
from config import LLM_PROVIDER, OLLAMA_URL, OLLAMA_MODEL
from logging_utils import get_logger

logger = get_logger()


def build_prompt(prompt: str, history: Optional[List[dict]] = None) -> str:
    """
    Builds a clean chat-style prompt for local LLM
    """
    if not history:
        return prompt

    chat = []
    for turn in history:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "")
        chat.append(f"{role}: {content}")

    chat.append(f"USER: {prompt}")
    chat.append("ASSISTANT:")

    return "\n".join(chat)


def generate_llm_response(prompt: str, history: Optional[List[dict]] = None) -> str:
    """
    Production-safe Local GPU LLM Inference using Ollama
    """
    if LLM_PROVIDER != "local":
        logger.error("[LLM] Non-local provider selected but local engine loaded.")
        return "LLM provider misconfigured."

    try:
        final_prompt = build_prompt(prompt, history)

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": False
        }

        res = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=90
        )

        if res.status_code != 200:
            logger.error(f"[LLM] Ollama HTTP {res.status_code}: {res.text}")
            return "Local LLM service error."

        data = res.json()
        response = data.get("response", "").strip()

        if not response:
            return "Local LLM returned empty output."

        return response

    except requests.exceptions.Timeout:
        logger.error("[LLM] Ollama request timeout.")
        return "Local LLM timed out."

    except requests.exceptions.ConnectionError:
        logger.error("[LLM] Ollama is not running.")
        return "Local LLM engine is not running."

    except Exception as e:
        logger.error(f"[LLM] Local LLM crashed: {e}")
        return "Local LLM failure."


# ✅ BACKWARD-COMPATIBLE ALIAS (THIS FIXES YOUR ERROR)
ask_llm = generate_llm_response
