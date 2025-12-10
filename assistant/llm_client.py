# llm_client.py
import requests
from typing import List, Optional, Any

from config import LLM_PROVIDER, OLLAMA_URL, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from logging_utils import get_logger

logger = get_logger()

def build_prompt(prompt: str, history: Optional[List[dict]] = None) -> str:
    """
    Builds a clean chat-style prompt for local LLM
    Kept simple: history is expected to be list of {'role','content'}.
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
    Production-safe Local GPU LLM Inference using Ollama (or fallback to OpenAI).
    Returns assistant text.
    """
    final_prompt = build_prompt(prompt, history)

    if LLM_PROVIDER == "local":
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": final_prompt,
                "stream": False
            }
            res = requests.post(OLLAMA_URL, json=payload, timeout=90)
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

    elif LLM_PROVIDER == "openai":
        # optional fallback: use OpenAI chat completions (if configured)
        try:
            if not OPENAI_API_KEY:
                logger.error("[LLM] OPENAI_API_KEY not set for openai provider.")
                return "OpenAI API key missing."

            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            # Use a Chat-style wrapper if needed; here we send the concatenated prompt as single message
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=float(OPENAI_TEMPERATURE)
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            logger.error(f"[LLM] OpenAI call failed: {e}")
            return "OpenAI LLM failure."

    else:
        logger.error("[LLM] Unknown LLM_PROVIDER configured.")
        return "LLM provider misconfigured."


# ----------------------
# Backward-compatible adapter expected by rest of code:
# ask_llm(objects: List[str], user_text: str, memory: Optional[ConversationMemory])
# ----------------------
def ask_llm(objects: List[str], user_text: str, memory: Optional[Any] = None) -> str:
    """
    Build a human-readable prompt from detected objects and user text,
    optionally include memory history (memory.get_history() -> List[dict]).
    This function matches existing call sites: ask_llm(objects, user_text, memory).
    """
    # Build object summary
    object_summary = ""
    if objects:
        if isinstance(objects, list):
            object_summary = "Detected objects: " + ", ".join(objects) + "."
        else:
            # defensive: if objects already a string
            object_summary = f"Detected objects: {objects}."

    # Compose message prompt
    prompt_parts = []
    if object_summary:
        prompt_parts.append(object_summary)
    if user_text:
        prompt_parts.append("User: " + user_text)

    prompt = "\n".join(prompt_parts).strip() or "Describe what you see."

    # Get history from memory if provided (expect memory.get_history())
    history = None
    try:
        if memory is not None and hasattr(memory, "get_history"):
            history = memory.get_history()
    except Exception:
        history = None

    # Call lower-level LLM function
    return generate_llm_response(prompt, history)


# Provide direct alias for backward compatibility if other code uses previous name:
ask_llm_alias = ask_llm
