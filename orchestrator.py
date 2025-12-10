from typing import List, Tuple, Optional

from vision import capture_frame, detect_objects_and_annotate
from audio_stt import record_audio, transcribe_audio
from llm_client import ask_llm
from tts_client import speak
from memory import get_global_memory
from logging_utils import get_logger

logger = get_logger()


def analyze_environment() -> Tuple[Optional[any], List[str]]:
    """
    Capture current frame and detect objects.
    Returns (annotated_frame, [object_names]).
    """
    frame = capture_frame()
    if frame is None:
        return None, []
    annotated, objects = detect_objects_and_annotate(frame)
    return annotated, objects


def multimodal_turn(user_text: str, use_audio: bool = False) -> Tuple[str, List[str], Optional[any]]:
    """
    One full turn:
    - Optionally record + transcribe audio and append to user_text
    - Capture frame + detect objects
    - LLM reasoning with memory and objects
    - Update memory
    - TTS output
    Returns: (reply, objects, frame)
    """
    memory = get_global_memory()

    if use_audio:
        audio_file = record_audio()
        stt_text = transcribe_audio(audio_file)
        if stt_text:
            user_text = (user_text + " " + stt_text).strip()
            logger.info(f"[CORE] Combined user text with STT: {user_text}")

    frame, objects = analyze_environment()

    if not user_text and not objects:
        reply = "I couldn't detect anything and you didn't say anything."
    else:
        # Store user turn in memory
        if user_text:
            memory.add_turn("user", user_text)
        reply = ask_llm(objects, user_text or "Describe what you see.", memory)
        # Store assistant reply
        memory.add_turn("assistant", reply)

    speak(reply)
    return reply, objects, frame
