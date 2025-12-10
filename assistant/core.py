# core.py
from typing import List, Tuple, Optional, Any

from vision import capture_frame, detect_objects_and_annotate, detect_objects
from audio_stt import record_audio, transcribe_audio
from llm_client import ask_llm
from tts_client import speak
from config import DEBUG


def analyze_environment() -> Tuple[Optional[Any], List[str]]:
    """
    Capture a single frame and detect objects.
    Returns (annotated_frame, objects).
    """
    frame = capture_frame()
    if frame is None:
        if DEBUG:
            print("[CORE] No frame captured.")
        return None, []

    # Try annotated detection first
    try:
        annotated, objects = detect_objects_and_annotate(frame)
        return annotated, objects
    except Exception:
        # Fallback: plain detection (compatibility)
        objects = detect_objects(frame)
        return frame, objects


def multimodal_interaction(user_text: str = "", use_audio: bool = False) -> Tuple[str, List[str], Optional[Any]]:
    """
    High-level multimodal flow:
    - Optionally record + transcribe audio
    - Capture frame + detect objects
    - Query LLM with objects + user text
    - Speak reply
    Returns: (reply, objects, frame)
    """
    # Step 1: Audio → STT
    if use_audio:
        audio_file = record_audio()
        stt_text = transcribe_audio(audio_file)
        if stt_text:
            user_text = (user_text + " " + stt_text).strip()

    # Step 2: Environment analysis
    frame, objects = analyze_environment()

    # Step 3: LLM reasoning
    if not user_text and not objects:
        reply = "I couldn't detect anything and didn't hear any question."
    else:
        reply = ask_llm(objects, user_text)

    # Step 4: TTS output
    speak(reply)

    return reply, objects, frame
