from typing import List, Tuple

from vision import capture_frame, detect_objects
from audio_stt import record_audio, transcribe_audio
from llm_client import ask_llm
from tts_client import speak
from config import DEBUG


def analyze_environment() -> Tuple[any, List[str]]:
    """
    Capture a single frame and detect objects.
    Returns (frame, objects).
    """
    frame = capture_frame()
    if frame is None:
        if DEBUG:
            print("[CORE] No frame captured.")
        return None, []
    objects = detect_objects(frame)
    return frame, objects


def multimodal_interaction(user_text: str = "", use_audio: bool = False) -> Tuple[str, List[str]]:
    """
    High-level function:
    - Optionally record audio + transcribe
    - Capture frame + detect objects
    - Send objects + text to LLM
    - Speak and return response
    """
    if use_audio:
        audio_file = record_audio()
        stt_text = transcribe_audio(audio_file)
        if stt_text:
            user_text = (user_text + " " + stt_text).strip()

    frame, objects = analyze_environment()

    if not user_text and not objects:
        reply = "I couldn't detect anything and didn't hear any question."
    else:
        reply = ask_llm(objects, user_text)

    # Optionally speak out response
    speak(reply)

    return reply, objects


