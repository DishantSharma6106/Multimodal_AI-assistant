from config import TTS_PROVIDER, TTS_VOICE
from logging_utils import get_logger

logger = get_logger()

_tts_engine = None


def _init_pyttsx3():
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    import pyttsx3
    engine = pyttsx3.init()
    if TTS_VOICE:
        # Try to set requested voice if available
        for voice in engine.getProperty("voices"):
            if TTS_VOICE.lower() in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
    _tts_engine = engine
    return engine


def speak(text: str) -> None:
    if TTS_PROVIDER == "none":
        logger.info("[TTS] Skipping speaking (provider = none).")
        return

    if TTS_PROVIDER == "pyttsx3":
        engine = _init_pyttsx3()
        logger.info(f"[TTS] Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
        return

    logger.warning(f"[TTS] Unknown provider '{TTS_PROVIDER}', not speaking.")
