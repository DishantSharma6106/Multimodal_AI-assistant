# audio_stt.py
import sounddevice as sd
from scipy.io.wavfile import write
from typing import Optional

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_SECONDS_DEFAULT,
    AUDIO_TMP_FILE,
    STT_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_WHISPER_MODEL,
)
from logging_utils import get_logger

logger = get_logger()

# Try initializing OpenAI client if API key exists
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    _openai_client = None
    logger.error(f"[STT] Failed to init OpenAI client: {e}")


def record_audio(seconds: int = AUDIO_SECONDS_DEFAULT,
                 filename: str = AUDIO_TMP_FILE) -> str:
    """
    Record audio from default mic into a WAV file.
    """
    logger.info(f"[AUDIO] Recording {seconds} seconds of audio...")
    audio = sd.rec(int(seconds * AUDIO_SAMPLE_RATE),
                   samplerate=AUDIO_SAMPLE_RATE,
                   channels=AUDIO_CHANNELS)
    sd.wait()
    write(filename, AUDIO_SAMPLE_RATE, audio)
    logger.info(f"[AUDIO] Saved recording to {filename}")
    return filename


def _transcribe_openai(filename: str) -> str:
    """
    Transcribe using OpenAI Whisper endpoint.
    Uses the model name defined in OPENAI_WHISPER_MODEL.
    """
    if not _openai_client:
        logger.error("[STT] OpenAI client not available.")
        return ""

    try:
        with open(filename, "rb") as f:
            resp = _openai_client.audio.transcriptions.create(
                model=OPENAI_WHISPER_MODEL,
                file=f
            )
        text = resp.text.strip()
        logger.info(f"[STT] Transcription: {text}")
        return text
    except Exception as e:
        logger.error(f"[STT] OpenAI transcription error: {e}")
        return ""


def transcribe_audio(filename: str) -> str:
    """
    Transcribe audio file to text via configured STT provider.
    """
    if STT_PROVIDER == "openai":
        return _transcribe_openai(filename)

    logger.info("[STT] No STT provider configured, returning empty text.")
    return ""
