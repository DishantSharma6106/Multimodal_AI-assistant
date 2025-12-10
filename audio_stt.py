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
)
from logging_utils import get_logger

logger = get_logger()

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
    if not _openai_client:
        logger.error("[STT] OpenAI client not available.")
        return ""

    # NOTE: adapt to exact Whisper endpoint you plan to use.
    try:
        with open(filename, "rb") as f:
            # If you use a Whisper model name, adjust here
            resp = _openai_client.audio.transcriptions.create(
                model="whisper-1",
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
