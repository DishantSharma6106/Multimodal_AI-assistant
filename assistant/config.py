import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Multimodal AI Assistant"
ENV = os.getenv("APP_ENV", "dev")  # dev / prod
DEBUG = ENV != "prod"

# ---------- Vision ----------
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ---------- Audio / STT ----------
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SECONDS_DEFAULT = 5
AUDIO_TMP_FILE = "tmp_audio.wav"

STT_PROVIDER = os.getenv("STT_PROVIDER", "none")  # "none", "openai", "local"
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

# ---------- LLM ----------
# ✅ DEFAULT = LOCAL GPU
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # "local", "openai", "dummy"

# ---- OpenAI (OPTIONAL FALLBACK ONLY) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# ✅ Only enforce key if OpenAI is explicitly selected
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY is required when LLM_PROVIDER=openai")

# ---------- Local LLM (Ollama) ----------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # mistral / phi3 also valid

# ---------- TTS ----------
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "pyttsx3")  # "none", "pyttsx3"
TTS_VOICE = os.getenv("TTS_VOICE", "")

# ---------- Memory ----------
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "10"))
