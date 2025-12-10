# Multimodal AI Assistant
A real-time multimodal system that combines **computer vision**, **speech recognition**, **local LLM reasoning**, and **text-to-speech** to create an interactive AI assistant.
Developed by **Dishant Sharma**, 2nd year B.Tech, IIT BHU.

---

## 🚀 Features

### **1. Vision (YOLOv8 Integration)**
- Captures live webcam frames
- Detects objects using YOLOv8n
- Returns annotated frames with bounding boxes

### **2. Speech-to-Text (STT)**
- Microphone input with 5-second recording window
- Optional OpenAI Whisper integration
- Falls back to silent mode if STT is disabled

### **3. Local LLM Reasoning (Ollama)**
- Uses **LLaMA 3 (default)** or any local model supported by Ollama
- Supports conversation memory
- Combines user text + detected objects + memory → coherent response

### **4. Text-to-Speech (TTS)**
- Uses `pyttsx3` for offline speech generation
- Configurable voices

### **5. Web UI (Streamlit)**
- Live camera preview
- Object detection button
- Chat interface with full conversation history
- Toggle for microphone STT

---

## 🧩 Architecture Overview

```
Streamlit UI
     |
     V
Orchestrator (multimodal_turn)
     |
     +-- Vision (YOLOv8)
     +-- Audio (record + transcribe)
     +-- LLM (Ollama / OpenAI)
     +-- Memory Buffer (Last N turns)
     +-- TTS (pyttsx3)
```

---

## 📁 Project Structure

```
Multimodal_AI-assistant/
│
├── app.py                 # Streamlit frontend
├── orchestrator.py        # Main multimodal logic
├── core.py                # Alternate core flow
├── vision.py              # YOLO detection + webcam capture
├── audio_stt.py           # Recording + Whisper STT
├── llm_client.py          # Local/OpenAI LLM interface
├── tts_client.py          # Text-to-speech interface
├── memory.py              # Conversation memory
├── config.py              # Environment config + .env loader
├── logging_utils.py       # Logger formatter
├── requirements.txt       # Python dependencies
├── .env                   # Local environment variables
└── yolov8n.pt             # YOLO model weights
```

---

## 🔧 Installation

### **1. Clone Repo**

```
git clone https://github.com/DishantSharma6106/Multimodal_AI-assistant.git
cd Multimodal_AI-assistant
```

### **2. Install Dependencies**

```
pip install -r requirements.txt
```

### **3. Install and Run Ollama**

Download from:
https://ollama.com/download

Then pull the model:

```
ollama pull llama3
```

---

## ▶️ Usage

Run Streamlit app:

```
streamlit run app.py
```

Features inside UI:

- Capture webcam frame
- Detect objects
- Chat using text or microphone
- Watch annotated images
- Hear spoken replies
- Auto memory for last 10 turns

---

## 🔑 Environment Configuration

Create `.env` file:

```
APP_ENV=dev
YOLO_MODEL_PATH=yolov8n.pt
CAMERA_INDEX=0

# Local LLM provider
LLM_PROVIDER=local
OLLAMA_MODEL=llama3
OLLAMA_URL=http://localhost:11434/api/generate

# TTS
TTS_PROVIDER=pyttsx3
```

For Whisper STT:

```
STT_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_WHISPER_MODEL=whisper-1
```

---

## 💡 Future Improvements

- Add face recognition
- Add gesture recognition
- Add audio streaming mode
- Add mobile/websocket support
- Add GPU acceleration settings in UI

---

## 👤 Author

**Dishant Sharma**
2nd Year Undergraduate, IIT BHU
Computer Vision · Large Language Models · Systems Engineering

---

## 📝 License

This project is open-source and available under the MIT License.
