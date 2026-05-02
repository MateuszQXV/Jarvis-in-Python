🧠 JARVIS v2 – Polish Voice Assistant
📌 Description

JARVIS v2 is a desktop voice assistant written in Python.
It supports voice commands, AI responses, and system control.

Main features:

🎤 Voice recognition (Google STT / Whisper)
🧠 AI integration (OpenAI / Ollama)
🗣️ Text-to-speech (offline)
🖥️ System control (apps, volume, windows)
🎧 Spotify integration
💾 Long-term memory
🎮 Push-to-Talk & wake word ("Jarvis")
🖼️ GUI (customtkinter)
⚙️ Requirements
🐍 Python
Python 3.10+ recommended
📦 Required packages

Install all dependencies:

pip install pyttsx3 SpeechRecognition pyaudio openai faster-whisper customtkinter pynput pvporcupine spotipy numpy
🔑 Optional API Keys (ENV VARIABLES)

Create environment variables:

OPENAI_API_KEY=your_key_here
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b
PICOVOICE_ACCESS_KEY=your_key_here

SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8765/callback

HF_TOKEN=your_token

👉 Not all are required:

OpenAI → for cloud AI
Ollama → for local AI
Spotify → music control
Porcupine → wake word
▶️ How to run
python jarvis.py
🧩 Features Explained
🎤 Voice Control
Say: Jarvis open Chrome
Or use Push-To-Talk
🧠 AI Assistant

Ask anything:

Jarvis what is Python?
🖥️ System Commands
Open apps
Close apps
Control volume
Manage windows
🎧 Spotify
Play music
Pause / next / previous
Play playlists
💾 Memory

Save facts:

remember that I like Python
Jarvis remembers across sessions
🖼️ GUI
Built with customtkinter
Shows:
mic activity
chat
memory
settings
⚠️ Notes
Works best on Windows
Requires microphone
Some features need API keys
Without keys → fallback modes are used
📁 Files
jarvis.py           # main application
jarvis_config.json # user settings (auto-created)
jarvis_memory.json # saved memory (auto-created)
jarvis_notes.txt   # notes
🚀 Future Improvements
Better UI
More commands
Mobile version
Voice cloning
👤 Author

Matforest
