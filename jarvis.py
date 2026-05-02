"""
JARVIS v2 - Polski asystent glosowy
Funkcje: Porcupine wake word + fallback, Push-to-Talk, streaming LLM,
pamiec dlugoterminowa, Spotify Web API, lepsze GUI (customtkinter).

WYMAGANE pakiety:
    pip install pyttsx3 SpeechRecognition pyaudio openai faster-whisper \
                customtkinter pynput pvporcupine spotipy numpy

Opcjonalne klucze (env vars):
    OPENAI_API_KEY            - dla GPT
    OLLAMA_URL, OLLAMA_MODEL  - dla lokalnego LLM
    PICOVOICE_ACCESS_KEY      - dla Porcupine wake word
    SPOTIFY_CLIENT_ID         - Spotify Web API
    SPOTIFY_CLIENT_SECRET     - Spotify Web API
    SPOTIFY_REDIRECT_URI      - np. http://127.0.0.1:8765/callback
    HF_TOKEN                  - dla faster-whisper (opcjonalne)

URUCHOMIENIE:
    python jarvis.py
"""

from __future__ import annotations

import ctypes
import datetime
import difflib
import io
import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from typing import Optional

import pyttsx3
import speech_recognition as sr

# Optional deps
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import pvporcupine
    import pyaudio
except Exception:
    pvporcupine = None
    pyaudio = None

try:
    from pynput import keyboard as pkeyboard
    from pynput import mouse as pmouse
except Exception:
    pkeyboard = None
    pmouse = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except Exception:
    spotipy = None
    SpotifyOAuth = None

try:
    import customtkinter as ctk
except Exception:
    ctk = None

try:
    import numpy as np
except Exception:
    np = None

# ================== CONFIG ==================
WAKE_WORD = "jarvis"
EXIT_WORD = "koniec"
MODEL_NAME = "gpt-4o-mini"
CALIBRATION_INTERVAL_SECONDS = 120
FOLLOWUP_WINDOW_SECONDS = 18
MAX_HISTORY_MESSAGES = 8
MAX_RESPONSE_TOKENS = 80
COMMAND_DEBOUNCE_SECONDS = 1.5
MIN_COMMAND_LENGTH = 2
MAX_PENDING_AI_COMMANDS = 1
AI_PREFIXES = ("zapytaj", "powiedz", "jarvis", "ai")

# Halucynacje Whispera na ciszy/szumach
WHISPER_HALLUCINATIONS = {
    "dziekuje za uwage",
    "napisy stworzone przez spolecznosc amara",
    "subscribe to my channel",
    "thanks for watching",
    "napisy przygotowane",
    "subskrybuj",
}

IGNORED_PHRASES = {
    "yyy", "hmm", "aha", "ok", "okej", "dobra", "no", "yyyy",
}

STT_BACKEND = "google"  # google = szybciej i dokladniej dla PL niz Whisper-base
WHISPER_MODEL_SIZE = "tiny"  # tiny = ~3x szybszy od base na CPU
WHISPER_COMPUTE_TYPE = "int8"
SKIP_WHISPER_WITHOUT_HF_TOKEN = True

DEFAULT_VOICE_PROFILE = "turbo"
VOICE_PROFILES = {
    "normal": {"phrase_time_limit": 12.0, "max_response_tokens": 100, "tts_rate": 185, "command_ack": True},
    "turbo": {"phrase_time_limit": 8.0, "max_response_tokens": 50, "tts_rate": 215, "command_ack": False},
}

# PTT modes
PTT_MODES = ("off", "ctrl", "mouse_x1", "ctrl+mouse")

# Plik konfiguracji uzytkownika (utrzymuje stan miedzy uruchomieniami)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_config.json")
MEMORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_memory.json")
NOTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_notes.txt")

# API/integration env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8765/callback")

# Win32 virtual keys
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MENU = 0x12
VK_F4 = 0x73
VK_LWIN = 0x5B
VK_D = 0x44
VK_M = 0x4D


# ================== STATE ==================
@dataclass
class AppState:
    running: bool = True
    awaiting_shutdown: bool = False
    last_command_text: str = ""
    last_command_ts: float = 0.0
    voice_profile: str = DEFAULT_VOICE_PROFILE
    require_ai_prefix: bool = False
    require_wake_word: bool = False
    ptt_mode: str = "off"  # off | ctrl | mouse_x1 | ctrl+mouse
    ptt_active: bool = False
    mic_muted: bool = False
    mic_level: float = 0.0
    last_mic_activity: float = 0.0
    chat_history: list = field(default_factory=lambda: [
        {"role": "system", "content": "Jestes Jarvisem. Mowisz po polsku, JEDNYM krotkim zdaniem, max 12 slow. Bez wstepow typu 'oczywiscie', 'jasne', 'rozumiem'. Konkret. Jesli nie wiesz - powiedz 'nie wiem'."}
    ])

state = AppState()
gui_queue: queue.Queue = queue.Queue()
speech_queue: queue.Queue = queue.Queue()
command_queue: queue.Queue = queue.Queue()
is_speaking = threading.Event()
cancel_speech = threading.Event()
ptt_pressed = threading.Event()  # set when PTT active
history_lock = threading.Lock()


# ================== CONFIG PERSISTENCE ==================
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        state.voice_profile = data.get("voice_profile", state.voice_profile)
        state.require_ai_prefix = bool(data.get("require_ai_prefix", state.require_ai_prefix))
        state.require_wake_word = bool(data.get("require_wake_word", state.require_wake_word))
        state.ptt_mode = data.get("ptt_mode", state.ptt_mode)
    except Exception as exc:
        print(f"[BLAD] Nie moge wczytac configu: {exc}")


def save_config():
    try:
        data = {
            "voice_profile": state.voice_profile,
            "require_ai_prefix": state.require_ai_prefix,
            "require_wake_word": state.require_wake_word,
            "ptt_mode": state.ptt_mode,
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[BLAD] Zapis configu: {exc}")


# ================== LONG-TERM MEMORY ==================
class Memory:
    def __init__(self, path):
        self.path = path
        self.facts: list[str] = []
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.facts = list(data.get("facts", []))
        except Exception:
            self.facts = []

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"facts": self.facts}, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[BLAD] Zapis pamieci: {exc}")

    def add(self, fact: str):
        fact = fact.strip()
        if not fact:
            return False
        # Deduplicate similar facts
        norm = normalize_text(fact)
        for existing in self.facts:
            if difflib.SequenceMatcher(None, norm, normalize_text(existing)).ratio() > 0.85:
                return False
        self.facts.append(fact)
        self.save()
        return True

    def remove(self, idx: int):
        if 0 <= idx < len(self.facts):
            self.facts.pop(idx)
            self.save()
            return True
        return False

    def clear(self):
        self.facts = []
        self.save()

    def system_prompt_addon(self) -> str:
        if not self.facts:
            return ""
        bullets = "\n".join(f"- {f}" for f in self.facts)
        return f"\n\nWazne fakty o uzytkowniku, ktore znasz:\n{bullets}"


memory = Memory(MEMORY_PATH)


# ================== TEXT HELPERS ==================
def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return text


def fuzzy_contains_phrase(text: str, phrase: str, cutoff: float = 0.78) -> bool:
    words = normalize_text(text).split()
    target = normalize_text(phrase).split()
    if not words or not target or len(words) < len(target):
        return False
    for i in range(len(words) - len(target) + 1):
        chunk = " ".join(words[i:i + len(target)])
        if difflib.SequenceMatcher(None, chunk, " ".join(target)).ratio() >= cutoff:
            return True
    return False


def fuzzy_contains_any(text: str, phrases, cutoff: float = 0.78) -> bool:
    norm = normalize_text(text)
    for phrase in phrases:
        p = normalize_text(phrase)
        if p in norm or fuzzy_contains_phrase(norm, p, cutoff):
            return True
    return False


def best_fuzzy_match(query: str, choices, cutoff: float = 0.7):
    nq = normalize_text(query)
    if not nq:
        return None
    matches = difflib.get_close_matches(nq, [normalize_text(c) for c in choices], n=1, cutoff=cutoff)
    if not matches:
        return None
    for original in choices:
        if normalize_text(original) == matches[0]:
            return original
    return None


def extract_after_phrase(text: str, phrases) -> str:
    norm = normalize_text(text)
    for phrase in phrases:
        p = normalize_text(phrase)
        idx = norm.find(p)
        if idx >= 0:
            value = text[idx + len(phrase):].strip(" ,.:;!?-")
            if value:
                return value
    return ""


# ================== GUI HELPERS ==================
def enqueue_gui(text: str, kind: str = "log"):
    gui_queue.put((kind, text))


def get_profile_value(key, default):
    return VOICE_PROFILES.get(state.voice_profile, {}).get(key, default)


def should_ack_commands() -> bool:
    return bool(get_profile_value("command_ack", True))


# ================== TTS ==================
engine = pyttsx3.init()
engine.setProperty("rate", VOICE_PROFILES[state.voice_profile]["tts_rate"])
tts_lock = threading.Lock()


def speaker_worker():
    while state.running:
        try:
            text = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if text is None:
            return
        if cancel_speech.is_set():
            cancel_speech.clear()
            continue
        try:
            is_speaking.set()
            with tts_lock:
                engine.say(text)
                engine.runAndWait()
        except Exception as exc:
            enqueue_gui(f"[BLAD TTS] {exc}")
        finally:
            is_speaking.clear()


def clear_pending_speech():
    while True:
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            break


def stop_speaking():
    clear_pending_speech()
    cancel_speech.set()
    try:
        with tts_lock:
            engine.stop()
    except Exception:
        pass


def speak(text: str):
    text = (text or "").strip()
    if not text:
        return
    enqueue_gui("Jarvis: " + text, kind="bot")
    if speech_queue.qsize() > 3:
        clear_pending_speech()
    speech_queue.put(text)


def speak_command(text: str):
    if should_ack_commands():
        speak(text)
    else:
        enqueue_gui("Jarvis: " + text, kind="bot")


def trim_history():
    with history_lock:
        if len(state.chat_history) <= MAX_HISTORY_MESSAGES + 1:
            return
        sys_msg = state.chat_history[0]
        recent = state.chat_history[-MAX_HISTORY_MESSAGES:]
        state.chat_history[:] = [sys_msg] + recent


def build_system_prompt() -> dict:
    base = "Jestes Jarvisem. Mowisz po polsku, JEDNYM krotkim zdaniem, max 12 slow. Bez wstepow typu 'oczywiscie', 'jasne', 'rozumiem'. Konkret. Jesli nie wiesz - powiedz 'nie wiem'."
    addon = memory.system_prompt_addon()
    return {"role": "system", "content": base + addon}


# ================== WHISPER STT ==================
whisper_model = None


def init_local_stt() -> bool:
    global whisper_model
    if WhisperModel is None:
        enqueue_gui("[INFO] faster-whisper nie zainstalowany - uzywam Google STT.")
        return False
    if SKIP_WHISPER_WITHOUT_HF_TOKEN and not os.getenv("HF_TOKEN"):
        enqueue_gui("[INFO] Brak HF_TOKEN - uzywam Google STT.")
        return False
    try:
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
        enqueue_gui("[INFO] Whisper aktywny.")
        return True
    except Exception as exc:
        enqueue_gui(f"[BLAD] Whisper init: {exc}")
        return False


def is_whisper_hallucination(text: str) -> bool:
    norm = normalize_text(text)
    return any(h in norm for h in WHISPER_HALLUCINATIONS)


def transcribe_with_whisper(audio) -> str:
    if whisper_model is None:
        return ""
    try:
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        segments, _ = whisper_model.transcribe(
            io.BytesIO(wav_bytes), language="pl", vad_filter=True, beam_size=1,
            no_speech_threshold=0.6,
        )
        parts = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0) > 0.7:
                continue
            parts.append(seg.text.strip())
        text = " ".join(parts).strip().lower()
        if text and not is_whisper_hallucination(text):
            enqueue_gui("Ty: " + text, kind="user")
            return text
    except Exception as exc:
        enqueue_gui(f"[BLAD Whisper] {exc}")
    return ""


def transcribe_with_google(audio) -> str:
    try:
        text = recognizer.recognize_google(audio, language="pl-PL").lower().strip()
        if text:
            enqueue_gui("Ty: " + text, kind="user")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        enqueue_gui(f"[BLAD Google STT] {exc}")
        return ""
    except Exception as exc:
        enqueue_gui(f"[BLAD STT] {exc}")
        return ""


# ================== MIC / VAD ==================
recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.0   # czas ciszy zeby uznac koniec wypowiedzi (bylo 0.4 - za krotko)
recognizer.non_speaking_duration = 0.5
recognizer.phrase_threshold = 0.2   # min dlugosc dzwieku zeby uznac za mowe (szybsze wykrycie startu)
recognizer.energy_threshold = 350
recognizer.dynamic_energy_threshold = True
recognizer.dynamic_energy_adjustment_damping = 0.15


def compute_mic_level(audio_data: bytes) -> float:
    """RMS znormalizowane do 0..1 dla wskaznika w GUI."""
    try:
        if np is not None:
            arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            if arr.size == 0:
                return 0.0
            rms = float(np.sqrt(np.mean(arr * arr)))
        else:
            n = len(audio_data) // 2
            if n == 0:
                return 0.0
            samples = struct.unpack(f"<{n}h", audio_data[:n * 2])
            rms = (sum(s * s for s in samples) / n) ** 0.5
        return max(0.0, min(1.0, rms / 8000.0))
    except Exception:
        return 0.0


# ================== PORCUPINE WAKE WORD ==================
class WakeWordDetector:
    """Porcupine wake word detector. Auto-fallback gdy brak klucza/lib."""

    def __init__(self):
        self.porcupine = None
        self.audio = None
        self.stream = None
        self.enabled = False
        self.frame_length = 512
        self.sample_rate = 16000

    def init(self) -> bool:
        if not PICOVOICE_ACCESS_KEY or pvporcupine is None or pyaudio is None:
            enqueue_gui("[INFO] Porcupine wylaczony - uzywam fallback string-match 'jarvis'.")
            return False
        try:
            self.porcupine = pvporcupine.create(
                access_key=PICOVOICE_ACCESS_KEY,
                keywords=["jarvis"],
            )
            self.frame_length = self.porcupine.frame_length
            self.sample_rate = self.porcupine.sample_rate
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                rate=self.sample_rate, channels=1, format=pyaudio.paInt16,
                input=True, frames_per_buffer=self.frame_length,
            )
            self.enabled = True
            enqueue_gui("[INFO] Porcupine wake-word aktywny.")
            return True
        except Exception as exc:
            enqueue_gui(f"[BLAD Porcupine] {exc} - fallback aktywny.")
            self.cleanup()
            return False

    def detected(self) -> bool:
        if not self.enabled:
            return False
        try:
            pcm = self.stream.read(self.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from("h" * self.frame_length, pcm)
            return self.porcupine.process(pcm_unpacked) >= 0
        except Exception:
            return False

    def cleanup(self):
        try:
            if self.stream:
                self.stream.close()
            if self.audio:
                self.audio.terminate()
            if self.porcupine:
                self.porcupine.delete()
        except Exception:
            pass
        self.enabled = False


wake_detector = WakeWordDetector()


# ================== PUSH-TO-TALK ==================
class PTTController:
    def __init__(self):
        self.kb_listener = None
        self.mouse_listener = None
        self.ctrl_held = False
        self.x1_held = False

    def update(self):
        ptt_pressed.clear()
        if state.ptt_mode == "ctrl" and self.ctrl_held:
            ptt_pressed.set()
        elif state.ptt_mode == "mouse_x1" and self.x1_held:
            ptt_pressed.set()
        elif state.ptt_mode == "ctrl+mouse" and (self.ctrl_held or self.x1_held):
            ptt_pressed.set()

    def on_press(self, key):
        try:
            if key in (pkeyboard.Key.ctrl_l, pkeyboard.Key.ctrl_r, pkeyboard.Key.ctrl):
                self.ctrl_held = True
                self.update()
        except Exception:
            pass

    def on_release(self, key):
        try:
            if key in (pkeyboard.Key.ctrl_l, pkeyboard.Key.ctrl_r, pkeyboard.Key.ctrl):
                self.ctrl_held = False
                self.update()
        except Exception:
            pass

    def on_click(self, x, y, button, pressed):
        try:
            if button == pmouse.Button.x1:
                self.x1_held = pressed
                self.update()
        except Exception:
            pass

    def start(self):
        if pkeyboard is None or pmouse is None:
            enqueue_gui("[INFO] pynput niedostepne - PTT wylaczone.")
            return
        try:
            self.kb_listener = pkeyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.kb_listener.daemon = True
            self.kb_listener.start()
            self.mouse_listener = pmouse.Listener(on_click=self.on_click)
            self.mouse_listener.daemon = True
            self.mouse_listener.start()
            enqueue_gui("[INFO] PTT listener aktywny.")
        except Exception as exc:
            enqueue_gui(f"[BLAD PTT] {exc}")


ptt = PTTController()


def is_listening_allowed() -> bool:
    """Czy w tym momencie powinnismy slyszec uzytkownika."""
    if state.mic_muted:
        return False
    if is_speaking.is_set():
        return False
    if state.ptt_mode != "off":
        return ptt_pressed.is_set()
    return True


# ================== SPOTIFY ==================
class SpotifyController:
    def __init__(self):
        self.sp: Optional["spotipy.Spotify"] = None
        self.enabled = False

    def init(self) -> bool:
        if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and spotipy and SpotifyOAuth):
            enqueue_gui("[INFO] Spotify Web API wylaczony - uzywam klawiszy multimedialnych.")
            return False
        try:
            scope = "user-modify-playback-state user-read-playback-state playlist-read-private user-read-currently-playing"
            auth = SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=scope,
                cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spotify_cache"),
                open_browser=True,
            )
            self.sp = spotipy.Spotify(auth_manager=auth)
            self.sp.current_user()  # test
            self.enabled = True
            enqueue_gui("[INFO] Spotify Web API aktywny.")
            return True
        except Exception as exc:
            enqueue_gui(f"[BLAD Spotify] {exc} - fallback na klawisze.")
            self.enabled = False
            return False

    def _get_device(self) -> Optional[str]:
        try:
            devices = self.sp.devices().get("devices", [])
            if not devices:
                return None
            for d in devices:
                if d.get("is_active"):
                    return d["id"]
            return devices[0]["id"]
        except Exception:
            return None

    def play_playlist(self, query: str) -> bool:
        if not self.enabled:
            return False
        try:
            res = self.sp.search(q=query, type="playlist", limit=1)
            items = res.get("playlists", {}).get("items", [])
            if not items:
                return False
            uri = items[0]["uri"]
            device = self._get_device()
            self.sp.start_playback(device_id=device, context_uri=uri)
            return True
        except Exception as exc:
            enqueue_gui(f"[Spotify] {exc}")
            return False

    def play_track(self, query: str) -> bool:
        if not self.enabled:
            return False
        try:
            res = self.sp.search(q=query, type="track", limit=1)
            items = res.get("tracks", {}).get("items", [])
            if not items:
                return False
            uri = items[0]["uri"]
            device = self._get_device()
            self.sp.start_playback(device_id=device, uris=[uri])
            return True
        except Exception as exc:
            enqueue_gui(f"[Spotify] {exc}")
            return False

    def pause(self) -> bool:
        if not self.enabled:
            return False
        try:
            self.sp.pause_playback()
            return True
        except Exception:
            return False

    def resume(self) -> bool:
        if not self.enabled:
            return False
        try:
            self.sp.start_playback()
            return True
        except Exception:
            return False

    def next(self) -> bool:
        if not self.enabled:
            return False
        try:
            self.sp.next_track()
            return True
        except Exception:
            return False

    def previous(self) -> bool:
        if not self.enabled:
            return False
        try:
            self.sp.previous_track()
            return True
        except Exception:
            return False

    def now_playing(self) -> str:
        if not self.enabled:
            return ""
        try:
            cur = self.sp.current_user_playing_track()
            if not cur or not cur.get("item"):
                return ""
            item = cur["item"]
            artists = ", ".join(a["name"] for a in item.get("artists", []))
            return f"{item.get('name', '')} - {artists}"
        except Exception:
            return ""


spotify = SpotifyController()


# ================== AI STREAMING ==================
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]+|[^.!?\n]+$")


def stream_split_for_speech(buffer: str) -> tuple[list[str], str]:
    """Z bufora tekstu wyciagnij gotowe zdania do TTS, reszte zostaw.
    Min 3 slowa zeby pierwsze TTS zaczelo - balans miedzy szybkoscia a plynnoscia."""
    sentences = SENTENCE_RE.findall(buffer)
    if not sentences:
        return [], buffer
    if not buffer.rstrip().endswith((".", "!", "?")):
        last = sentences[-1]
        complete = sentences[:-1]
        result = [s.strip() for s in complete if s.strip()]
        if result and len(result[0].split()) < 3:
            return [], buffer
        return result, last
    result = [s.strip() for s in sentences if s.strip()]
    return result, ""


def warmup_tts():
    """Pierwsza inicjalizacja silnika TTS jest wolna (~200-300ms na SAPI5).
    Robimy to w tle przy starcie zeby user nie czekal na pierwszej wypowiedzi."""
    try:
        with tts_lock:
            engine.say(" ")
            engine.runAndWait()
    except Exception:
        pass


def warmup_ollama():
    """Rozgrzewa model w tle przy starcie zeby pierwsza komenda nie czekala na load do RAM."""
    try:
        payload = {"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": "ok"}], "stream": False}
        req = urllib.request.Request(
            OLLAMA_URL.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        enqueue_gui("[INFO] Ollama warmup OK.")
    except Exception:
        enqueue_gui("[INFO] Ollama warmup pominiety (offline?).")


def stream_ollama(prompt: str):
    """Yieldy fragmentow tekstu od Ollama. Rzuca wyjatek przy braku polaczenia."""
    with history_lock:
        messages = [build_system_prompt()] + state.chat_history[1:]
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": True}
    req = urllib.request.Request(
        OLLAMA_URL.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            chunk = (obj.get("message") or {}).get("content", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


def stream_openai(prompt: str):
    if not (OpenAI and OPENAI_API_KEY):
        return
    client = OpenAI(api_key=OPENAI_API_KEY)
    with history_lock:
        messages = [build_system_prompt()] + state.chat_history[1:]
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.6,
        max_tokens=int(get_profile_value("max_response_tokens", MAX_RESPONSE_TOKENS)),
        stream=True,
    )
    for chunk in response:
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            yield delta


def ask_ai_streaming(prompt: str) -> str:
    """Streamuje LLM, mowi zdania na biezaco. Zwraca pelna odpowiedz.
    Tnie odpowiedz po 2 zdaniach zeby Jarvis nie paplal."""
    with history_lock:
        state.chat_history.append({"role": "user", "content": prompt})
    trim_history()

    full_text = ""
    buffer = ""
    spoken_sentences = 0
    MAX_SENTENCES = 1
    sources = []
    # OpenAI w chmurze jest 3-5x szybszy od lokalnego Ollama na CPU - probuj go pierwszego
    if OPENAI_API_KEY and OpenAI:
        sources.append(("openai", stream_openai))
    sources.append(("ollama", stream_ollama))

    last_error = None
    for name, fn in sources:
        try:
            for chunk in fn(prompt):
                buffer += chunk
                full_text += chunk
                ready, buffer = stream_split_for_speech(buffer)
                for sentence in ready:
                    if spoken_sentences >= MAX_SENTENCES:
                        break
                    speak(sentence)
                    spoken_sentences += 1
                if spoken_sentences >= MAX_SENTENCES:
                    break
            # Resztki
            if buffer.strip() and spoken_sentences < MAX_SENTENCES:
                speak(buffer.strip())
                buffer = ""
            if full_text.strip():
                with history_lock:
                    state.chat_history.append({"role": "assistant", "content": full_text})
                trim_history()
                return full_text
        except urllib.error.URLError as exc:
            last_error = f"Ollama offline ({exc})"
            continue
        except Exception as exc:
            last_error = str(exc)
            continue

    fallback = "Nie wiem."
    if last_error:
        enqueue_gui(f"[AI] {last_error}")
    speak(fallback)
    return fallback


# ================== APP CATALOG ==================
APP_ACTION_OPEN = ("odpal", "uruchom", "otworz", "wlacz")
APP_ACTION_CLOSE = ("zamknij", "wylacz", "zabij")

APP_CATALOG = {
    "chrome": {"aliases": ("chrome", "google chrome", "przegladarka"), "start": "chrome.exe", "processes": ("chrome.exe",)},
    "steam": {"aliases": ("steam", "seam", "stim"), "start": "steam://open/main", "processes": ("steam.exe",)},
    "discord": {"aliases": ("discord",), "start": "discord.exe", "processes": ("discord.exe",)},
    "spotify": {"aliases": ("spotify",), "start": "spotify.exe", "processes": ("spotify.exe",)},
    "explorer": {"aliases": ("explorer", "foldery", "menedzer plikow"), "start": "explorer.exe", "processes": ("explorer.exe",)},
    "notepad": {"aliases": ("notatnik", "notepad"), "start": "notepad.exe", "processes": ("notepad.exe",)},
    "edge": {"aliases": ("edge", "microsoft edge"), "start": "msedge.exe", "processes": ("msedge.exe",)},
    "firefox": {"aliases": ("firefox", "mozilla"), "start": "firefox.exe", "processes": ("firefox.exe",)},
    "vscode": {"aliases": ("vscode", "visual studio code", "cursor", "edytor"), "start": "code.exe", "processes": ("code.exe", "cursor.exe")},
    "calculator": {"aliases": ("kalkulator", "calculator"), "start": "calc.exe", "processes": ("calculatorapp.exe", "calc.exe")},
    "paint": {"aliases": ("paint", "rysowanie"), "start": "mspaint.exe", "processes": ("mspaint.exe",)},
    "taskmgr": {"aliases": ("menedzer zadan", "task manager"), "start": "taskmgr.exe", "processes": ("taskmgr.exe",)},
    "cmd": {"aliases": ("cmd", "wiersz polecen"), "start": "cmd.exe", "processes": ("cmd.exe",)},
    "powershell": {"aliases": ("powershell", "terminal"), "start": "powershell.exe", "processes": ("powershell.exe", "pwsh.exe")},
}

WEBSITE_CATALOG = {
    "youtube": "https://youtube.com", "google": "https://google.com",
    "gmail": "https://mail.google.com", "facebook": "https://facebook.com",
    "instagram": "https://instagram.com", "x": "https://x.com",
    "twitter": "https://x.com", "twitch": "https://twitch.tv",
    "netflix": "https://netflix.com", "github": "https://github.com",
}


def resolve_app_key(text: str):
    norm = normalize_text(text)
    for app_key, cfg in APP_CATALOG.items():
        for alias in cfg["aliases"]:
            if normalize_text(alias) in norm:
                return app_key
    words = norm.split()
    if not words:
        return None
    aliases_flat = []
    alias_to_app = {}
    for app_key, cfg in APP_CATALOG.items():
        for a in cfg["aliases"]:
            aliases_flat.append(a)
            alias_to_app[normalize_text(a)] = app_key
    candidate = best_fuzzy_match(" ".join(words[-2:]), aliases_flat, 0.72) or best_fuzzy_match(words[-1], aliases_flat, 0.72)
    return alias_to_app.get(normalize_text(candidate)) if candidate else None


def resolve_website(text: str):
    norm = normalize_text(text)
    for key, url in WEBSITE_CATALOG.items():
        if normalize_text(key) in norm:
            return key, url
    words = norm.split()
    if words:
        cand = best_fuzzy_match(words[-1], list(WEBSITE_CATALOG.keys()), 0.7)
        if cand:
            return cand, WEBSITE_CATALOG[cand]
    return None, None


# ================== KEYBOARD HOTKEYS (Win32) ==================
def tap_key(vk: int):
    try:
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, 2, 0)
    except Exception:
        pass


def send_hotkey(vk_mod: int, vk_key: int):
    try:
        ctypes.windll.user32.keybd_event(vk_mod, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk_key, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk_key, 0, 2, 0)
        ctypes.windll.user32.keybd_event(vk_mod, 0, 2, 0)
    except Exception:
        pass


# ================== APP / WEB COMMANDS ==================
def launch_app(app_key: str) -> bool:
    cfg = APP_CATALOG.get(app_key)
    if not cfg:
        return False
    target = cfg["start"]
    try:
        if target.startswith(("http://", "https://", "steam://")):
            webbrowser.open(target)
        else:
            subprocess.Popen(target, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:
        enqueue_gui(f"[BLAD] Launch {app_key}: {exc}")
        return False


def close_app(app_key: str) -> bool:
    cfg = APP_CATALOG.get(app_key)
    if not cfg:
        return False
    closed = False
    for proc in cfg["processes"]:
        try:
            r = subprocess.run(["taskkill", "/IM", proc, "/F"], capture_output=True, text=True, check=False)
            if r.returncode == 0:
                closed = True
        except Exception as exc:
            enqueue_gui(f"[BLAD] Kill {proc}: {exc}")
    return closed


def handle_app_control(text: str) -> bool:
    norm = normalize_text(text)
    if fuzzy_contains_any(norm, APP_ACTION_OPEN, 0.74):
        action = "open"
    elif fuzzy_contains_any(norm, APP_ACTION_CLOSE, 0.74):
        action = "close"
    else:
        return False
    app_key = resolve_app_key(text)
    if not app_key:
        return False  # niech AI sprobuje cos zrobic
    if action == "open":
        if launch_app(app_key):
            speak_command(f"Uruchamiam {app_key}.")
        else:
            speak(f"Nie udalo mi sie uruchomic {app_key}.")
    else:
        if close_app(app_key):
            speak_command(f"Zamykam {app_key}.")
        else:
            speak(f"Nie widze procesu {app_key}.")
    return True


def handle_web_commands(text: str) -> bool:
    norm = normalize_text(text)
    for phrases, kind in [
        (("szukaj w google", "wyszukaj w google", "google to", "znajdz w google"), "google"),
        (("szukaj na youtube", "wyszukaj na youtube", "youtube to", "znajdz na youtube"), "yt"),
    ]:
        q = extract_after_phrase(text, phrases)
        if q:
            if kind == "google":
                webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(q))
                speak_command(f"Szukam w Google: {q}.")
            else:
                webbrowser.open("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q))
                speak_command(f"Szukam na YouTube: {q}.")
            return True

    if fuzzy_contains_any(norm, ("otworz strone",), 0.74):
        raw = extract_after_phrase(text, ("otworz strone",))
        if raw:
            domain = raw.replace(" ", "").lower()
            if "." not in domain:
                domain += ".com"
            webbrowser.open("https://" + domain)
            speak_command(f"Otwieram {domain}.")
            return True

    site, url = resolve_website(text)
    if url and fuzzy_contains_any(norm, ("otworz", "odpal", "wlacz"), 0.74):
        webbrowser.open(url)
        speak_command(f"Otwieram {site}.")
        return True
    return False


# ================== MEDIA / SPOTIFY ==================
def handle_spotify_commands(text: str) -> bool:
    norm = normalize_text(text)

    # "wlacz playliste X" / "odpal playliste X"
    pl = extract_after_phrase(text, ("wlacz playliste", "odpal playliste", "wlacz playlista", "odtworz playliste", "playlista"))
    if pl:
        if spotify.enabled and spotify.play_playlist(pl):
            speak_command(f"Wlaczam playliste {pl}.")
        else:
            speak("Nie moge wlaczyc tej playlisty. Sprawdz Spotify Premium i autoryzacje.")
        return True

    # "zagraj X" / "wlacz utwor X" / "puscic X"
    track = extract_after_phrase(text, ("zagraj utwor", "wlacz utwor", "zagraj piosenke", "wlacz piosenke", "zagraj"))
    if track and (spotify.enabled or fuzzy_contains_any(norm, ("spotify",), 0.8)):
        if spotify.enabled and spotify.play_track(track):
            speak_command(f"Odtwarzam {track}.")
            return True

    if fuzzy_contains_any(norm, ("co teraz leci", "co gra", "co sluchamy"), 0.75):
        if spotify.enabled:
            np_text = spotify.now_playing()
            speak(f"Teraz: {np_text}." if np_text else "Nic nie leci.")
        else:
            speak("Spotify Web API nie jest aktywne.")
        return True

    return False


def handle_media_commands(text: str) -> bool:
    norm = normalize_text(text)

    if any(x in norm for x in ("cisza", "stop mow", "przestan mowic")):
        stop_speaking()
        enqueue_gui("Jarvis: Zatrzymuje mowienie.", kind="bot")
        return True

    if any(x in norm for x in ("glosniej", "poglosnij", "podglosnij")):
        for _ in range(8):
            tap_key(VK_VOLUME_UP)
        speak_command("Podglasniam.")
        return True

    if any(x in norm for x in ("ciszej", "scisz", "przycisz")):
        for _ in range(8):
            tap_key(VK_VOLUME_DOWN)
        speak_command("Przyciszam.")
        return True

    if any(x in norm for x in ("wycisz", "odcisz", "mute")):
        tap_key(VK_VOLUME_MUTE)
        speak_command("Przelaczam wyciszenie.")
        return True

    # Pauza/play - prefer Spotify Web API
    if any(x in norm for x in ("pauza", "wznow", "play", "stop muzyke", "wstrzymaj")):
        if spotify.enabled:
            ok = spotify.pause() or spotify.resume()
            if ok:
                speak_command("Przelaczam.")
                return True
        tap_key(VK_MEDIA_PLAY_PAUSE)
        speak_command("Przelaczam odtwarzanie.")
        return True

    if any(x in norm for x in ("nastepny utwor", "kolejny utwor", "next", "dalej utwor")):
        if spotify.enabled and spotify.next():
            speak_command("Nastepny.")
            return True
        tap_key(VK_MEDIA_NEXT_TRACK)
        speak_command("Nastepny.")
        return True

    if any(x in norm for x in ("poprzedni utwor", "wroc utwor", "previous")):
        if spotify.enabled and spotify.previous():
            speak_command("Poprzedni.")
            return True
        tap_key(VK_MEDIA_PREV_TRACK)
        speak_command("Poprzedni.")
        return True

    return False


# ================== WINDOW / FOLDERS / NOTES / SYSTEM ==================
def handle_window_commands(text: str) -> bool:
    norm = normalize_text(text)
    if "pokaz pulpit" in norm:
        send_hotkey(VK_LWIN, VK_D); speak_command("Pokazuje pulpit."); return True
    if "minimalizuj wszystko" in norm or "zminimalizuj wszystko" in norm:
        send_hotkey(VK_LWIN, VK_M); speak_command("Minimalizuje."); return True
    if "zamknij okno" in norm or "zamknij aktywne okno" in norm:
        send_hotkey(VK_MENU, VK_F4); speak_command("Zamykam okno."); return True
    return False


def handle_folder_commands(text: str) -> bool:
    norm = normalize_text(text)
    folder_map = {
        "pulpit": os.path.join(os.path.expanduser("~"), "Desktop"),
        "pobrane": os.path.join(os.path.expanduser("~"), "Downloads"),
        "dokumenty": os.path.join(os.path.expanduser("~"), "Documents"),
        "obrazy": os.path.join(os.path.expanduser("~"), "Pictures"),
        "muzyka": os.path.join(os.path.expanduser("~"), "Music"),
        "wideo": os.path.join(os.path.expanduser("~"), "Videos"),
    }
    if fuzzy_contains_any(norm, ("otworz folder", "otworz katalog", "pokaz folder"), 0.74):
        for name, path in folder_map.items():
            if name in norm or fuzzy_contains_phrase(norm, name, 0.75):
                if os.path.exists(path):
                    subprocess.Popen(["explorer", path], shell=False)
                    speak_command(f"Otwieram folder {name}.")
                else:
                    speak(f"Brak folderu {name}.")
                return True
    return False


def handle_notes_commands(text: str) -> bool:
    norm = normalize_text(text)
    note = extract_after_phrase(text, ("zapisz notatke", "dodaj notatke", "notatka"))
    if note:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(NOTES_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {note}\n")
        speak_command("Notatka zapisana.")
        return True
    if "otworz notatki" in norm or "pokaz notatki" in norm:
        if not os.path.exists(NOTES_PATH):
            with open(NOTES_PATH, "w", encoding="utf-8") as f:
                f.write("Notatki Jarvisa\n")
        subprocess.Popen(["notepad.exe", NOTES_PATH], shell=False)
        speak_command("Otwieram notatki.")
        return True
    return False


def run_shell(cmd: str):
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:
        enqueue_gui(f"[BLAD shell] {exc}")
        return False


def handle_system_commands(text: str) -> bool:
    norm = normalize_text(text)
    if fuzzy_contains_phrase(norm, "zablokuj komputer", 0.76):
        run_shell("rundll32.exe user32.dll,LockWorkStation"); speak("Blokuje."); return True
    if fuzzy_contains_any(norm, ("uruchom ponownie komputer", "restart komputera"), 0.75):
        run_shell("shutdown /r /t 1"); speak("Restartuje."); return True
    if "wyloguj" in norm:
        run_shell("shutdown /l"); speak("Wylogowuje."); return True
    if "anuluj wylaczenie" in norm or "anuluj shutdown" in norm:
        run_shell("shutdown /a"); speak("Anulowano."); return True
    if "uspij komputer" in norm:
        run_shell("rundll32.exe powrprof.dll,SetSuspendState 0,1,0"); speak("Usypiam."); return True
    return False


def handle_info_commands(text: str) -> bool:
    norm = normalize_text(text)
    if "godzina" in norm:
        speak(f"Jest godzina {datetime.datetime.now().strftime('%H:%M')}."); return True
    if "data" in norm and "dzis" in norm:
        speak(f"Dzisiaj jest {datetime.datetime.now().strftime('%d-%m-%Y')}."); return True
    if fuzzy_contains_phrase(norm, "dzien tygodnia", 0.76):
        days = ["poniedzialek", "wtorek", "sroda", "czwartek", "piatek", "sobota", "niedziela"]
        speak(f"Dzisiaj jest {days[datetime.datetime.now().weekday()]}."); return True
    return False


# ================== MEMORY COMMANDS ==================
def handle_memory_commands(text: str) -> bool:
    norm = normalize_text(text)

    fact = extract_after_phrase(text, ("zapamietaj ze", "zapamietaj że", "zapamietaj", "zapisz fakt", "pamietaj ze", "pamietaj że"))
    if fact:
        if memory.add(fact):
            speak_command("Zapamietalem.")
            enqueue_gui("[PAMIEC] +" + fact, kind="info")
        else:
            speak("To juz pamietam.")
        return True

    if fuzzy_contains_any(norm, ("co pamietasz", "wymien fakty", "pokaz pamiec"), 0.75):
        if not memory.facts:
            speak("Pamiec jest pusta.")
        else:
            preview = "; ".join(memory.facts[:5])
            speak(f"Pamietam: {preview}.")
        return True

    if fuzzy_contains_any(norm, ("zapomnij wszystko", "wyczysc pamiec"), 0.78):
        memory.clear()
        speak_command("Pamiec wyczyszczona.")
        return True

    return False


# ================== MODES ==================
def set_voice_profile(name: str) -> bool:
    if name not in VOICE_PROFILES:
        return False
    state.voice_profile = name
    try:
        engine.setProperty("rate", VOICE_PROFILES[name]["tts_rate"])
    except Exception:
        pass
    save_config()
    return True


def handle_mode_commands(text: str) -> bool:
    norm = normalize_text(text)
    if fuzzy_contains_any(norm, ("tryb turbo", "wlacz turbo"), 0.75):
        set_voice_profile("turbo"); speak("Turbo aktywny."); return True
    if fuzzy_contains_any(norm, ("tryb normalny", "wylacz turbo"), 0.75):
        set_voice_profile("normal"); speak("Normalny aktywny."); return True
    if fuzzy_contains_any(norm, ("tryb rozmowy", "tryb chat"), 0.75):
        state.require_ai_prefix = False; save_config()
        speak("Tryb rozmowy."); return True
    if fuzzy_contains_any(norm, ("tryb komend", "tryb oszczedny"), 0.75):
        state.require_ai_prefix = True; save_config()
        speak("Tryb komend."); return True
    return False


# ================== MAIN HANDLER ==================
def handle_commands(text: str) -> bool:
    if state.awaiting_shutdown:
        norm = normalize_text(text)
        if any(w in norm for w in ("tak", "potwierdzam", "yes")):
            speak("Wylaczam komputer.")
            os.system("shutdown /s /t 1"); state.awaiting_shutdown = False; return True
        if any(w in norm for w in ("nie", "anuluj", "cancel")):
            speak("Anulowano."); state.awaiting_shutdown = False; return True
        speak("Powiedz tak albo nie."); return True

    if "wylacz komputer" in normalize_text(text):
        state.awaiting_shutdown = True
        speak("Czy na pewno wylaczyc komputer? Tak albo nie.")
        return True

    handlers = (
        handle_memory_commands,
        handle_spotify_commands,
        handle_app_control,
        handle_web_commands,
        handle_folder_commands,
        handle_media_commands,
        handle_window_commands,
        handle_notes_commands,
        handle_system_commands,
        handle_info_commands,
        handle_mode_commands,
    )
    for h in handlers:
        try:
            if h(text):
                return True
        except Exception as exc:
            enqueue_gui(f"[BLAD handler {h.__name__}] {exc}")
    return False


def should_ignore_input(cmd: str) -> bool:
    n = normalize_text(cmd)
    if not n or len(n) < MIN_COMMAND_LENGTH:
        return True
    if n in IGNORED_PHRASES:
        return True
    return False


def should_send_to_ai(cmd: str) -> bool:
    n = normalize_text(cmd)
    if not state.require_ai_prefix:
        return True
    return any(n.startswith(p + " ") or n == p for p in AI_PREFIXES)


def strip_ai_prefix(cmd: str) -> str:
    n = normalize_text(cmd)
    for p in AI_PREFIXES:
        if n.startswith(p + " "):
            return cmd[len(p) + 1:].strip()
        if n == p:
            return ""
    return cmd


# ================== LISTENING LOOP (mic kept open) ==================
last_calibration = 0.0


def listen_once(source) -> str:
    global last_calibration
    now = time.monotonic()
    if now - last_calibration > CALIBRATION_INTERVAL_SECONDS:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.25)
        except Exception:
            pass
        last_calibration = now

    try:
        audio = recognizer.listen(
            source, timeout=0.5,
            phrase_time_limit=float(get_profile_value("phrase_time_limit", 8.0)),
        )
    except sr.WaitTimeoutError:
        return ""

    # Mic level update
    try:
        state.mic_level = compute_mic_level(audio.get_raw_data())
        state.last_mic_activity = time.monotonic()
    except Exception:
        pass

    if STT_BACKEND == "whisper" or (STT_BACKEND == "auto" and whisper_model is not None):
        text = transcribe_with_whisper(audio)
        if text:
            return text
        if STT_BACKEND == "whisper":
            return ""
    return transcribe_with_google(audio)


def jarvis_loop():
    """Glowna petla nasluchu. Mikrofon trzymany otwarty caly czas."""
    followup_until = 0.0
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            global last_calibration
            last_calibration = time.monotonic()

            while state.running:
                # Respect mute / PTT / TTS
                if not is_listening_allowed():
                    time.sleep(0.05)
                    continue

                heard = listen_once(source)
                if not heard:
                    time.sleep(0.02)
                    continue

                command = ""
                now = time.monotonic()

                # PTT mode bypassuje wake word
                if state.ptt_mode != "off":
                    command = heard
                elif not state.require_wake_word:
                    command = heard
                elif now < followup_until:
                    command = heard
                elif WAKE_WORD in heard:
                    idx = heard.find(WAKE_WORD)
                    inline = heard[idx + len(WAKE_WORD):].strip(" ,.:;!?-")
                    if inline:
                        command = inline
                    else:
                        speak("Slucham.")
                        command = listen_once(source)
                        if not command:
                            continue
                else:
                    continue

                followup_until = time.monotonic() + FOLLOWUP_WINDOW_SECONDS

                if should_ignore_input(command):
                    continue

                norm_cmd = normalize_text(command)
                if (
                    norm_cmd == state.last_command_text
                    and (time.monotonic() - state.last_command_ts) < COMMAND_DEBOUNCE_SECONDS
                ):
                    continue
                state.last_command_text = norm_cmd
                state.last_command_ts = time.monotonic()

                # Gdy uslyszelismy nowa komende - uciszaj stare TTS i porzucaj stare AI
                if is_speaking.is_set() or not command_queue.empty() or not speech_queue.empty():
                    stop_speaking()
                    while not command_queue.empty():
                        try:
                            command_queue.get_nowait()
                        except queue.Empty:
                            break

                if EXIT_WORD in command:
                    command_queue.put(EXIT_WORD)
                    return

                if handle_commands(command):
                    continue

                if should_send_to_ai(command):
                    ai_cmd = strip_ai_prefix(command)
                    if ai_cmd:
                        while command_queue.qsize() >= MAX_PENDING_AI_COMMANDS:
                            try:
                                command_queue.get_nowait()
                            except queue.Empty:
                                break
                        command_queue.put(ai_cmd)
    except Exception as exc:
        enqueue_gui(f"[KRYTYCZNY BLAD MIC] {exc}")


def assistant_worker():
    while state.running:
        try:
            cmd = command_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if not cmd:
            continue
        if EXIT_WORD in cmd:
            speak("Wylaczam sie.")
            state.running = False
            try:
                root.after(500, root.quit)
            except Exception:
                pass
            return
        ask_ai_streaming(cmd)


# ================== GUI (customtkinter) ==================
USE_CTK = ctk is not None
if USE_CTK:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

import tkinter as tk

if USE_CTK:
    root = ctk.CTk()
else:
    root = tk.Tk()

root.title("JARVIS v2")
root.geometry("1080x680")
root.minsize(900, 600)


def on_close():
    state.running = False
    speech_queue.put(None)
    save_config()
    try:
        wake_detector.cleanup()
    except Exception:
        pass
    try:
        root.quit()
    except Exception:
        pass


root.protocol("WM_DELETE_WINDOW", on_close)


# ===== Layout =====
if USE_CTK:
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # Sidebar
    sidebar = ctk.CTkFrame(root, width=270, corner_radius=0)
    sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
    sidebar.grid_propagate(False)

    title_lbl = ctk.CTkLabel(sidebar, text="JARVIS", font=ctk.CTkFont(size=28, weight="bold"))
    title_lbl.pack(pady=(20, 4), padx=20)
    subtitle_lbl = ctk.CTkLabel(sidebar, text="v2 - Wake + PTT + Streaming", font=ctk.CTkFont(size=11), text_color="#888")
    subtitle_lbl.pack(pady=(0, 20))

    # Status frame
    status_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
    status_frame.pack(fill="x", padx=20, pady=4)
    status_dot = ctk.CTkLabel(status_frame, text="●", font=ctk.CTkFont(size=18), text_color="#22c55e")
    status_dot.pack(side="left")
    status_text = ctk.CTkLabel(status_frame, text="Sluche", font=ctk.CTkFont(size=13))
    status_text.pack(side="left", padx=8)

    # Mic level
    ctk.CTkLabel(sidebar, text="Mikrofon", font=ctk.CTkFont(size=11), text_color="#888").pack(anchor="w", padx=20, pady=(15, 2))
    mic_bar = ctk.CTkProgressBar(sidebar, height=8)
    mic_bar.pack(fill="x", padx=20)
    mic_bar.set(0)

    # Voice profile
    ctk.CTkLabel(sidebar, text="Profil glosu", font=ctk.CTkFont(size=11), text_color="#888").pack(anchor="w", padx=20, pady=(20, 2))
    profile_var = tk.StringVar(value=state.voice_profile)
    def on_profile_change(value):
        set_voice_profile(value)
        enqueue_gui(f"[PROFIL] {value}", kind="info")
    profile_menu = ctk.CTkSegmentedButton(sidebar, values=["turbo", "normal"], variable=profile_var, command=on_profile_change)
    profile_menu.pack(fill="x", padx=20)

    # AI mode
    ctk.CTkLabel(sidebar, text="Tryb AI", font=ctk.CTkFont(size=11), text_color="#888").pack(anchor="w", padx=20, pady=(15, 2))
    ai_mode_var = tk.StringVar(value="rozmowa" if not state.require_ai_prefix else "komendy")
    def on_ai_mode_change(value):
        state.require_ai_prefix = (value == "komendy")
        save_config()
    ai_mode_menu = ctk.CTkSegmentedButton(sidebar, values=["rozmowa", "komendy"], variable=ai_mode_var, command=on_ai_mode_change)
    ai_mode_menu.pack(fill="x", padx=20)

    # PTT
    ctk.CTkLabel(sidebar, text="Push-to-Talk", font=ctk.CTkFont(size=11), text_color="#888").pack(anchor="w", padx=20, pady=(15, 2))
    ptt_var = tk.StringVar(value=state.ptt_mode)
    def on_ptt_change(value):
        state.ptt_mode = value
        save_config()
        enqueue_gui(f"[PTT] {value}", kind="info")
    ptt_menu = ctk.CTkOptionMenu(sidebar, variable=ptt_var, values=list(PTT_MODES), command=on_ptt_change)
    ptt_menu.pack(fill="x", padx=20)

    # Wake word
    wake_var = tk.BooleanVar(value=state.require_wake_word)
    def on_wake_toggle():
        state.require_wake_word = bool(wake_var.get())
        save_config()
    wake_cb = ctk.CTkCheckBox(sidebar, text="Wymagaj 'jarvis'", variable=wake_var, command=on_wake_toggle)
    wake_cb.pack(anchor="w", padx=20, pady=(15, 4))

    # Mic mute
    mute_var = tk.BooleanVar(value=False)
    def on_mute_toggle():
        state.mic_muted = bool(mute_var.get())
    mute_cb = ctk.CTkCheckBox(sidebar, text="Wycisz mikrofon", variable=mute_var, command=on_mute_toggle)
    mute_cb.pack(anchor="w", padx=20, pady=4)

    # Buttons
    btn_stop = ctk.CTkButton(sidebar, text="Zatrzymaj mowienie", command=lambda: stop_speaking())
    btn_stop.pack(fill="x", padx=20, pady=(15, 4))
    btn_clear_chat = ctk.CTkButton(sidebar, text="Wyczysc historie", fg_color="#444",
                                    command=lambda: (state.chat_history.__setitem__(slice(1, None), []), enqueue_gui("[CHAT] wyczyszczono", kind="info")))
    btn_clear_chat.pack(fill="x", padx=20, pady=4)

    btn_exit = ctk.CTkButton(sidebar, text="Zamknij", fg_color="#7f1d1d", hover_color="#991b1b", command=on_close)
    btn_exit.pack(fill="x", padx=20, pady=(20, 12), side="bottom")

    # Main content
    main = ctk.CTkFrame(root, fg_color="transparent")
    main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
    main.grid_columnconfigure(0, weight=2)
    main.grid_columnconfigure(1, weight=1)
    main.grid_rowconfigure(0, weight=1)

    # Conversation area
    conv_frame = ctk.CTkFrame(main)
    conv_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    ctk.CTkLabel(conv_frame, text="Rozmowa", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=14, pady=(10, 4))

    text_area = ctk.CTkTextbox(conv_frame, font=ctk.CTkFont(family="Consolas", size=12), wrap="word")
    text_area.pack(fill="both", expand=True, padx=10, pady=10)
    text_area.tag_config("user", foreground="#60a5fa")
    text_area.tag_config("bot", foreground="#34d399")
    text_area.tag_config("info", foreground="#fbbf24")
    text_area.tag_config("err", foreground="#f87171")

    # Memory area
    mem_frame = ctk.CTkFrame(main)
    mem_frame.grid(row=0, column=1, sticky="nsew")
    ctk.CTkLabel(mem_frame, text="Pamiec dlugoterminowa", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=14, pady=(10, 4))

    mem_listbox = tk.Listbox(mem_frame, bg="#1a1a1a", fg="#e5e5e5", font=("Consolas", 10),
                             selectbackground="#3b82f6", borderwidth=0, highlightthickness=0)
    mem_listbox.pack(fill="both", expand=True, padx=10, pady=(4, 8))

    mem_entry = ctk.CTkEntry(mem_frame, placeholder_text="Dodaj fakt...")
    mem_entry.pack(fill="x", padx=10, pady=4)

    def refresh_memory_list():
        mem_listbox.delete(0, tk.END)
        for f in memory.facts:
            mem_listbox.insert(tk.END, f)

    def add_memory_fact():
        v = mem_entry.get().strip()
        if v and memory.add(v):
            mem_entry.delete(0, tk.END)
            refresh_memory_list()
            enqueue_gui("[PAMIEC] +" + v, kind="info")

    def remove_selected_fact():
        sel = mem_listbox.curselection()
        if sel:
            memory.remove(sel[0])
            refresh_memory_list()

    mem_btn_frame = ctk.CTkFrame(mem_frame, fg_color="transparent")
    mem_btn_frame.pack(fill="x", padx=10, pady=(0, 10))
    ctk.CTkButton(mem_btn_frame, text="+ Dodaj", width=80, command=add_memory_fact).pack(side="left", padx=2)
    ctk.CTkButton(mem_btn_frame, text="- Usun", width=80, fg_color="#7f1d1d", hover_color="#991b1b", command=remove_selected_fact).pack(side="left", padx=2)

    refresh_memory_list()

    # Bottom status bar
    bottom = ctk.CTkFrame(root, height=30, corner_radius=0)
    bottom.grid(row=1, column=1, sticky="ew", padx=0, pady=0)
    bottom_lbl = ctk.CTkLabel(bottom, text="Gotowy", font=ctk.CTkFont(size=11), text_color="#888")
    bottom_lbl.pack(side="left", padx=12, pady=4)
    integ_lbl = ctk.CTkLabel(bottom, text="", font=ctk.CTkFont(size=11), text_color="#888")
    integ_lbl.pack(side="right", padx=12, pady=4)

else:
    # Plain tkinter fallback
    root.configure(bg="black")
    text_area = tk.Text(root, bg="black", fg="lime", font=("Consolas", 10), wrap="word")
    text_area.pack(expand=True, fill="both")
    text_area.tag_config("user", foreground="#60a5fa")
    text_area.tag_config("bot", foreground="#34d399")
    text_area.tag_config("info", foreground="#fbbf24")
    text_area.tag_config("err", foreground="#f87171")
    bottom_lbl = tk.Label(root, text="Status: aktywny", bg="black", fg="cyan", anchor="w", padx=8)
    bottom_lbl.pack(fill="x")
    mic_bar = None
    status_dot = None
    status_text = None
    integ_lbl = None


def append_log(line: str, kind: str):
    tag_map = {"user": "user", "bot": "bot", "info": "info", "err": "err", "log": None}
    tag = tag_map.get(kind)
    try:
        if USE_CTK:
            text_area.insert("end", line + "\n", tag if tag else None)
            text_area.see("end")
            # Rotate: keep last ~600 lines
            line_count = int(text_area.index("end-1c").split(".")[0])
            if line_count > 600:
                text_area.delete("1.0", f"{line_count - 500}.0")
        else:
            text_area.insert(tk.END, line + "\n", tag if tag else ())
            text_area.see(tk.END)
    except Exception:
        pass


def process_gui_queue():
    while True:
        try:
            kind, text = gui_queue.get_nowait()
        except queue.Empty:
            break
        # Detect error type
        k = kind
        if "[BLAD" in text:
            k = "err"
        append_log(text, k)
    root.after(80, process_gui_queue)


def update_status_loop():
    """Aktualizuje pasek mikrofonu, status LED i info o integracjach."""
    if USE_CTK:
        # Mic level z lekkim wygaszaniem
        level = state.mic_level
        try:
            mic_bar.set(level)
        except Exception:
            pass
        # Wygaszanie levelu
        state.mic_level = max(0.0, state.mic_level - 0.04)

        # Status dot
        if state.mic_muted:
            status_dot.configure(text_color="#ef4444"); status_text.configure(text="Mikrofon wyciszony")
        elif is_speaking.is_set():
            status_dot.configure(text_color="#fbbf24"); status_text.configure(text="Mowie...")
        elif state.ptt_mode != "off" and not ptt_pressed.is_set():
            status_dot.configure(text_color="#94a3b8"); status_text.configure(text=f"PTT czeka ({state.ptt_mode})")
        elif ptt_pressed.is_set():
            status_dot.configure(text_color="#22c55e"); status_text.configure(text="Sluche (PTT)")
        else:
            status_dot.configure(text_color="#22c55e"); status_text.configure(text="Sluche")

        # Integration label
        bits = []
        bits.append("Whisper" if whisper_model else "Google STT")
        bits.append("Porcupine" if wake_detector.enabled else "wake-fb")
        bits.append("Spotify" if spotify.enabled else "media-keys")
        bits.append("Ollama+OpenAI" if OPENAI_API_KEY else "Ollama")
        try:
            integ_lbl.configure(text=" | ".join(bits))
        except Exception:
            pass

    root.after(100, update_status_loop)


# ================== STARTUP ==================
def startup():
    load_config()
    if STT_BACKEND in ("auto", "whisper"):
        init_local_stt()
    set_voice_profile(state.voice_profile)
    spotify.init()
    wake_detector.init()
    ptt.start()

    threading.Thread(target=speaker_worker, daemon=True).start()
    threading.Thread(target=assistant_worker, daemon=True).start()
    threading.Thread(target=jarvis_loop, daemon=True).start()
    threading.Thread(target=warmup_ollama, daemon=True).start()
    threading.Thread(target=warmup_tts, daemon=True).start()

    speak(
        f"Jarvis v2 uruchomiony. Profil {state.voice_profile}. "
        f"{'Tryb komend.' if state.require_ai_prefix else 'Tryb rozmowy.'}"
        f"{' PTT ' + state.ptt_mode + '.' if state.ptt_mode != 'off' else ''}"
    )


root.after(100, process_gui_queue)
root.after(100, update_status_loop)
root.after(200, startup)

try:
    root.mainloop()
finally:
    state.running = False
    speech_queue.put(None)
    save_config()
    wake_detector.cleanup()
