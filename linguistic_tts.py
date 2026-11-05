"""
Linguistic Text-To-Speech (TTS) module for SignBridge.
Supports natural Indian linguistic voices (Telugu, Hindi, English, Tamil, etc.)
with automatic translation from English sign language into vernacular speech,
non-blocking playback, and automatic offline fallback.
"""

import io
import os
import sys
import time
import queue
import hashlib
import logging
import threading
from typing import Optional, Dict

# Safe utf-8 console output for Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Silence pygame welcome banner
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

from translator import LanguageTranslator


# Supported languages with display names
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "te": "Telugu (తెలుగు)",
    "hi": "Hindi (हिन्दी)",
    "en": "English",
    "ta": "Tamil (தமிழ்)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "mr": "Marathi (मराठी)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "ur": "Urdu (اردو)",
}


class LinguisticTTS:
    """
    Multi-language TTS supporting Telugu, Hindi, English, and more.
    Translates English sign words/sentences into the target Indian language
    prior to synthesis so speech sounds authentic and natural.
    """

    def __init__(self, default_lang: str = "hi", speech_rate: int = 110):
        self.default_lang = default_lang if default_lang in SUPPORTED_LANGUAGES else "hi"
        self.current_lang = self.default_lang
        self.speech_rate = speech_rate
        self.is_speaking = False

        # Translator instance
        self.translator = LanguageTranslator()

        # In-memory audio cache: hash(lang + spoken_text) -> bytes
        self._audio_cache: Dict[str, bytes] = {}

        # Background speech queue
        self._speech_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # Initialize pygame mixer
        self._mixer_initialized = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self._mixer_initialized = True
            except Exception as e:
                print(f"[LinguisticTTS] Pygame mixer init failed: {e}")

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

        print(f"[LinguisticTTS] Initialized with language: {self.get_language_name(self.current_lang)}")

    # --------------------------------------------------------
    # Language Management
    # --------------------------------------------------------
    def list_languages(self) -> Dict[str, str]:
        """Return available languages dictionary: {code: name}."""
        return dict(SUPPORTED_LANGUAGES)

    def set_language(self, lang_code: str) -> bool:
        """Set active language code (e.g., 'te', 'hi', 'en')."""
        lang_code = lang_code.lower().strip()
        if lang_code in SUPPORTED_LANGUAGES:
            self.current_lang = lang_code
            print(f"[LinguisticTTS] Active voice language changed to: {self.get_language_name(lang_code)}")
            return True
        print(f"[LinguisticTTS] Unsupported language '{lang_code}'. Available: {list(SUPPORTED_LANGUAGES.keys())}")
        return False

    def cycle_language(self, options=("hi", "te", "en")) -> str:
        """Cycle through a preferred subset of languages (e.g. Hindi -> Telugu -> English)."""
        valid_opts = [opt for opt in options if opt in SUPPORTED_LANGUAGES]
        if not valid_opts:
            valid_opts = list(SUPPORTED_LANGUAGES.keys())

        if self.current_lang in valid_opts:
            idx = (valid_opts.index(self.current_lang) + 1) % len(valid_opts)
            self.current_lang = valid_opts[idx]
        else:
            self.current_lang = valid_opts[0]

        print(f"[LinguisticTTS] Cycled voice to: {self.get_language_name(self.current_lang)}")
        return self.current_lang

    def get_language(self) -> str:
        """Return current language code."""
        return self.current_lang

    def get_language_name(self, lang_code: Optional[str] = None) -> str:
        """Return human-readable display name."""
        code = lang_code or self.current_lang
        return SUPPORTED_LANGUAGES.get(code, code)

    def translate(self, text: str, lang: Optional[str] = None) -> str:
        """Translate text to the target language without speaking."""
        target_lang = lang if (lang and lang in SUPPORTED_LANGUAGES) else self.current_lang
        return self.translator.translate(text, target_lang=target_lang)

    # --------------------------------------------------------
    # Speaking
    # --------------------------------------------------------
    def speak(self, text: str, lang: Optional[str] = None, block: bool = False):
        """
        Speak text. Automatically translates English text into the target
        vernacular language (Hindi, Telugu, etc.) prior to speaking.
        """
        if not text:
            return

        text = str(text).strip()
        if not text:
            return

        target_lang = lang if (lang and lang in SUPPORTED_LANGUAGES) else self.current_lang

        if block:
            self._execute_speak(text, target_lang)
        else:
            self._speech_queue.put((text, target_lang))

    def _speech_worker(self):
        """Worker thread processing speech queue sequentially."""
        while not self._stop_event.is_set():
            try:
                item = self._speech_queue.get(timeout=0.2)
                if item is None:
                    break
                text, lang = item
                self.is_speaking = True
                self._execute_speak(text, lang)
                self.is_speaking = False
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.is_speaking = False
                print(f"[LinguisticTTS] Worker error: {e}")

    def _execute_speak(self, text: str, lang: str):
        """Translate text if needed, then synthesize and play audio."""
        # 1. Translate English into target language (Hindi, Telugu, etc.)
        spoken_text = text
        if lang != "en":
            try:
                translated = self.translator.translate(text, target_lang=lang)
                if translated:
                    spoken_text = translated
            except Exception as trans_err:
                print(f"[LinguisticTTS] Translation note: {trans_err}")

        print(f"[LinguisticTTS] [{self.get_language_name(lang)}] English: '{text}' -> Spoken: '{spoken_text}'")

        # 2. Try online gTTS with the translated vernacular text
        if GTTS_AVAILABLE and self._mixer_initialized:
            try:
                cache_key = f"{lang}_{spoken_text}"
                if cache_key in self._audio_cache:
                    audio_data = self._audio_cache[cache_key]
                else:
                    fp = io.BytesIO()
                    tts = gTTS(text=spoken_text, lang=lang, slow=False)
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    audio_data = fp.read()
                    if len(self._audio_cache) < 150:
                        self._audio_cache[cache_key] = audio_data

                # Play via pygame mixer
                sound_fp = io.BytesIO(audio_data)
                pygame.mixer.music.load(sound_fp)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(20)
                return
            except Exception as err:
                print(f"[LinguisticTTS] gTTS playback failed ({err}). Falling back to offline TTS...")

        # 3. Offline fallback: pyttsx3 (speaks original English or system voice)
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.speech_rate)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as err:
                print(f"[LinguisticTTS] pyttsx3 fallback failed: {err}")
        else:
            print("[LinguisticTTS] No TTS engine available.")

    def stop(self):
        """Stop playback and background worker."""
        self._stop_event.set()
        self._speech_queue.put(None)
        if self._mixer_initialized and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()


# Backward compatibility alias
TextToSpeech = LinguisticTTS


if __name__ == "__main__":
    print("Testing LinguisticTTS with translation...")
    tts = LinguisticTTS()

    print("\n1. Testing Hindi with English input ('how are you'):")
    tts.speak("how are you", lang="hi", block=True)

    print("\n2. Testing Telugu with English input ('how are you'):")
    tts.speak("how are you", lang="te", block=True)

    print("\n3. Testing Hindi with English input ('thank you very much'):")
    tts.speak("thank you very much", lang="hi", block=True)

    print("\n4. Testing Telugu with English input ('thank you very much'):")
    tts.speak("thank you very much", lang="te", block=True)

    print("\nTests completed successfully.")
