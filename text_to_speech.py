"""
Text-to-Speech wrapper for SignBridge.
Provides backward-compatible interface backed by the new LinguisticTTS engine,
supporting Telugu, Hindi, English, and other regional linguistic voices.
"""

from linguistic_tts import LinguisticTTS, SUPPORTED_LANGUAGES

class TextToSpeech(LinguisticTTS):
    """
    Drop-in replacement for the original TextToSpeech class.
    Inherits all linguistic voice capabilities (Telugu, Hindi, English).
    """
    def __init__(self, default_lang: str = "hi", speech_rate: int = 110):
        super().__init__(default_lang=default_lang, speech_rate=speech_rate)
