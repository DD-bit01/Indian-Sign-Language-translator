"""
Real-time English-to-Indian Language Translator for SignBridge.
Translates detected English sign language text into Hindi, Telugu, Tamil, etc.,
before feeding into linguistic Text-to-Speech engines.
"""

import sys
import requests
from typing import Dict, Optional

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Offline fallback dictionary for common sign phrases
OFFLINE_DICTIONARY: Dict[str, Dict[str, str]] = {
    "hello": {"hi": "नमस्ते", "te": "నమస్కారం", "ta": "வணக்கம்"},
    "hi": {"hi": "नमस्ते", "te": "హాయ్", "ta": "வணக்கம்"},
    "good morning": {"hi": "शुभ प्रभात", "te": "శుభోదయం", "ta": "காலை வணக்கம்"},
    "good afternoon": {"hi": "शुभ दोपहर", "te": "శుభ మధ్యాహ్నం", "ta": "மதிய வணக்கம்"},
    "good evening": {"hi": "शुभ संध्या", "te": "శుభ సాయంత్రం", "ta": "மாலை வணக்கம்"},
    "good night": {"hi": "शुभ रात्रि", "te": "శుభరాత్రి", "ta": "இரவு வணக்கம்"},
    "how are you": {"hi": "आप कैसे हैं", "te": "మీరు ఎలా ఉన్నారు", "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்"},
    "i am fine": {"hi": "मैं ठीक हूँ", "te": "నేను బాగున్నాను", "ta": "நான் நலமாக இருக்கிறேன்"},
    "thank you": {"hi": "धन्यवाद", "te": "ధన్యవాదాలు", "ta": "நன்றி"},
    "thanks": {"hi": "धन्यवाद", "te": "ధన్యవాదాలు", "ta": "நன்றி"},
    "welcome": {"hi": "स्वागत हे", "te": "స్వాగతం", "ta": "வரவேற்பு"},
    "please": {"hi": "कृपया", "te": "దయచేసి", "ta": "தயவுசெய்து"},
    "sorry": {"hi": "माफ़ कीजिये", "te": "నన్ను క్షమించండి", "ta": "மன்னிக்கவும்"},
    "yes": {"hi": "हाँ", "te": "అవును", "ta": "ஆம்"},
    "no": {"hi": "नहीं", "te": "లేదు", "ta": "இல்லை"},
    "water": {"hi": "पानी", "te": "నీరు", "ta": "தண்ணீர்"},
    "food": {"hi": "खाना", "te": "ఆహారం", "ta": "உணவு"},
    "home": {"hi": "घर", "te": "ఇల్లు", "ta": "வீடு"},
    "help": {"hi": "मदद", "te": "సహాయం", "ta": "உதவி"},
    "i need water": {"hi": "मुझे पानी चाहिए", "te": "నాకు నీరు కావాలి", "ta": "எனக்கு தண்ணீர் வேண்டும்"},
    "i want food": {"hi": "मुझे खाना चाहिए", "te": "నాకు ఆహారం కావాలి", "ta": "எனக்கு உணவு வேண்டும்"},
    "where is the hospital": {"hi": "अस्पताल कहाँ है", "te": "ఆసుపత్రి ఎక్కడ ఉంది", "ta": "மருத்துவமனை எங்கே உள்ளது"},
    "where is the washroom": {"hi": "शौचालय कहाँ है", "te": "వాష్‌రూమ్ ఎక్కడ ఉంది", "ta": "கழிப்பறை எங்கே உள்ளது"},
    "please call a doctor": {"hi": "कृपया डॉक्टर को बुलाएँ", "te": "దయచేసి డాక్టర్‌ని పిలవండి", "ta": "தயவுசெய்து மருத்துவரை அழைக்கவும்"},
}


class LanguageTranslator:
    """
    Translates English sign-language words and sentences to Hindi, Telugu, etc.
    Uses Google's public translation endpoint with an in-memory cache and offline dictionary fallback.
    """

    def __init__(self):
        # In-memory translation cache: (text, target_lang) -> translated_text
        self._cache: Dict[str, str] = {}
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

    def translate(self, text: str, target_lang: str = "hi", source_lang: str = "en") -> str:
        """
        Translate input text to the target language (e.g. 'hi', 'te', 'en').
        If target is already English ('en'), returns text unmodified.
        """
        if not text:
            return ""

        text = str(text).strip()
        target_lang = target_lang.lower().strip()

        # If target language is English, no translation needed
        if target_lang == "en" or not target_lang:
            return text

        cache_key = f"{target_lang}:{text.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check offline dictionary first for instant response
        clean_key = text.lower().strip(".,!? ")
        if clean_key in OFFLINE_DICTIONARY and target_lang in OFFLINE_DICTIONARY[clean_key]:
            result = OFFLINE_DICTIONARY[clean_key][target_lang]
            self._cache[cache_key] = result
            return result

        # Try Google Translate public endpoint
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": source_lang,
                "tl": target_lang,
                "dt": "t",
                "q": text,
            }
            resp = self._session.get(url, params=params, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                translated_parts = [part[0] for part in data[0] if part and part[0]]
                translated = "".join(translated_parts).strip()
                if translated:
                    self._cache[cache_key] = translated
                    return translated
        except Exception as e:
            # Silently fallback to word-by-word offline lookup or original
            pass

        # Fallback: check individual words in offline dictionary
        words = text.lower().split()
        translated_words = []
        has_any_match = False
        for w in words:
            cw = w.strip(".,!? ")
            if cw in OFFLINE_DICTIONARY and target_lang in OFFLINE_DICTIONARY[cw]:
                translated_words.append(OFFLINE_DICTIONARY[cw][target_lang])
                has_any_match = True
            else:
                translated_words.append(w)

        if has_any_match:
            result = " ".join(translated_words)
            self._cache[cache_key] = result
            return result

        # Return original text if translation unavailable
        return text


# Global singleton instance
_default_translator = LanguageTranslator()


def translate_text(text: str, target_lang: str = "hi", source_lang: str = "en") -> str:
    """Helper function to translate text using the default translator."""
    return _default_translator.translate(text, target_lang=target_lang, source_lang=source_lang)


if __name__ == "__main__":
    t = LanguageTranslator()
    test_phrases = [
        "hello",
        "how are you",
        "i am dilnawaz",
        "i need water please",
        "thank you very much",
        "where is the hospital",
    ]

    print("Testing LanguageTranslator...")
    for phrase in test_phrases:
        hi = t.translate(phrase, target_lang="hi")
        te = t.translate(phrase, target_lang="te")
        print(f"\nEnglish: {phrase}")
        print(f"  Hindi : {hi}")
        print(f"  Telugu: {te}")
