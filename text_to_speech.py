import pyttsx3


class TextToSpeech:

    def __init__(self):
        self.rate = 100
        self.volume = 1.0

        voices = pyttsx3.init().getProperty("voices")

        self.voice_id = voices[0].id

        print("TTS initialized")
        print("Voice:", voices[0].name)
        print("Rate:", self.rate)

    def speak(self, text):

        if not text:
            return

        text = str(text).strip()

        if not text:
            return

        print(f"Speaking: {text}")

        # Create a fresh engine for every sentence
        engine = pyttsx3.init()

        engine.setProperty("voice", self.voice_id)
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        engine.say(text)
        engine.runAndWait()
        engine.stop()