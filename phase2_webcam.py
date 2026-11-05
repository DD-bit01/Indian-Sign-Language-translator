import os
import sys
import time
import cv2
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from collections import Counter, deque
import concurrent.futures

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from word_builder import WordBuilder
from word_suggester import WordSuggester
from linguistic_tts import LinguisticTTS
from sentence_predictor import SentencePredictor


# ============================================================
# UNICODE TEXT RENDERING FOR HINDI & TELUGU ON OPENCV
# Optimized with ROI cropping (18x speedup over full-frame conversion)
# ============================================================
UNICODE_FONT = None
try:
    font_path = "C:/Windows/Fonts/Nirmala.ttc"
    if os.path.exists(font_path):
        UNICODE_FONT = ImageFont.truetype(font_path, 20)
except Exception:
    UNICODE_FONT = None


def render_unicode_overlay(frame, text, pos=(20, 180), color_rgb=(0, 255, 255)):
    """Fast localized ROI-based Unicode text rendering for Hindi & Telugu."""
    if not text:
        return frame
    if UNICODE_FONT is None:
        # Fallback to plain putText if font unavailable
        safe_text = text.encode("ascii", "replace").decode("ascii")
        cv2.putText(frame, safe_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        return frame

    try:
        fh, fw = frame.shape[:2]
        x, y = pos
        # Localized bounding box for single-line text banner (avoids copying entire frame)
        x1 = max(0, x - 5)
        y1 = max(0, y - 5)
        x2 = min(fw, x + 680)
        y2 = min(fh, y + 36)

        if x2 <= x1 or y2 <= y1:
            return frame

        roi = frame[y1:y2, x1:x2]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(roi_pil)
        draw.text((x - x1, y - y1), text, font=UNICODE_FONT, fill=color_rgb)
        frame[y1:y2, x1:x2] = cv2.cvtColor(np.array(roi_pil), cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        return frame


# ============================================================
# SIGN HAND-COUNT RULES
# ============================================================
TWO_HAND_CLASSES = {
    "A", "B", "D", "E", "F", "G", "H", "J", "K", "M",
    "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z",
}

ONE_HAND_CLASSES = {
    "C", "I", "L", "O", "U", "V",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
}


def validate_hand_count(label, detected_hands):
    """Reject labels whose required number of hands is not detected."""
    label = str(label).upper()

    if detected_hands < 1:
        return False, "No hand detected"
    if label in TWO_HAND_CLASSES and detected_hands < 2:
        return False, f"{label} needs BOTH hands"
    if label in ONE_HAND_CLASSES and detected_hands != 1:
        return False, f"{label} needs ONE hand"
    return True, "OK"


# ============================================================
# LOAD MODELS
# ============================================================
MODEL_PATH = "model_final.p"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("[Model] Random Forest ISL classifier loaded.")

# Load Sentence Predictor (GRU)
SENTENCE_MODEL_PATH = "model_sentence_gru.pt"
if os.path.exists(SENTENCE_MODEL_PATH):
    sentence_predictor = SentencePredictor.load(SENTENCE_MODEL_PATH)
    print(f"[Model] GRU Sentence Predictor loaded from {SENTENCE_MODEL_PATH}.")
else:
    print("[Model] Training initial GRU Sentence Predictor...")
    sentence_predictor = SentencePredictor(rnn_type="gru")
    sentence_predictor.train(epochs=120)
    sentence_predictor.save(SENTENCE_MODEL_PATH)


# ============================================================
# MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5,
    model_complexity=0,
)


# ============================================================
# RUNTIME OBJECTS
# ============================================================
builder = WordBuilder(cooldown=0.7)
suggester = WordSuggester()

# Linguistic TTS: starts with Hindi ('hi'), easily toggled to Telugu ('te') or English ('en')
tts = LinguisticTTS(default_lang="hi")

# Background thread executor for non-blocking real-time translation (eliminates webcam lag)
_trans_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="TransWorker")
_trans_future = None

last_processed_text = None
last_processed_lang = None
cached_suggestion = ""
cached_next_word_preds = []
cached_completed_sentence = ""
cached_translated_preview = ""


# ============================================================
# PREDICTION SMOOTHING
# ============================================================
prediction_history = deque(maxlen=7)
stable_prediction = ""
validation_message = "Show your sign"
confidence = 0.0

# Special sign 2 state (triggers space + speak)
special_sign_active = False


# ============================================================
# SENTENCE CORRECTION
# ============================================================
def correct_sentence(text, word_suggester):
    if not text:
        return ""

    words = text.strip().split()
    corrected_words = []

    for word in words:
        suggestions = word_suggester.suggest(word, max_suggestions=1)
        if suggestions:
            corrected_words.append(suggestions[0])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# ============================================================
# SPEAK CURRENT WORD
# ============================================================
def speak_current_word():
    current_text = builder.word.strip()
    if not current_text:
        return

    current_word = current_text.split()[-1]
    suggestions = suggester.suggest(current_word, max_suggestions=1)
    word_to_speak = suggestions[0] if suggestions else current_word

    print(f"Speaking word [{tts.get_language_name()}]: {word_to_speak}")
    tts.speak(word_to_speak)


# ============================================================
# OPEN WEBCAM (DirectShow on Windows for maximum stability)
# ============================================================
def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass
    return cap

cap = open_camera()

if not cap.isOpened():
    hands.close()
    raise RuntimeError("ERROR: Could not open webcam.")


# ============================================================
# START MESSAGE
# ============================================================
print()
print("==========================================================")
print(" ISL TRANSLATOR - WITH VERNACULAR TRANSLATION & GRU")
print("==========================================================")
print("Active Voice Language:", tts.get_language_name())
print("Hand-count rules: Enabled (Two-hand vs One-hand)")
print("Sentence Predictor: GRU Recurrent Neural Network")
print()
print("Keyboard Controls:")
print("L     -> Cycle Voice Language (Hindi -> Telugu -> English)")
print("P     -> Auto-accept predicted sentence completion")
print("D     -> Delete last character")
print("C     -> Clear text")
print("SPACE -> Add space + speak current word")
print("S     -> Speak translated complete sentence")
print("Q     -> Quit")
print()

consecutive_drops = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_drops += 1
            if consecutive_drops % 15 == 0:
                print(f"[Webcam] Frame drop detected ({consecutive_drops}). Re-opening device...")
                cap.release()
                time.sleep(0.5)
                cap = open_camera()
            if consecutive_drops > 90:
                print("Failed to re-establish webcam feed after multiple attempts. Exiting.")
                break
            time.sleep(0.03)
            continue
        consecutive_drops = 0

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected_hand_count = 0

        if results.multi_hand_landmarks:
            detected_hands = results.multi_hand_landmarks
            detected_hand_count = len(detected_hands)

            # Sort detected hands left-to-right based on wrist x-coordinate
            detected_hands = sorted(
                detected_hands,
                key=lambda hand: hand.landmark[0].x
            )

            data_aux = []

            for hand_landmarks in detected_hands[:2]:
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                min_x = min(x_coords)
                max_x = max(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)

                hand_width = max(max_x - min_x, 1e-6)
                hand_height = max(max_y - min_y, 1e-6)

                for landmark in hand_landmarks.landmark:
                    normalized_x = (landmark.x - min_x) / hand_width
                    normalized_y = (landmark.y - min_y) / hand_height
                    data_aux.extend([normalized_x, normalized_y])

            # Zero-padding for 84 features (2 hands x 21 landmarks x 2)
            while len(data_aux) < 84:
                data_aux.append(0.0)
            data_aux = data_aux[:84]

            features = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)

            # Random Forest classification
            prediction = str(model.predict(features)[0]).upper()
            probabilities = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities) * 100)

            # Hand-count validation
            accepted, rule_message = validate_hand_count(prediction, detected_hand_count)

            if not accepted:
                prediction_history.clear()
                stable_prediction = ""
                special_sign_active = False
                builder.reset_sign()
                validation_message = rule_message
            else:
                validation_message = f"Hands detected: {detected_hand_count}"
                prediction_history.append(prediction)

                counts = Counter(prediction_history)
                most_common_prediction, count = counts.most_common(1)[0]

                # Require at least 4 of the last 7 accepted frames
                if count >= 4:
                    current_prediction = most_common_prediction

                    if current_prediction == "2":
                        if not special_sign_active:
                            special_sign_active = True
                            speak_current_word()
                            builder.add_space()
                            print("Special sign 2 -> SPACE + Speak")
                        stable_prediction = "2"
                    else:
                        special_sign_active = False
                        if current_prediction != stable_prediction:
                            stable_prediction = current_prediction
                            builder.add_prediction(stable_prediction)

            # Draw landmarks
            for hand_landmarks in detected_hands[:2]:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )
        else:
            prediction_history.clear()
            stable_prediction = ""
            special_sign_active = False
            builder.reset_sign()
            validation_message = "Show your sign"

        # ----------------------------------------------------
        # TEXT, SUGGESTIONS & GRU SENTENCE PREDICTOR (NON-BLOCKING & CACHED)
        # Evaluated ONLY on text/language changes to maintain 60 FPS without lag
        # ----------------------------------------------------
        detected_text = builder.word
        active_lang = tts.get_language()

        if detected_text != last_processed_text or active_lang != last_processed_lang:
            last_processed_text = detected_text
            last_processed_lang = active_lang

            cached_suggestion = correct_sentence(detected_text, suggester)
            effective_txt = (cached_suggestion if cached_suggestion else detected_text).strip()

            if effective_txt:
                try:
                    cached_next_word_preds = sentence_predictor.predict_next_words(effective_txt, top_k=2)
                    cached_completed_sentence = sentence_predictor.complete_sentence(effective_txt, max_words=3)
                except Exception:
                    cached_next_word_preds = []
                    cached_completed_sentence = ""
            else:
                cached_next_word_preds = []
                cached_completed_sentence = ""

            # Check translation cache or dispatch background async task
            if effective_txt and active_lang != "en":
                cache_key = f"{active_lang}:{effective_txt.lower()}"
                if cache_key in tts.translator._cache:
                    cached_translated_preview = tts.translator._cache[cache_key]
                    _trans_future = None
                else:
                    _trans_future = _trans_executor.submit(tts.translate, effective_txt, active_lang)
            else:
                cached_translated_preview = ""
                _trans_future = None

        # Non-blocking poll for finished background translation
        if _trans_future is not None and _trans_future.done():
            try:
                cached_translated_preview = _trans_future.result()
            except Exception:
                pass
            _trans_future = None

        suggestion = cached_suggestion
        effective_text = cached_suggestion if cached_suggestion else detected_text
        next_word_preds = cached_next_word_preds
        completed_sentence = cached_completed_sentence
        translated_preview = cached_translated_preview

        # ----------------------------------------------------
        # HUD DISPLAY OVERLAYS
        # ----------------------------------------------------
        # Prediction & Confidence
        cv2.putText(
            frame,
            f"Sign: {stable_prediction or '-'}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
        )

        if detected_hand_count:
            cv2.putText(
                frame,
                f"Conf: {confidence:.0f}%",
                (200, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Voice status
        lang_short = active_lang.upper()
        cv2.putText(
            frame,
            f"Voice: [{lang_short}] {tts.get_language_name()[:10]} (L: Cycle)",
            (360, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 50),
            2,
        )

        # Validation rule status
        cv2.putText(
            frame,
            validation_message[:55],
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255) if validation_message not in ("Show your sign",) and "Hands detected" not in validation_message else (200, 200, 200),
            2,
        )

        # Detected English text
        cv2.putText(
            frame,
            f"English: {detected_text}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        # Fuzzy spell suggestion
        if suggestion and suggestion.lower() != detected_text.lower().strip():
            cv2.putText(
                frame,
                f"Corrected: {suggestion}"[:55],
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

        # Vernacular Translation in native Hindi / Telugu script!
        if translated_preview:
            lang_label = "Hindi" if active_lang == "hi" else ("Telugu" if active_lang == "te" else active_lang.upper())
            frame = render_unicode_overlay(
                frame,
                f"Trans [{lang_label}]: {translated_preview}",
                pos=(20, 185),
                color_rgb=(50, 255, 255),
            )

        # GRU next-word predictions & sentence auto-complete
        if next_word_preds:
            preds_str = " | ".join([f"{w} ({int(p*100)}%)" for w, p in next_word_preds])
            cv2.putText(
                frame,
                f"Next: {preds_str}",
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (100, 255, 100),
                2,
            )

        if completed_sentence and completed_sentence.lower() != effective_text.lower().strip():
            cv2.putText(
                frame,
                f"Auto: {completed_sentence} [P: Accept]"[:55],
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

        # Action shortcuts
        cv2.putText(
            frame,
            "Sign 2 = Word Spoken + Space",
            (20, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "D: Del | C: Clr | S: Speak | L: Lang (HI/TE/EN) | P: Auto | Q: Quit",
            (20, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ISL Translator - Linguistic Voice & GRU", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("d"):
            builder.delete_last()
        elif key == ord("c"):
            builder.clear()
            prediction_history.clear()
            stable_prediction = ""
        elif key == ord(" "):
            speak_current_word()
            builder.add_space()
        elif key == ord("l"):
            # Cycle through Hindi -> Telugu -> English
            cycled_lang = tts.cycle_language(["hi", "te", "en"])
            print(f"[Language Switch] Active voice switched to: {tts.get_language_name(cycled_lang)}")
        elif key == ord("p"):
            # Auto-accept GRU sentence completion
            if completed_sentence:
                builder.word = completed_sentence.upper()
                print(f"[GRU Auto-Complete] Accepted: {builder.word}")
        elif key == ord("s"):
            text_to_speak = effective_text
            if text_to_speak:
                print(f"Speaking sentence [{tts.get_language_name()}]: {text_to_speak}")
                tts.speak(text_to_speak)

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    tts.stop()
    _trans_executor.shutdown(wait=False)


# ============================================================
# FINAL OUTPUT
# ============================================================
print()
print("==========================================")
print("Final detected text:")
print(builder.word)
print()
print("Final suggested text:")
final_sentence = correct_sentence(builder.word, suggester)
print(final_sentence)
if final_sentence:
    print("\nHindi Translation :", tts.translate(final_sentence, lang="hi"))
    print("Telugu Translation:", tts.translate(final_sentence, lang="te"))
print("==========================================")
