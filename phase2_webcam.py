import cv2
import pickle
import numpy as np
import mediapipe as mp

from collections import Counter, deque

from word_builder import WordBuilder
from word_suggester import WordSuggester
from text_to_speech import TextToSpeech


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "model_final.p"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


# ============================================================
# OBJECTS
# ============================================================

builder = WordBuilder(cooldown=0.7)
suggester = WordSuggester()
tts = TextToSpeech()


# ============================================================
# PREDICTION SMOOTHING
# ============================================================

prediction_history = deque(maxlen=7)

stable_prediction = ""


# ============================================================
# SPECIAL SIGN STATE
# ============================================================

# Prevents the "2" sign from triggering repeatedly
special_sign_active = False


# ============================================================
# SENTENCE CORRECTION
# ============================================================

def correct_sentence(text, suggester):

    if not text:
        return ""

    words = text.strip().split()

    corrected_words = []

    for word in words:

        suggestions = suggester.suggest(
            word,
            max_suggestions=1
        )

        if suggestions:
            corrected_words.append(
                suggestions[0]
            )
        else:
            corrected_words.append(
                word
            )

    return " ".join(corrected_words)


# ============================================================
# SPEAK CURRENT WORD
# ============================================================

def speak_current_word():

    current_text = builder.word.strip()

    if not current_text:
        return

    current_word = current_text.split()[-1]

    suggestions = suggester.suggest(
        current_word,
        max_suggestions=1
    )

    if suggestions:
        word_to_speak = suggestions[0]
    else:
        word_to_speak = current_word

    print(
        f"Speaking word: {word_to_speak}"
    )

    tts.speak(
        word_to_speak
    )


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# START MESSAGE
# ============================================================

print()
print("==========================================")
print(" ISL TRANSLATOR - PHASE 3")
print("==========================================")
print()
print("Special Sign:")
print("2 -> Speak current word + Add space")
print()
print("Keyboard Controls:")
print("D     -> Delete last character")
print("C     -> Clear text")
print("SPACE -> Backup space + Speak word")
print("S     -> Speak complete sentence")
print("Q     -> Quit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read webcam frame."
        )

        break


    # ========================================================
    # FLIP WEBCAM
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        rgb
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        # Sort hands left-to-right
        detected_hands = sorted(
            results.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )


        data_aux = []


        # ====================================================
        # EXTRACT LANDMARKS
        # ====================================================

        for hand_landmarks in detected_hands[:2]:

            x_values = [
                landmark.x
                for landmark in hand_landmarks.landmark
            ]

            y_values = [
                landmark.y
                for landmark in hand_landmarks.landmark
            ]


            min_x = min(x_values)
            max_x = max(x_values)

            min_y = min(y_values)
            max_y = max(y_values)


            hand_width = max(
                max_x - min_x,
                1e-6
            )

            hand_height = max(
                max_y - min_y,
                1e-6
            )


            # =================================================
            # NORMALIZE LANDMARKS
            # =================================================

            for landmark in hand_landmarks.landmark:

                normalized_x = (
                    landmark.x - min_x
                ) / hand_width

                normalized_y = (
                    landmark.y - min_y
                ) / hand_height


                data_aux.append(
                    normalized_x
                )

                data_aux.append(
                    normalized_y
                )


        # ====================================================
        # ZERO PAD
        # ====================================================

        while len(data_aux) < 84:

            data_aux.append(0.0)


        data_aux = data_aux[:84]


        # ====================================================
        # NUMPY FEATURES
        # ====================================================

        features = np.asarray(
            data_aux,
            dtype=np.float32
        ).reshape(1, -1)


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            features
        )[0]


        # ====================================================
        # CONFIDENCE
        # ====================================================

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = (
            np.max(probabilities) * 100
        )


        # ====================================================
        # TEMPORAL SMOOTHING
        # ====================================================

        prediction_history.append(
            prediction
        )


        counts = Counter(
            prediction_history
        )


        most_common_prediction, count = (
            counts.most_common(1)[0]
        )


        # Need 4 of 7 frames
        if count >= 4:

            current_prediction = (
                most_common_prediction
            )


            # =================================================
            # SPECIAL SIGN: 2
            # =================================================

            if current_prediction == "2":

                # Trigger ONLY once
                if not special_sign_active:

                    special_sign_active = True

                    # Speak current word
                    speak_current_word()

                    # Add space
                    builder.add_space()

                    print(
                        "Special sign 2 -> SPACE"
                    )


                # Keep displaying 2
                stable_prediction = "2"


            # =================================================
            # NORMAL LETTER
            # =================================================

            else:

                # 2 sign has ended
                special_sign_active = False


                # Only accept when prediction changes
                if current_prediction != stable_prediction:

                    stable_prediction = (
                        current_prediction
                    )

                    builder.add_prediction(
                        stable_prediction
                    )


        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        for hand_landmarks in detected_hands[:2]:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        prediction_history.clear()

        stable_prediction = ""

        # IMPORTANT:
        # Allows the next "2" sign to trigger again
        special_sign_active = False

        builder.reset_sign()


    # ========================================================
    # DETECTED TEXT
    # ========================================================

    detected_text = builder.word


    # ========================================================
    # FULL SENTENCE SUGGESTION
    # ========================================================

    suggestion = correct_sentence(
        detected_text,
        suggester
    )


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    cv2.putText(
        frame,
        f"Prediction: {stable_prediction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )


    # ========================================================
    # DISPLAY CONFIDENCE
    # ========================================================

    if results.multi_hand_landmarks:

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DISPLAY DETECTED TEXT
    # ========================================================

    cv2.putText(
        frame,
        f"Detected: {detected_text}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY SUGGESTION
    # ========================================================

    if suggestion:

        cv2.putText(
            frame,
            f"Suggested: {suggestion}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 255),
            2
        )


    # ========================================================
    # SPECIAL SIGN INFO
    # ========================================================

    cv2.putText(
        frame,
        "Sign 2 = Speak Word + Space",
        (20, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "D: Delete | C: Clear | S: Speak Sentence | Q: Quit",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # SHOW WINDOW
   

    cv2.imshow(
        "ISL Translator - Phase 3",
        frame
    )


    # KEYBOARD INPUT
   

    key = cv2.waitKey(1) & 0xFF


   
    # QUIT
    

    if key == ord("q"):

        break

    # DELETE
   

    elif key == ord("d"):

        builder.delete_last()

    # CLEAR
   

    elif key == ord("c"):

        builder.clear()

    # BACKUP SPACE
 

    elif key == ord(" "):

        speak_current_word()

        builder.add_space()

    # SPEAK COMPLETE SENTENCE


    elif key == ord("s"):

        if suggestion:

            print(
                f"Speaking sentence: {suggestion}"
            )

            tts.speak(
                suggestion
            )

        elif detected_text:

            print(
                f"Speaking sentence: {detected_text}"
            )

            tts.speak(
                detected_text
            )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==========================================")

print(
    "Final detected text:"
)

print(
    builder.word
)

print()

print(
    "Final suggested text:"
)

print(
    correct_sentence(
        builder.word,
        suggester
    )
)

print("==========================================")