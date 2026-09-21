import cv2
import pickle
import numpy as np
import mediapipe as mp


# ==========================================
# LOAD MODEL
# ==========================================

with open("model_normalized.p", "rb") as f:
    model = pickle.load(f)


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)


    if results.multi_hand_landmarks:

        detected_hands = sorted(
            results.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )

        data_aux = []


        # ==========================================
        # SCALE-NORMALIZED FEATURES
        # ==========================================

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

            hand_width = max_x - min_x
            hand_height = max_y - min_y

            hand_width = max(hand_width, 1e-6)
            hand_height = max(hand_height, 1e-6)


            for landmark in hand_landmarks.landmark:

                normalized_x = (
                    landmark.x - min_x
                ) / hand_width

                normalized_y = (
                    landmark.y - min_y
                ) / hand_height

                data_aux.append(normalized_x)
                data_aux.append(normalized_y)


        # ==========================================
        # PAD TO 84
        # ==========================================

        while len(data_aux) < 84:
            data_aux.append(0.0)

        data_aux = data_aux[:84]

        features = np.asarray(data_aux).reshape(1, -1)


        # ==========================================
        # PREDICTION PROBABILITIES
        # ==========================================

        probabilities = model.predict_proba(features)[0]

        classes = model.classes_

        sorted_indices = np.argsort(
            probabilities
        )[::-1]


        print("\n" + "=" * 50)
        print("TOP PREDICTIONS")
        print("=" * 50)

        for index in sorted_indices[:5]:

            print(
                f"{classes[index]} : "
                f"{probabilities[index] * 100:.2f}%"
            )


        prediction = classes[sorted_indices[0]]

        confidence = probabilities[
            sorted_indices[0]
        ] * 100


        # ==========================================
        # DRAW LANDMARKS
        # ==========================================

        for hand_landmarks in detected_hands[:2]:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


        # ==========================================
        # DISPLAY
        # ==========================================

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    cv2.imshow(
        "I Diagnostic",
        frame
    )


    key = cv2.waitKey(1)

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

hands.close()