import cv2
import mediapipe as mp
import pickle
import numpy as np


# ==========================================
# LOAD MODEL
# ==========================================

with open("model_webcam.p", "rb") as f:
    model = pickle.load(f)


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

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

print("========================================")
print("ISL SCALE NORMALIZATION TEST")
print("========================================")
print("Show ISL SIGN 5")
print("Press SPACE to test")
print("Press Q to quit")
print("========================================")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break


    # ======================================
    # FLIP WEBCAM
    # ======================================

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape


    # ======================================
    # MEDIAPIPE
    # ======================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)


    # ======================================
    # DRAW HAND LANDMARKS
    # ======================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # ======================================
    # DISPLAY INSTRUCTIONS
    # ======================================

    cv2.putText(
        frame,
        "Show 5 - Press SPACE to test",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "ISL Scale Normalization Test",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ======================================
    # CAPTURE / TEST
    # ======================================

    if key == ord(" "):

        if not results.multi_hand_landmarks:

            print("\nNO HAND DETECTED")
            continue


        # ==================================
        # SORT HANDS LEFT → RIGHT
        # ==================================

        detected_hands = sorted(
            results.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )


        data_aux = []


        # ==================================
        # EXTRACT SCALE-NORMALIZED FEATURES
        # ==================================

        for hand_landmarks in detected_hands[:2]:

            x_coords = [
                lm.x
                for lm in hand_landmarks.landmark
            ]

            y_coords = [
                lm.y
                for lm in hand_landmarks.landmark
            ]


            # Hand bounding box

            min_x = min(x_coords)
            max_x = max(x_coords)

            min_y = min(y_coords)
            max_y = max(y_coords)


            # Hand dimensions

            hand_width = max_x - min_x
            hand_height = max_y - min_y


            # Prevent division by zero

            if hand_width == 0:
                hand_width = 1e-6

            if hand_height == 0:
                hand_height = 1e-6


            # ==================================
            # NORMALIZE EACH LANDMARK
            # ==================================

            for lm in hand_landmarks.landmark:

                normalized_x = (
                    lm.x - min_x
                ) / hand_width

                normalized_y = (
                    lm.y - min_y
                ) / hand_height


                data_aux.append(
                    normalized_x
                )

                data_aux.append(
                    normalized_y
                )


        # ==================================
        # PAD TO 84 FEATURES
        # ==================================

        while len(data_aux) < 84:

            data_aux.append(0.0)


        # ==================================
        # SAFETY CHECK
        # ==================================

        if len(data_aux) != 84:

            print(
                "ERROR: Feature length =",
                len(data_aux)
            )

            continue


        # Convert to NumPy array

        data_aux = np.asarray(
            data_aux
        ).reshape(1, -1)


        # ==================================
        # PRINT FEATURE INFORMATION
        # ==================================

        print("\n================================")
        print("NEW NORMALIZED FEATURE TEST")
        print("================================")

        print(
            "Feature shape:",
            data_aux.shape
        )

        print(
            "Feature minimum:",
            np.min(data_aux)
        )

        print(
            "Feature maximum:",
            np.max(data_aux)
        )


        # ==================================
        # MODEL PREDICTION
        # ==================================
        #
        # NOTE:
        # The current model was trained using
        # OLD features, so this prediction is
        # ONLY for debugging.
        #

        prediction = model.predict(
            data_aux
        )[0]


        print(
            "Prediction using OLD model:",
            prediction
        )


        # ==================================
        # PROBABILITIES
        # ==================================

        probabilities = model.predict_proba(
            data_aux
        )[0]

        classes = model.classes_


        top_indices = np.argsort(
            probabilities
        )[::-1][:5]


        print("\nTop 5 predictions:")

        for i in top_indices:

            print(
                f"{classes[i]} : "
                f"{probabilities[i] * 100:.2f}%"
            )


        print("================================\n")


    # ======================================
    # QUIT
    # ======================================

    if key == ord("q"):
        break


# ==========================================
# RELEASE
# ==========================================

cap.release()
cv2.destroyAllWindows()