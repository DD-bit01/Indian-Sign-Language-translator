import os
import pickle
import cv2
import mediapipe as mp


LABEL = "5"
NUM_SAMPLES = 100

SAVE_DIR = os.path.join(
    "webcam_data_normalized",
    LABEL
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ==========================================
# MEDIAPIPE
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


print("======================================")
print("NORMALIZED WEBCAM DATA COLLECTION")
print("======================================")
print("Sign:", LABEL)
print("Use your RIGHT hand")
print("Keep the sign correct")
print("You can slowly change distance")
print("Press SPACE to start")
print("Press Q to quit")
print("======================================")


started = False
counter = 0


while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not access webcam.")
        break


    # Same orientation used during inference
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape


    # ======================================
    # MEDIAPIPE
    # ======================================

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        frame_rgb
    )


    # ======================================
    # DRAW LANDMARKS
    # ======================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # ======================================
    # TEXT
    # ======================================

    if not started:

        cv2.putText(
            frame,
            "Show 5 - Press SPACE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            f"Collecting 5: "
            f"{counter}/{NUM_SAMPLES}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


    # ======================================
    # START COLLECTION
    # ======================================

    if started and results.multi_hand_landmarks:

        detected_hands = sorted(
            results.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )

        data_aux = []


        # ==================================
        # PROCESS EACH HAND
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


            min_x = min(x_coords)
            max_x = max(x_coords)

            min_y = min(y_coords)
            max_y = max(y_coords)


            hand_width = max_x - min_x
            hand_height = max_y - min_y


            # Prevent division by zero
            if hand_width == 0:
                hand_width = 1e-6

            if hand_height == 0:
                hand_height = 1e-6


            # ==================================
            # SCALE NORMALIZATION
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
        # PAD TO 84
        # ==================================

        while len(data_aux) < 84:

            data_aux.append(0.0)


        # ==================================
        # SAVE
        # ==================================

        if len(data_aux) == 84:

            file_path = os.path.join(
                SAVE_DIR,
                f"{counter}.pkl"
            )

            with open(
                file_path,
                "wb"
            ) as f:

                pickle.dump(
                    data_aux,
                    f
                )

            counter += 1


    # ======================================
    # SHOW WINDOW
    # ======================================

    cv2.imshow(
        "Normalized Data Collector",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord(" "):

        started = True


    if key == ord("q"):

        break


    if counter >= NUM_SAMPLES:

        print("\nCollection complete!")
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print(
    f"Saved {counter} samples "
    f"to {SAVE_DIR}"
)