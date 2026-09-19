import os
import pickle
import cv2
import mediapipe as mp

# =========================
# SETTINGS
# =========================

LABEL = "5"
NUM_SAMPLES = 100

SAVE_DIR = os.path.join("webcam_data", LABEL)
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

print("Get ready...")
print("Show the ISL sign:", LABEL)
print("Press SPACE to start collecting.")
print("Press Q to quit.")

started = False
counter = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(frame_rgb)

    # =========================
    # DRAW HAND LANDMARKS
    # =========================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =========================
    # START COLLECTION
    # =========================

    if not started:

        cv2.putText(
            frame,
            "Show sign 5 - Press SPACE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            f"Collecting 5: {counter}/{NUM_SAMPLES}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # =========================
        # EXTRACT LANDMARKS
        # =========================

        if results.multi_hand_landmarks:

            detected_hands = results.multi_hand_landmarks

            # Sort hands left → right
            detected_hands = sorted(
                detected_hands,
                key=lambda hand: hand.landmark[0].x
            )

            data_aux = []

            for hand_landmarks in detected_hands[:2]:

                x_coords = [
                    landmark.x
                    for landmark in hand_landmarks.landmark
                ]

                y_coords = [
                    landmark.y
                    for landmark in hand_landmarks.landmark
                ]

                min_x = min(x_coords)
                min_y = min(y_coords)

                for landmark in hand_landmarks.landmark:

                    data_aux.append(
                        landmark.x - min_x
                    )

                    data_aux.append(
                        landmark.y - min_y
                    )

            # Pad one-hand signs to 84 features
            while len(data_aux) < 84:
                data_aux.append(0.0)

            # Save only valid 84-feature samples
            if len(data_aux) == 84:

                file_path = os.path.join(
                    SAVE_DIR,
                    f"{counter}.pkl"
                )

                with open(file_path, "wb") as f:
                    pickle.dump(data_aux, f)

                counter += 1

    # =========================
    # SHOW FRAME
    # =========================

    cv2.imshow(
        "ISL Webcam Data Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        started = True

    if key == ord("q"):
        break

    if counter >= NUM_SAMPLES:
        print("Collection complete!")
        break

cap.release()
cv2.destroyAllWindows()