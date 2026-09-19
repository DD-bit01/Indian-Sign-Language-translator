import cv2
import mediapipe as mp
import pickle
import os
import time

# =========================
# SETTINGS
# =========================

SAVE_DIR = "webcam_data_normalized/I"
TOTAL_SAMPLES = 150

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

print("\n===================================")
print(" ISL I-SIGN WEBCAM DATA COLLECTOR")
print("===================================")
print("Show the ISL letter I.")
print()
print("We will collect:")
print("50 CLOSE")
print("50 MEDIUM")
print("50 FAR")
print()
print("Press Q to quit.")
print("Press SPACE to start/pause collection.")
print("===================================\n")

collecting = False
count = 0

# =========================
# FEATURE EXTRACTION
# =========================

def extract_landmarks(results):

    all_hands = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))

            # -------------------------
            # SCALE NORMALIZATION
            # -------------------------

            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]

            min_x = min(xs)
            max_x = max(xs)

            min_y = min(ys)
            max_y = max(ys)

            hand_width = max(max_x - min_x, 1e-6)
            hand_height = max(max_y - min_y, 1e-6)

            normalized = []

            for x, y in landmarks:

                normalized_x = (x - min_x) / hand_width
                normalized_y = (y - min_y) / hand_height

                normalized.append(normalized_x)
                normalized.append(normalized_y)

            # wrist x used for sorting hands
            wrist_x = landmarks[0][0]

            all_hands.append((wrist_x, normalized))

    # -------------------------
    # SORT LEFT → RIGHT
    # -------------------------

    all_hands.sort(key=lambda x: x[0])

    features = []

    for _, hand_features in all_hands:
        features.extend(hand_features)

    # -------------------------
    # PAD TO 84 FEATURES
    # -------------------------

    while len(features) < 84:
        features.extend([0.0] * 42)

    features = features[:84]

    return features


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =========================
    # COLLECT
    # =========================

    if collecting and results.multi_hand_landmarks:

        features = extract_landmarks(results)

        filename = os.path.join(
            SAVE_DIR,
            f"I_{count + 1}.pkl"
        )

        with open(filename, "wb") as f:
            pickle.dump(features, f)

        count += 1

        # Small delay between samples
        time.sleep(0.05)

        if count >= TOTAL_SAMPLES:

            collecting = False

            print("\n===================================")
            print("COLLECTION COMPLETE!")
            print(f"Saved {count} samples.")
            print("===================================\n")

    # =========================
    # UI
    # =========================

    status = "COLLECTING" if collecting else "PAUSED"

    cv2.putText(
        frame,
        f"I Samples: {count}/{TOTAL_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        status,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Start/Pause | Q = Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Collect ISL I", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        collecting = not collecting

        if collecting:
            print("Collection STARTED")
        else:
            print("Collection PAUSED")

    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"\nTotal samples collected: {count}")