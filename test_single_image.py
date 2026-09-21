import pickle
import cv2
import mediapipe as mp

# =========================
# LOAD MODEL
# =========================

with open("model.p", "rb") as f:
    model = pickle.load(f)

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

# =========================
# FUNCTION TO TEST IMAGE
# =========================

def test_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Could not read:", image_path)
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        print("No hand detected:", image_path)
        return

    detected_hands = results.multi_hand_landmarks

    # Same ordering used during dataset creation
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

            data_aux.append(landmark.x - min_x)
            data_aux.append(landmark.y - min_y)

    # Pad to 84 features
    while len(data_aux) < 84:
        data_aux.append(0.0)

    print("\nImage:", image_path)
    print("Number of features:", len(data_aux))

    # Prediction
    prediction = model.predict([data_aux])[0]

    # Confidence
    probabilities = model.predict_proba([data_aux])[0]
    confidence = max(probabilities) * 100

    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")

# =========================
# TEST IMAGES
# =========================

test_image(r"D:\Indian_sign_Language\data\5\2.jpg")

test_image(r"D:\Indian_sign_Language\data\Z\3.jpg")