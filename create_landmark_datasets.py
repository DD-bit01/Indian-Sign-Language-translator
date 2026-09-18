import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    max_num_hands=2
)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(class_path):
        continue

    for img_path in os.listdir(class_path):
        img_full_path = os.path.join(class_path, img_path)

        img = cv2.imread(img_full_path)

        if img is None:
            print("Could not read:", img_full_path)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        detected_hands = results.multi_hand_landmarks

        # Sort hands from left to right using wrist x-coordinate
        detected_hands = sorted(
            detected_hands,
            key=lambda hand: hand.landmark[0].x
        )

        sample_features = []

        for hand_landmarks in detected_hands[:2]:

            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            min_x = min(x_coords)
            min_y = min(y_coords)

            for landmark in hand_landmarks.landmark:
                sample_features.append(landmark.x - min_x)
                sample_features.append(landmark.y - min_y)

        # If only one hand is detected, add zero features for second hand
        while len(sample_features) < 84:
            sample_features.append(0.0)

        # Safety check
        if len(sample_features) == 84:
            data.append(sample_features)
            labels.append(dir_)

print("Total valid samples:", len(data))

with open('data.pickle', 'wb') as f:
    pickle.dump({
        'data': data,
        'labels': labels
    }, f)

print("Dataset saved successfully as data.pickle")