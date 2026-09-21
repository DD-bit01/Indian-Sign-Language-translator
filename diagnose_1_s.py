import pickle
import numpy as np
from collections import Counter

# =========================
# LOAD DATA
# =========================

with open("data_normalized.pickle", "rb") as f:
    data_dict = pickle.load(f)

X = np.array(data_dict["data"])
y = np.array(data_dict["labels"])

with open("model_I_improved.p", "rb") as f:
    model = pickle.load(f)
# =========================
# TEST 1
# =========================

for target_class in ["1", "S"]:

    indices = [i for i, label in enumerate(y) if label == target_class]

    X_class = X[indices]

    predictions = model.predict(X_class)

    counts = Counter(predictions)

    print("\n==============================")
    print("CLASS:", target_class)
    print("SAMPLES:", len(X_class))
    print("==============================")

    print("Predictions:")

    for label, count in counts.most_common():
        print(f"{label}: {count}")

    wrong = sum(
        count for label, count in counts.items()
        if label != target_class
    )

    print("Wrong predictions:", wrong)

# =========================
# 1 / S CONFUSION
# =========================

print("\n==============================")
print("1 / S CONFUSION")
print("==============================")

for target_class in ["1", "S"]:

    indices = [i for i, label in enumerate(y) if label == target_class]

    predictions = model.predict(X[indices])

    wrong_predictions = [
        p for p in predictions
        if p != target_class
    ]

    print("\nActual:", target_class)
    print("Total:", len(predictions))
    print("Wrong:", len(wrong_predictions))

    if wrong_predictions:
        print("Wrong predictions:", Counter(wrong_predictions))