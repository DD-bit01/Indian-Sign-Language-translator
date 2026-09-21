import pickle
import numpy as np
from collections import Counter

# ==========================================
# LOAD NORMALIZED DATASET
# ==========================================

with open("data_normalized.pickle", "rb") as f:
    dataset = pickle.load(f)

data = np.asarray(dataset["data"])
labels = np.asarray(dataset["labels"])


print("Dataset shape:", data.shape)
print()


# ==========================================
# CHECK I AND Q
# ==========================================

for label in ["I", "Q"]:

    indices = np.where(labels == label)[0]

    class_data = data[indices]

    print("=" * 50)
    print("CLASS:", label)
    print("Samples:", len(class_data))

    print(
        "Feature mean:",
        np.mean(class_data)
    )

    print(
        "Feature std:",
        np.std(class_data)
    )

    print(
        "Feature min:",
        np.min(class_data)
    )

    print(
        "Feature max:",
        np.max(class_data)
    )

    print()


# ==========================================
# TRAINED MODEL
# ==========================================

with open("model_normalized.p", "rb") as f:
    model = pickle.load(f)


# ==========================================
# TEST HOW MODEL CLASSIFIES ITS OWN
# I AND Q SAMPLES
# ==========================================

print("=" * 50)
print("MODEL SELF-TEST")
print("=" * 50)


for label in ["I", "Q"]:

    indices = np.where(labels == label)[0]

    predictions = model.predict(data[indices])

    counts = Counter(predictions)

    print()
    print("Actual class:", label)
    print("Predictions:", dict(counts))


# ==========================================
# CONFUSION SPECIFICALLY BETWEEN I AND Q
# ==========================================

print()
print("=" * 50)
print("I / Q CONFUSION")
print("=" * 50)

for label in ["I", "Q"]:

    indices = np.where(labels == label)[0]

    predictions = model.predict(data[indices])

    wrong = predictions[predictions != label]

    print()
    print("Actual:", label)
    print("Total:", len(predictions))
    print("Wrong predictions:", len(wrong))

    if len(wrong) > 0:
        print("Wrong prediction breakdown:")
        print(dict(Counter(wrong)))