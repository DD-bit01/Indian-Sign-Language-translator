import pickle
import os
import numpy as np

# =========================
# LOAD ORIGINAL DATA
# =========================

with open("data_normalized.pickle", "rb") as f:
    original_data = pickle.load(f)

X = list(original_data["data"])
y = list(original_data["labels"])

print("Original samples:", len(X))

# =========================
# LOAD WEBCAM I SAMPLES
# =========================

I_DIR = "webcam_data_normalized/I"

files = [
    f for f in os.listdir(I_DIR)
    if f.endswith(".pkl")
]

added = 0

for file in files:

    path = os.path.join(I_DIR, file)

    with open(path, "rb") as f:
        features = pickle.load(f)

    if len(features) == 84:

        X.append(features)
        y.append("I")

        added += 1

# =========================
# SAVE COMBINED DATA
# =========================

combined = {
    "data": np.array(X, dtype=np.float32),
    "labels": np.array(y)
}

with open("combined_I_data.pickle", "wb") as f:
    pickle.dump(combined, f)

print("\n==============================")
print("COMBINATION COMPLETE")
print("==============================")
print("Original samples:", len(original_data["data"]))
print("I webcam samples:", added)
print("Total samples:", len(X))
print("Saved as: combined_I_data.pickle")