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
# FUNCTION TO ADD WEBCAM DATA
# =========================

def add_webcam_samples(folder, label):

    global X, y

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".pkl")
    ]

    added = 0

    for file in files:

        path = os.path.join(folder, file)

        with open(path, "rb") as f:
            features = pickle.load(f)

        if len(features) == 84:

            X.append(features)
            y.append(label)

            added += 1

    print(f"Webcam {label} samples added:", added)


# =========================
# ADD I SAMPLES
# =========================

add_webcam_samples(
    "webcam_data_normalized/I",
    "I"
)


# =========================
# ADD 1 SAMPLES
# =========================

add_webcam_samples(
    "webcam_data_normalized/1",
    "1"
)


# =========================
# CONVERT TO NUMPY
# =========================

X = np.array(X, dtype=np.float32)
y = np.array(y)


# =========================
# SAVE FINAL DATASET
# =========================

final_data = {
    "data": X,
    "labels": y
}

with open("combined_final_data.pickle", "wb") as f:
    pickle.dump(final_data, f)


# =========================
# SUMMARY
# =========================

print("\n==============================")
print("FINAL DATASET CREATED")
print("==============================")

print("Original samples:", len(original_data["data"]))
print("Webcam I samples: 150")
print("Webcam 1 samples: 150")
print("Total samples:", len(X))

print("Feature shape:", X.shape)

print("\nSaved as:")
print("combined_final_data.pickle")