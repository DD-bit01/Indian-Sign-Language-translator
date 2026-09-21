import os
import glob
import pickle

# -----------------------------
# 1. Load original dataset
# -----------------------------
with open("data.pickle", "rb") as f:
    dataset = pickle.load(f)

data = dataset["data"]
labels = dataset["labels"]

print("Original samples:", len(data))

# -----------------------------
# 2. Load webcam samples
# -----------------------------
webcam_files = glob.glob("webcam_data/5/*.pkl")

webcam_data = []

for file in webcam_files:
    with open(file, "rb") as f:
        sample = pickle.load(f)

    if len(sample) == 84:
        webcam_data.append(sample)

print("Webcam samples:", len(webcam_data))

# -----------------------------
# 3. Add webcam samples
# -----------------------------
data.extend(webcam_data)
labels.extend(["5"] * len(webcam_data))

# -----------------------------
# 4. Save combined dataset
# -----------------------------
combined_dataset = {
    "data": data,
    "labels": labels
}

with open("combined_data.pickle", "wb") as f:
    pickle.dump(combined_dataset, f)

print("Combined samples:", len(data))
print("Combined dataset saved as combined_data.pickle")