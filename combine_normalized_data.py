import os
import pickle

# -----------------------------
# Load original normalized data
# -----------------------------

with open("data_normalized.pickle", "rb") as f:
    original = pickle.load(f)

data = original["data"]
labels = original["labels"]

print("Original normalized samples:", len(data))


# -----------------------------
# Load normalized webcam data
# -----------------------------

WEBCAM_DIR = "webcam_data_normalized"

webcam_count = 0

for label in os.listdir(WEBCAM_DIR):

    label_path = os.path.join(WEBCAM_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for filename in os.listdir(label_path):

        if not filename.endswith(".pkl"):
            continue

        file_path = os.path.join(label_path, filename)

        with open(file_path, "rb") as f:
            sample = pickle.load(f)

        # Make sure feature vector has 84 values
        if len(sample) != 84:
            print("Skipping invalid sample:", file_path)
            continue

        data.append(sample)
        labels.append(label)

        webcam_count += 1


# -----------------------------
# Save combined dataset
# -----------------------------

combined = {
    "data": data,
    "labels": labels
}

with open("combined_normalized_data.pickle", "wb") as f:
    pickle.dump(combined, f)


print()
print("Webcam samples added:", webcam_count)
print("Total combined samples:", len(data))
print("Combined normalized dataset saved successfully!")