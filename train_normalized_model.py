import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Load combined normalized data
# -----------------------------

with open("combined_normalized_data.pickle", "rb") as f:
    dataset = pickle.load(f)

data = np.asarray(dataset["data"])
labels = np.asarray(dataset["labels"])


print("Dataset shape:", data.shape)
print("Labels:", labels.shape)


# -----------------------------
# Train / Test split
# -----------------------------

x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.20,
    shuffle=True,
    stratify=labels,
    random_state=42
)


print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))


# -----------------------------
# Create model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)


# -----------------------------
# Train
# -----------------------------

print("\nTraining model...")

model.fit(x_train, y_train)


# -----------------------------
# Test
# -----------------------------

y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)

print("\nAccuracy:", accuracy)


print("\nClassification Report:")
print(classification_report(y_test, y_predict))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predict))


# -----------------------------
# Save model
# -----------------------------

with open("model_normalized.p", "wb") as f:
    pickle.dump(model, f)


print("\nModel saved as model_normalized.p")