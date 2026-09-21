import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load combined dataset
# -----------------------------
with open("combined_data.pickle", "rb") as f:
    dataset = pickle.load(f)

data = np.asarray(dataset["data"])
labels = np.asarray(dataset["labels"])

print("Dataset shape:", data.shape)
print("Labels:", len(labels))

# -----------------------------
# 2. Train/Test split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# -----------------------------
# 3. Train Random Forest
# -----------------------------
print("\nTraining model...")

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

classifier.fit(x_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------
y_predict = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_predict))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predict))

# -----------------------------
# 5. Save model
# -----------------------------
with open("model_webcam.p", "wb") as f:
    pickle.dump(classifier, f)

print("\nModel saved as model_webcam.p")