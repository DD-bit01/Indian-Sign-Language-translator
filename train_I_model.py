import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD DATA
# =========================

with open("combined_I_data.pickle", "rb") as f:
    data_dict = pickle.load(f)

X = data_dict["data"]
y = data_dict["labels"]

print("Dataset shape:", X.shape)
print("Labels:", y.shape)

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=True,
    stratify=y,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================
# RANDOM FOREST
# =========================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...\n")

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("==============================")
print("MODEL RESULTS")
print("==============================")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

MODEL_PATH = "model_I_improved.p"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\n==============================")
print("MODEL SAVED")
print("==============================")
print(MODEL_PATH)
