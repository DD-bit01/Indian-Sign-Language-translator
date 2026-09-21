# SignBridge — Real-Time Indian Sign Language Translator

> A computer-vision and machine-learning based prototype that recognizes Indian Sign Language (ISL) alphabet signs from a camera, builds words and sentences, and converts the generated text into speech.

## Demo

**Example:**  
`H → I → [2] → I → [2] → A → M → [2] → D → I → L → N → A → W → A → Z`

The system can build:

**`hi i am dilnawaz`**

and speak the generated text using text-to-speech.

---

## Problem

People who communicate primarily through Indian Sign Language may face communication barriers when interacting with people who do not understand ISL.

SignBridge explores a lightweight way to reduce this barrier using a camera, hand-landmark detection, machine learning, text processing, and speech synthesis.

The current prototype focuses on **ISL alphabet/static-sign recognition and sentence construction**. It is a foundation for future continuous/dynamic ISL translation.

---

## What SignBridge Does

The current prototype provides an end-to-end pipeline:

1. Captures live video from a webcam.
2. Detects up to two hands using MediaPipe.
3. Extracts 21 landmarks per hand.
4. Converts the landmarks into a fixed **84-feature vector** using x/y coordinates.
5. Normalizes the landmark coordinates.
6. Uses a trained Random Forest classifier to recognize the sign.
7. Applies temporal smoothing to reduce unstable frame-to-frame predictions.
8. Uses sign-boundary logic to prevent the same held sign from being added repeatedly.
9. Builds characters into words and words into sentences.
10. Provides lightweight word correction/suggestion using Python's `difflib`.
11. Uses text-to-speech to speak completed words or the full sentence.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │   Camera / Webcam   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      MediaPipe      │
                    │    Hand Landmarks   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Processing  │
                    │ 84 normalized x/y   │
                    │      features       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Random Forest ML   │
                    │     Classifier      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Temporal Smoothing  │
                    │ & Sign Boundary     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Word Builder     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Word Suggestions /  │
                    │    Correction       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text-to-Speech    │
                    └─────────────────────┘
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Hand Landmark Detection | MediaPipe |
| Machine Learning | Scikit-learn |
| Model | Random Forest Classifier |
| Numerical Processing | NumPy |
| Text Suggestion | Python `difflib` |
| Text-to-Speech | `pyttsx3` |
| Development | Python 3.11 |
| Version Control | Git / GitHub |

---

## Dataset

The original dataset is organized by class:

```text
data/
├── A/
├── B/
├── C/
├── ...
├── Z/
├── 1/
├── 2/
├── ...
└── 9/
```

The dataset is processed programmatically rather than requiring manually created train/test folders.

### Data preparation

For each image:

1. MediaPipe detects the hands.
2. Up to two hands are retained.
3. Hands are ordered from left to right using wrist x-coordinate.
4. Each hand contributes:
   - 21 landmarks
   - x and y coordinates
   - 42 values per hand
5. Two hands therefore produce:

```text
2 × 21 × 2 = 84 features
```

If only one hand is detected, the second hand is zero-padded.

---

## Landmark Normalization

Raw image coordinates can change when the hand moves around the frame.

To reduce the effect of translation and hand position, landmarks are normalized using the hand bounding region:

```text
normalized_x = (landmark.x - min_x) / hand_width
normalized_y = (landmark.y - min_y) / hand_height
```

The normalized landmark vectors are then used as the model input.

---

## Model

The current classifier is a **Random Forest** trained on the extracted landmark features.

### Current training configuration

```python
RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
```

The dataset is split using a stratified 80/20 train-test split.

### Current validation result

The current image-level holdout evaluation achieved approximately:

**99.98% accuracy**

This result should be interpreted carefully. The split is performed at the sample/image level, so it does **not** establish generalization to completely unseen users, cameras, lighting conditions, or signing styles.

A subject-separated evaluation is a planned improvement.

---

## Real-World Webcam Data

To improve practical recognition for difficult or visually similar signs, additional webcam samples were collected and incorporated into the final training dataset.

Examples of problem cases investigated during development included visually similar signs such as:

- `1` / `S`
- `I` / `Q`

The final model was trained using the combined normalized data.

---

## Sentence Construction

### WordBuilder

`word_builder.py` manages character-by-character construction.

The system is designed so that:

- holding the same sign does not repeatedly add the same character;
- a sign must change or disappear before another occurrence is accepted;
- the last character can be deleted;
- the current sentence can be cleared;
- a space can be inserted between words.

### Word Suggestions

`word_suggester.py` provides lightweight suggestions/correction using:

```python
difflib.get_close_matches()
```

For example:

```text
Input:
dilnwaz

Suggestion:
dilnawaz
```

The project also supports sentence-level correction by processing individual words.

---

## Special Controls

The current webcam prototype uses the following controls:

| Control | Action |
|---|---|
| Sign `2` | Speak the current word and add a space |
| `D` | Delete the last character |
| `C` | Clear the current sentence |
| `S` | Speak the complete sentence |
| `Space` | Keyboard space backup |
| `Q` | Quit the application |

> The exact sign/control mapping can be changed as the project evolves.

---

## Text-to-Speech

Speech is implemented with `pyttsx3`.

The current configuration uses a Windows-installed voice and a speech rate of **110** for a relatively clear, moderately slow demo experience.

The TTS implementation initializes a fresh speech engine for each spoken text segment because this was found to be more reliable on the development environment.

### Available development voices

The current development machine exposes:

- Microsoft David Desktop — English (United States)
- Microsoft Zira Desktop — English (United States)

An Indian English voice can be added later if a suitable Windows/system or cloud voice is available.

---

## Project Structure

```text
Indian-Sign-Language-translator/
│
├── data/
│   └── A ... Z, 1 ... 9
│
├── model_final.p
│
├── create_landmark_datasets.py
├── create_normalized_webcam.py
├── train_classifier.py
├── train_final_model.py
│
├── inference_classifier.py
├── phase2_webcam.py
│
├── word_builder.py
├── word_suggester.py
├── text_to_speech.py
│
├── collect_webcam_data.py
├── collect_webcam_1.py
├── collect_webcam_I.py
├── combine_final.py
│
├── requirements.txt
└── README.md
```

Some experimental/debug scripts may exist locally during development but are not part of the core runtime pipeline.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DD-bit001/Indian-Sign-Language-translator.git
cd Indian-Sign-Language-translator
```

### 2. Create a virtual environment (recommended)

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### `requirements.txt`

```text
opencv-python
mediapipe
numpy
scikit-learn
pyttsx3
```

---

## Running the Prototype

Make sure:

- a working camera/webcam is connected;
- `model_final.p` is present in the project directory;
- the required Python packages are installed.

Run:

```powershell
python phase2_webcam.py
```

The application opens the camera feed and begins recognizing signs.

---

## Typical Usage

1. Start the application.
2. Position your hand clearly inside the camera view.
3. Hold a sign steadily.
4. Wait for the prediction to stabilize.
5. Change/remove the sign before repeating the same character.
6. Build the desired word.
7. Use sign `2` to complete/speak the word and insert a space.
8. Continue signing the next word.
9. Press `S` to speak the complete sentence.
10. Press `D` to delete a character or `C` to clear the text.

---

## Why Temporal Smoothing Is Used

A camera produces a continuous stream of frames. Even when the hand is stationary, individual frames may occasionally produce different predictions.

To reduce this problem, SignBridge keeps a short prediction history and uses majority voting to decide when a prediction is stable enough to accept.

Conceptually:

```text
Frame 1 → H
Frame 2 → H
Frame 3 → H
Frame 4 → H
Frame 5 → I
Frame 6 → H
Frame 7 → H

Majority → H
```

This makes the system less sensitive to small frame-to-frame fluctuations.

---

## Current Scope

The current version is primarily an **alphabet/static-sign recognition prototype**.

It is not yet a complete continuous natural-language ISL translator.

That distinction is important because natural sign language can include:

- dynamic movements;
- transitions between signs;
- facial expressions;
- body posture;
- grammar that is different from spoken English;
- context-dependent meaning.

The current system focuses on establishing a reliable technical foundation before moving to these harder problems.

---

## Limitations

### 1. Alphabet-focused recognition

The current model mainly recognizes the sign classes present in the dataset rather than unrestricted conversational ISL.

### 2. Image-level validation

The reported validation accuracy comes from an image-level split. A stronger evaluation should separate data by signer/subject.

### 3. Environmental sensitivity

Recognition can vary with:

- lighting;
- camera quality;
- background;
- hand occlusion;
- distance from the camera;
- unusual hand orientations.

### 4. Sentence generation

The current sentence builder is character-driven and uses lightweight dictionary-based suggestions. It is not a full natural-language ISL-to-English translation model.

### 5. TTS voice availability

Speech quality and available voices depend on the operating system and installed speech engines.

---

## Future Roadmap

### Next Development

- Cleaner user interface
- Next-word suggestions in the UI
- Better sentence formatting
- Confidence/probability display
- Improved error handling
- More robust sign-boundary detection

### Mobile / iQOO Deployment

The next major direction is to move the recognition experience to an Android/iQOO smartphone:

```text
iQOO Camera
     ↓
Hand Detection
     ↓
Landmark Extraction
     ↓
On-device ISL Model
     ↓
Word / Sentence Builder
     ↓
Phone TTS
```

The goal is to make the system portable and usable without requiring a laptop webcam.

### Longer-Term Research Direction

- signer-independent evaluation;
- more ISL classes;
- dynamic gesture recognition;
- continuous sign segmentation;
- sentence-level language modeling;
- richer ISL grammar handling;
- multilingual output;
- accessibility-focused mobile UX;
- optimized on-device inference.

---

## Reproducibility

The project keeps the landmark extraction and model-training process in code so that the model pipeline can be recreated from the source dataset.

The main stages are:

```text
Raw Images
   ↓
MediaPipe Landmark Extraction
   ↓
Normalization
   ↓
84-Dimensional Feature Dataset
   ↓
Random Forest Training
   ↓
model_final.p
   ↓
Real-Time Inference
```

---

## Model and Data Notes

The repository may contain the final trained model required for inference.

Large raw datasets and intermediate generated files can be excluded from normal Git tracking because of repository size and distribution considerations. If the dataset is not included in the repository, obtain it from the project owner/source used for training before attempting to rebuild the model.

---

## GitHub

Repository:

**https://github.com/DD-bit001/Indian-Sign-Language-translator**

---

## Project Status

**Phase 1 / Prototype: Working**

Current capabilities:

- ✅ Real-time webcam input
- ✅ Hand landmark extraction
- ✅ 84-feature normalized representation
- ✅ Random Forest sign classification
- ✅ Temporal smoothing
- ✅ Sign-boundary handling
- ✅ Character-to-word construction
- ✅ Sentence construction
- ✅ Basic spelling/word correction
- ✅ Text-to-speech
- ✅ Final trained model

Planned:

- 🔄 Next-word prediction UI
- 🔄 Android/iQOO deployment
- 🔄 Dynamic ISL recognition
- 🔄 Stronger signer-independent evaluation
- 🔄 Continuous natural-language translation

---

## Disclaimer

SignBridge is a student-built prototype intended for experimentation, demonstration, and further research. It should not be treated as a certified accessibility or communication system without additional validation across diverse users and real-world conditions.

---

## Contributors

**Dilnawaz / Team SignBridge**

Built as a student machine-learning and computer-vision project focused on improving communication accessibility through Indian Sign Language technology.

---

## License

Add an appropriate open-source license before distributing the project publicly. Common choices include the MIT License or Apache License 2.0, depending on the project's intended use and any third-party dataset/model restrictions.
