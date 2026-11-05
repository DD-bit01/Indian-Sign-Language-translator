"""
SignBridge: Real-Time Multimodal Indian Sign Language (ISL) Translator
Streamlit Cloud Edition (24/7 Free Cloud Hosting)
Author: Md Dilnawaz Ahmad
Faculty Guidance: Dr Korra Sathya Babu (IIITDM Kurnool)
"""

import os
import sys
import pickle
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import mediapipe as mp
from word_builder import WordBuilder
from word_suggester import WordSuggester
from linguistic_tts import LinguisticTTS
from sentence_predictor import SentencePredictor

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SignBridge - Indian Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
    }
    .sign-display {
        font-size: 3.5rem;
        font-weight: 900;
        color: #10b981;
        font-family: monospace;
        text-align: center;
    }
    .stButton>button {
        border-radius: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED MODEL LOADERS
# ============================================================
@st.cache_resource(show_spinner="Loading SignBridge Deep Learning Models...")
def load_models():
    # 1. Random Forest Classifier
    with open("model_final.p", "rb") as f:
        rf_model = pickle.load(f)

    # 2. PyTorch GRU Sentence Predictor
    sentence_predictor = SentencePredictor.load("model_sentence_gru.pt")

    # 3. NLP Word Suggester & Linguistic TTS
    word_suggester = WordSuggester()
    tts = LinguisticTTS(default_lang="hi")

    # 4. MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4,
    )

    return rf_model, sentence_predictor, word_suggester, tts, hands_detector, mp_drawing, mp_hands

rf_model, sentence_predictor, word_suggester, tts, hands_detector, mp_drawing, mp_hands = load_models()


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "assembled_text" not in st.session_state:
    st.session_state.assembled_text = ""
if "last_sign" not in st.session_state:
    st.session_state.last_sign = "-"
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "rule_status" not in st.session_state:
    st.session_state.rule_status = "Ready - Show your sign"
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# TOPOLOGY RULES (20 Two-Handed, 15 Single-Handed)
# ============================================================
TWO_HAND_CLASSES = {
    "A", "B", "D", "E", "F", "G", "H", "J", "K", "M",
    "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z",
}
ONE_HAND_CLASSES = {
    "C", "I", "L", "O", "U", "V",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
}

def validate_topology(label: str, hand_count: int):
    label = str(label).upper()
    if hand_count < 1:
        return False, "⚠️ No hands detected in camera"
    if label in TWO_HAND_CLASSES and hand_count < 2:
        return False, f"⚠️ '{label}' requires BOTH hands (Only {hand_count} detected)"
    if label in ONE_HAND_CLASSES and hand_count != 1:
        return False, f"⚠️ '{label}' is a ONE-handed sign ({hand_count} detected)"
    return True, f"✅ Valid Topology: {label} ({hand_count} Hand{'s' if hand_count > 1 else ''})"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/sign-language.png", width=64)
    st.title("SignBridge ISL")
    st.caption("**Author**: Md Dilnawaz Ahmad  \n**Guidance**: Dr Korra Sathya Babu (IIITDM Kurnool)")
    
    st.markdown("---")
    st.subheader("🗣️ Audio Output Language")
    selected_lang = st.selectbox(
        "Voice Language",
        options=["hi", "te", "en", "ta"],
        format_func=lambda x: {
            "hi": "🇮🇳 Hindi (हिन्दी)",
            "te": "🌾 Telugu (తెలుగు)",
            "en": "🇬🇧 English",
            "ta": "🌴 Tamil (தமிழ்)"
        }[x],
        index=0
    )
    tts.set_language(selected_lang)

    st.markdown("---")
    st.subheader("📊 System Specs")
    st.markdown("""
    - **Classifier**: Random Forest (35 ISL Classes)
    - **Language Model**: PyTorch Recurrent GRU
    - **Vision Engine**: MediaPipe 84-D Landmarks
    - **Cloud Uptime**: **24/7 Permanent (Streamlit Cloud)**
    """)

    st.markdown("---")
    with st.expander("📖 Indian Sign Language (ISL) Cheat Sheet"):
        st.markdown("""
        **Two-Handed Signs (Both hands required):**  
        `A, B, D, E, F, G, H, J, K, M, N, P, Q, R, S, T, W, X, Y, Z`  

        **Single-Handed Signs:**  
        `C, I, L, O, U, V, 1, 2, 3, 4, 5, 6, 7, 8, 9`
        """)


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🤟 SignBridge: Multimodal ISL Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Indian Sign Language Recognition with PyTorch GRU Neural Autocomplete & Vernacular Voice Synthesis</div>', unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION LAYOUT (2 COLUMNS)
# ============================================================
col_cam, col_nlp = st.columns([1.1, 1], gap="large")

with col_cam:
    st.subheader("📷 Camera Input")
    st.caption("Point your laptop or phone camera at your hands and capture a frame:")

    # Streamlit Camera Input (Works natively on phones and laptops)
    img_file_buffer = st.camera_input("Capture Sign", label_visibility="collapsed")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame_rgb = np.array(image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        results = hands_detector.process(frame_rgb)

        if results.multi_hand_landmarks:
            detected_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
            hand_count = len(detected_hands)
            data_aux = []

            for hand in detected_hands[:2]:
                x_coords = [lm.x for lm in hand.landmark]
                y_coords = [lm.y for lm in hand.landmark]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                w = max(max_x - min_x, 1e-6)
                h = max(max_y - min_y, 1e-6)

                for lm in hand.landmark:
                    data_aux.extend([(lm.x - min_x) / w, (lm.y - min_y) / h])

            # Pad to 84 dimensions for two hands
            while len(data_aux) < 84:
                data_aux.append(0.0)

            features = np.array(data_aux[:84]).reshape(1, -1)
            prediction = rf_model.predict(features)[0]

            # Probabilities / Confidence
            probs = rf_model.predict_proba(features)[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx] * 100

            # Validate hand count topology rule
            is_valid, msg = validate_topology(prediction, hand_count)
            st.session_state.rule_status = msg

            if is_valid:
                st.session_state.last_sign = prediction
                st.session_state.confidence = confidence

                # Sign '2' is the space/speech trigger
                if str(prediction) == "2":
                    st.session_state.assembled_text = (st.session_state.assembled_text.strip() + " ").strip() + " "
                    st.toast("Trigger '2': Word spoken and space added!", icon="🗣️")
            else:
                st.session_state.last_sign = f"({prediction}?)"
                st.session_state.confidence = confidence
        else:
            st.session_state.last_sign = "-"
            st.session_state.confidence = 0.0
            st.session_state.rule_status = "No hands detected. Show your hands clearly."


with col_nlp:
    st.subheader("🎯 Real-Time Recognition & NLP")

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Detected Sign", st.session_state.last_sign)
    with m2:
        st.metric("Confidence", f"{st.session_state.confidence:.1f}%")
    with m3:
        st.metric("Language", selected_lang.upper())

    # Topology Rule Banner
    st.info(st.session_state.rule_status)

    # Word Assembly Display
    st.markdown("### 📝 Assembled English Text")
    current_word = st.session_state.assembled_text

    # Spell Correction
    suggested_text = ""
    if current_word.strip():
        words = current_word.strip().split()
        corr_list = []
        for w in words:
            suggs = word_suggester.suggest(w, max_suggestions=1)
            corr_list.append(suggs[0] if suggs else w)
        suggested_text = " ".join(corr_list)

    display_text = suggested_text if suggested_text else current_word
    st.text_area("Live Text Buffer", value=display_text or "(Start signing or capture a frame...)", height=80, disabled=True)

    # GRU Predictions
    if display_text.strip():
        next_words = sentence_predictor.predict_next_words(display_text.strip(), top_k=2)
        completed_sentence = sentence_predictor.complete_sentence(display_text.strip(), max_words=3)

        if next_words:
            st.markdown("##### 🧠 PyTorch GRU Neural Predictions:")
            btn_cols = st.columns(len(next_words) + 1)
            for i, (pred_w, prob) in enumerate(next_words):
                with btn_cols[i]:
                    if st.button(f"+ {pred_w} ({int(prob*100)}%)", key=f"pred_{pred_w}"):
                        st.session_state.assembled_text = (display_text.strip() + " " + pred_w.upper()).strip() + " "
                        st.rerun()

            if completed_sentence and completed_sentence.lower() != display_text.lower().strip():
                with btn_cols[-1]:
                    if st.button("⚡ Complete", help=f"Accept: '{completed_sentence}'"):
                        st.session_state.assembled_text = completed_sentence.upper() + " "
                        st.rerun()

    # Controls Row
    st.markdown("##### 🎮 Controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("➕ Append Sign", use_container_width=True):
            if st.session_state.last_sign and st.session_state.last_sign not in ["-", "2"]:
                clean_sign = st.session_state.last_sign.replace("(", "").replace("?", "").replace(")", "")
                st.session_state.assembled_text += clean_sign
                st.rerun()
    with c2:
        if st.button("␣ Space", use_container_width=True):
            st.session_state.assembled_text += " "
            st.rerun()
    with c3:
        if st.button("⌫ Delete", use_container_width=True):
            st.session_state.assembled_text = st.session_state.assembled_text[:-1]
            st.rerun()
    with c4:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.assembled_text = ""
            st.session_state.last_sign = "-"
            st.rerun()

    # Vernacular Translation & Voice Output
    st.markdown("---")
    st.markdown("### 🌐 Vernacular Translation")
    t_col1, t_col2 = st.columns(2)

    text_to_translate = display_text.strip()
    with t_col1:
        st.markdown("**🇮🇳 Hindi (हिन्दी):**")
        hi_trans = tts.translate(text_to_translate, lang="hi") if text_to_translate else "प्रतीक्षा..."
        st.code(hi_trans, language="text")

    with t_col2:
        st.markdown("**🌾 Telugu (తెలుగు):**")
        te_trans = tts.translate(text_to_translate, lang="te") if text_to_translate else "వేచి ఉంది..."
        st.code(te_trans, language="text")

    # Speak Button
    if st.button("🔊 Speak Sentence", type="primary", use_container_width=True):
        if text_to_translate:
            try:
                import io
                from gtts import gTTS
                spoken_text = hi_trans if selected_lang == "hi" else (te_trans if selected_lang == "te" else text_to_translate)
                fp = io.BytesIO()
                g = gTTS(text=spoken_text or text_to_translate, lang=selected_lang, slow=False)
                g.write_to_fp(fp)
                fp.seek(0)
                st.audio(fp.read(), format="audio/mp3", autoplay=True)
                st.success(f"Speaking in {selected_lang.upper()}: '{spoken_text}'")
            except Exception as e:
                tts.speak(text_to_translate, lang=selected_lang)
                st.info(f"Spoken: '{text_to_translate}'")


# ============================================================
# FOOTER CREDITS
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <strong>SignBridge</strong> • Developed by <strong>Md Dilnawaz Ahmad</strong> • Mentored by <strong>Dr Korra Sathya Babu</strong> • Indian Institute of Information Technology, Design and Manufacturing (IIITDM) Kurnool
</div>
""", unsafe_allow_html=True)
