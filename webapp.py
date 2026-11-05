"""
SignBridge: Real-Time Multimodal Indian Sign Language Translator
Interactive FastAPI Web Application Prototype - Mobile & Desktop Edition
Author: Md Dilnawaz Ahmad, Under the Guidance of Dr Korra Sathya Babu (IIITDM Kurnool)
"""

import os
import sys
import cv2
import time
import json
import base64
import socket
import pickle
import threading
import numpy as np
from typing import Dict, List, Optional
from collections import Counter, deque
import concurrent.futures

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import mediapipe as mp
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from word_builder import WordBuilder
from word_suggester import WordSuggester
from linguistic_tts import LinguisticTTS
from sentence_predictor import SentencePredictor

# ============================================================
# NETWORK UTILS: DETECT LOCAL IP FOR PHONE PAIRING
# ============================================================
def get_local_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

LAN_IP = get_local_lan_ip()
DEFAULT_PORT = 8050

# ============================================================
# APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="SignBridge - Mobile & Desktop ISL Translator",
    description="Real-Time Indian Sign Language Recognition & Vernacular Speech Translation Prototype",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODELS & LOGIC
# ============================================================
MODEL_PATH = "model_final.p"
SENTENCE_MODEL_PATH = "model_sentence_gru.pt"

print("[Init] Loading Random Forest ISL classifier...")
with open(MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

print("[Init] Loading GRU Sentence Predictor...")
sentence_predictor = SentencePredictor.load(SENTENCE_MODEL_PATH)

word_builder = WordBuilder(cooldown=0.7)
word_suggester = WordSuggester()
tts = LinguisticTTS(default_lang="hi")

# Thread pool for non-blocking background translation
trans_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="WebTrans")

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5,
    model_complexity=0,
)

# Rule Engine Definitions
TWO_HAND_CLASSES = {
    "A", "B", "D", "E", "F", "G", "H", "J", "K", "M",
    "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z",
}
ONE_HAND_CLASSES = {
    "C", "I", "L", "O", "U", "V",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
}

def validate_hand_count(label: str, detected_hands: int):
    label = str(label).upper()
    if detected_hands < 1:
        return False, "No hands detected"
    if label in TWO_HAND_CLASSES and detected_hands < 2:
        return False, f"{label} needs BOTH hands"
    if label in ONE_HAND_CLASSES and detected_hands != 1:
        return False, f"{label} needs ONE hand"
    return True, "Valid hand topology"


# ============================================================
# RUNTIME STATE
# ============================================================
class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_sign = ""
        self.confidence = 0.0
        self.detected_hands = 0
        self.rule_message = "Ready - Show your sign"
        self.prediction_history = deque(maxlen=7)
        self.special_sign_active = False

        self.last_text = None
        self.last_lang = None
        self.suggestion = ""
        self.effective_text = ""
        self.translated_hi = ""
        self.translated_te = ""
        self.next_word_preds: List[tuple] = []
        self.completed_sentence = ""
        self.fps = 0.0
        self.active_lang = "hi"
        self.camera_active = True

state = PipelineState()


# ============================================================
# HELPER: EXTRACT 84-D LANDMARKS & PREDICT
# ============================================================
def process_landmarks_and_predict(frame_rgb, frame_bgr=None):
    results = hands_detector.process(frame_rgb)
    detected_count = 0
    candidate_prediction = ""
    conf = 0.0
    landmarks_summary = []

    if results.multi_hand_landmarks:
        detected_hands = results.multi_hand_landmarks
        detected_count = len(detected_hands)

        # Sort hands left-to-right by wrist x
        detected_hands = sorted(detected_hands, key=lambda h: h.landmark[0].x)
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

            landmarks_summary.append({
                "wrist": {"x": hand.landmark[0].x, "y": hand.landmark[0].y},
                "bbox": {"x": min_x, "y": min_y, "w": w, "h": h},
                "points": [{"x": lm.x, "y": lm.y} for lm in hand.landmark]
            })

            # Draw visual skeleton if BGR frame provided
            if frame_bgr is not None:
                mp_drawing.draw_landmarks(frame_bgr, hand, mp_hands.HAND_CONNECTIONS)

        while len(data_aux) < 84:
            data_aux.append(0.0)
        data_aux = data_aux[:84]

        features = np.asarray(data_aux, dtype=np.float32).reshape(1, -1)
        candidate_prediction = str(rf_model.predict(features)[0]).upper()
        probabilities = rf_model.predict_proba(features)[0]
        conf = float(np.max(probabilities) * 100)

    # State update with lock
    with state.lock:
        state.detected_hands = detected_count
        if detected_count == 0:
            state.prediction_history.clear()
            state.current_sign = ""
            state.special_sign_active = False
            word_builder.reset_sign()
            state.rule_message = "Show your sign clearly"
            state.confidence = 0.0
        else:
            state.confidence = conf
            accepted, msg = validate_hand_count(candidate_prediction, detected_count)
            if not accepted:
                state.prediction_history.clear()
                state.current_sign = ""
                state.special_sign_active = False
                word_builder.reset_sign()
                state.rule_message = msg
            else:
                state.rule_message = f"Hands detected: {detected_count} (Valid)"
                state.prediction_history.append(candidate_prediction)
                counts = Counter(state.prediction_history)
                most_common, count = counts.most_common(1)[0]

                if count >= 4:
                    if most_common == "2":
                        if not state.special_sign_active:
                            state.special_sign_active = True
                            speak_curr_word()
                            word_builder.add_space()
                        state.current_sign = "2"
                    else:
                        state.special_sign_active = False
                        if most_common != state.current_sign:
                            state.current_sign = most_common
                            word_builder.add_prediction(state.current_sign)

        # NLP Update on change
        current_word_str = word_builder.word
        if current_word_str != state.last_text or state.active_lang != state.last_lang:
            state.last_text = current_word_str
            state.last_lang = state.active_lang

            # Spell correction
            if current_word_str.strip():
                words = current_word_str.strip().split()
                corr_list = []
                for w in words:
                    suggs = word_suggester.suggest(w, max_suggestions=1)
                    corr_list.append(suggs[0] if suggs else w)
                state.suggestion = " ".join(corr_list)
            else:
                state.suggestion = ""

            state.effective_text = state.suggestion if state.suggestion else current_word_str

            # GRU predictions
            eff_clean = state.effective_text.strip()
            if eff_clean:
                try:
                    state.next_word_preds = sentence_predictor.predict_next_words(eff_clean, top_k=2)
                    state.completed_sentence = sentence_predictor.complete_sentence(eff_clean, max_words=3)
                except Exception:
                    state.next_word_preds = []
                    state.completed_sentence = ""

                # Background translation
                def _do_trans(txt):
                    h = tts.translate(txt, lang="hi")
                    t = tts.translate(txt, lang="te")
                    with state.lock:
                        state.translated_hi = h
                        state.translated_te = t

                trans_executor.submit(_do_trans, eff_clean)
            else:
                state.next_word_preds = []
                state.completed_sentence = ""
                state.translated_hi = ""
                state.translated_te = ""

    return candidate_prediction, conf, landmarks_summary


def speak_curr_word():
    txt = word_builder.word.strip()
    if txt:
        target = txt.split()[-1]
        suggs = word_suggester.suggest(target, max_suggestions=1)
        w_to_speak = suggs[0] if suggs else target
        tts.speak(w_to_speak)


# ============================================================
# SINGLETON CAMERA MANAGER (Thread-safe, multi-client broadcast)
# ============================================================
class CameraManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.latest_jpeg = None
        self.frame_id = 0
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(0)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass

        last_time = time.time()
        blank_frame = None

        while self.running:
            with state.lock:
                active = state.camera_active

            if not active:
                if cap.isOpened():
                    cap.release()

                if blank_frame is None:
                    standby = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.rectangle(standby, (40, 160), (600, 320), (20, 24, 33), -1)
                    cv2.rectangle(standby, (40, 160), (600, 320), (55, 65, 81), 2)
                    cv2.putText(standby, "CAMERA CLOSED", (175, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (230, 230, 235), 2)
                    cv2.putText(standby, "Click 'Open Camera' or press 'C' to start", (125, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (156, 163, 175), 1)
                    _, jpeg = cv2.imencode(".jpg", standby, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    blank_frame = jpeg.tobytes()

                with self.lock:
                    self.latest_jpeg = blank_frame
                    self.frame_id += 1

                time.sleep(0.2)
                continue
            else:
                blank_frame = None

            if not cap.isOpened():
                cap.open(0)
                time.sleep(0.5)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            process_landmarks_and_predict(rgb, frame_bgr=frame)

            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-4)
            last_time = now
            with state.lock:
                state.fps = round(fps, 1)
                c_sign = state.current_sign
                c_conf = state.confidence
                c_hands = state.detected_hands
                c_word = word_builder.word

            # Top HUD Bar
            cv2.rectangle(frame, (0, 0), (640, 48), (20, 20, 30), -1)
            cv2.putText(frame, f"SIGN: {c_sign or '-'}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 128), 2)
            cv2.putText(frame, f"CONF: {c_conf:.0f}%", (180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"HANDS: {c_hands}", (340, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2)
            cv2.putText(frame, f"{state.fps} FPS", (530, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 255), 2)

            # Bottom Status Bar
            cv2.rectangle(frame, (0, 435), (640, 480), (20, 20, 30), -1)
            cv2.putText(frame, f"Text: {c_word[:32]}", (15, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpeg.tobytes()

            with self.lock:
                self.latest_jpeg = jpeg_bytes
                self.frame_id += 1

            time.sleep(0.015)

        cap.release()

    def get_stream(self):
        last_id = -1
        while self.running:
            frame_bytes = None
            with self.lock:
                if self.frame_id != last_id and self.latest_jpeg is not None:
                    last_id = self.frame_id
                    frame_bytes = self.latest_jpeg

            if frame_bytes:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.02)
            else:
                time.sleep(0.015)

camera_manager = CameraManager()


# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/video_feed")
def video_feed():
    """MJPEG Live Camera Stream with overlay (multi-client safe broadcast)."""
    return StreamingResponse(
        camera_manager.get_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/info")
def get_info():
    """Returns local network LAN IP and pairing URL for mobile phones."""
    return {
        "lan_ip": LAN_IP,
        "port": DEFAULT_PORT,
        "phone_url": f"http://{LAN_IP}:{DEFAULT_PORT}",
        "author": "Md Dilnawaz Ahmad",
        "mentor": "Dr Korra Sathya Babu",
        "institution": "IIITDM Kurnool"
    }


@app.get("/api/telemetry")
def get_telemetry():
    """Retrieve full real-time state for UI synchronizer."""
    with state.lock:
        return {
            "sign": state.current_sign,
            "confidence": round(state.confidence, 1),
            "hands": state.detected_hands,
            "rule_message": state.rule_message,
            "word": word_builder.word,
            "suggestion": state.suggestion,
            "effective_text": state.effective_text,
            "translated_hi": state.translated_hi,
            "translated_te": state.translated_te,
            "next_word_preds": state.next_word_preds,
            "completed_sentence": state.completed_sentence,
            "fps": state.fps,
            "language": state.active_lang,
            "camera_active": state.camera_active,
        }


class ActionRequest(BaseModel):
    action: str
    payload: Optional[str] = ""


@app.post("/api/action")
def perform_action(req: ActionRequest):
    """Execute action from web controls or mobile buttons."""
    act = req.action.lower()
    with state.lock:
        if act == "delete":
            word_builder.delete_last()
        elif act == "clear":
            word_builder.clear()
            state.prediction_history.clear()
            state.current_sign = ""
        elif act == "space":
            speak_curr_word()
            word_builder.add_space()
        elif act == "accept_autocomplete":
            if state.completed_sentence:
                word_builder.word = state.completed_sentence.upper()
        elif act == "append_prediction":
            if req.payload:
                word_builder.word = (word_builder.word.strip() + " " + req.payload.upper()).strip() + " "
        elif act == "set_language":
            if req.payload in ["hi", "te", "en", "ta"]:
                state.active_lang = req.payload
                tts.set_language(req.payload)
        elif act in ["toggle_camera", "toggle_cam"]:
            state.camera_active = not state.camera_active
        elif act in ["open_camera", "start_camera"]:
            state.camera_active = True
        elif act in ["close_camera", "stop_camera"]:
            state.camera_active = False
        elif act == "speak_sentence":
            text_to_speak = state.effective_text or word_builder.word
            if text_to_speak.strip():
                tts.speak(text_to_speak, lang=state.active_lang)

    return {"status": "ok", "current_word": word_builder.word, "camera_active": state.camera_active}


class FrameData(BaseModel):
    image_base64: str
    is_front_camera: Optional[bool] = True


@app.post("/api/process_browser_frame")
def process_browser_frame(data: FrameData):
    """
    Directly processes camera frames captured by mobile browsers via WebRTC/Canvas!
    Returns recognized sign, probabilities, hand count, and landmark coordinates.
    """
    try:
        header, encoded = data.image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return {"error": "Invalid frame buffer"}

        # Flip horizontally only if front/selfie camera
        if data.is_front_camera:
            frame_bgr = cv2.flip(frame_bgr, 1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        sign, conf, landmarks = process_landmarks_and_predict(frame_rgb)

        with state.lock:
            return {
                "sign": state.current_sign or sign,
                "confidence": round(conf, 1),
                "hands": state.detected_hands,
                "rule_message": state.rule_message,
                "word": word_builder.word,
                "suggestion": state.suggestion,
                "effective_text": state.effective_text,
                "translated_hi": state.translated_hi,
                "translated_te": state.translated_te,
                "next_word_preds": state.next_word_preds,
                "completed_sentence": state.completed_sentence,
                "landmarks": landmarks,
            }
    except Exception as e:
        return {"error": str(e)}


import asyncio

# ============================================================
# WEBSOCKET REAL-TIME SYNC
# ============================================================
@app.websocket("/ws")
async def websocket_telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with state.lock:
                payload = {
                    "sign": state.current_sign,
                    "confidence": round(state.confidence, 1),
                    "hands": state.detected_hands,
                    "rule_message": state.rule_message,
                    "word": word_builder.word,
                    "suggestion": state.suggestion,
                    "effective_text": state.effective_text,
                    "translated_hi": state.translated_hi,
                    "translated_te": state.translated_te,
                    "next_word_preds": state.next_word_preds,
                    "completed_sentence": state.completed_sentence,
                    "fps": state.fps,
                    "language": state.active_lang,
                }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ============================================================
# SPA FRONTEND TEMPLATE (MOBILE & DESKTOP ULTRA-RESPONSIVE)
# ============================================================
HTML_CONTENT = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="theme-color" content="#0b0f19">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>SignBridge // Mobile & Desktop ISL AI Translator</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- QRCode.js CDN for instant phone camera pairing -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        brand: {{
                            50: '#f0fdf4',
                            400: '#34d399',
                            500: '#10b981',
                            600: '#059669',
                            700: '#047857',
                        }},
                        darkslate: '#0f172a',
                        cardbg: '#1e293b',
                    }}
                }}
            }}
        }}
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@500;700&family=Noto+Sans+Telugu:wght@500;700&display=swap" rel="stylesheet">
    <style>
        body {{ 
            font-family: 'Plus Jakarta Sans', sans-serif;
            -webkit-tap-highlight-color: transparent;
        }}
        .font-hindi {{ font-family: 'Noto Sans Devanagari', sans-serif; }}
        .font-telugu {{ font-family: 'Noto Sans Telugu', sans-serif; }}
        .glass-panel {{
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .pulse-dot {{
            animation: pulseGlow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}
        @keyframes pulseGlow {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.4; transform: scale(1.15); }}
        }}
        /* Prevent elastic pull-to-refresh on mobile devices */
        html, body {{
            overscroll-behavior-y: contain;
        }}
    </style>
</head>
<body class="bg-[#0b0f19] text-slate-100 min-h-screen flex flex-col antialiased selection:bg-brand-500 selection:text-white pb-20 lg:pb-0">

    <!-- Top Navigation Bar -->
    <header class="border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur-md sticky top-0 z-40">
        <div class="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 h-14 sm:h-16 flex items-center justify-between">
            <div class="flex items-center space-x-2.5 sm:space-x-3">
                <div class="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-blue-500 flex items-center justify-center shadow-lg shadow-brand-500/20 shrink-0">
                    <span class="text-lg sm:text-xl">🤟</span>
                </div>
                <div>
                    <div class="flex items-center space-x-1.5 sm:space-x-2">
                        <span class="font-extrabold text-base sm:text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-brand-400 bg-clip-text text-transparent">SignBridge</span>
                        <span class="px-1.5 py-0.5 text-[10px] sm:text-xs font-semibold rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">Mobile & Web</span>
                    </div>
                    <p class="text-[11px] text-slate-400 hidden sm:block">Real-Time Indian Sign Language to Vernacular Speech</p>
                </div>
            </div>

            <!-- Authors & Academic Badge (Desktop) -->
            <div class="hidden md:flex flex-col text-right">
                <span class="text-xs font-medium text-slate-300">Md Dilnawaz Ahmad</span>
                <span class="text-[11px] text-slate-400">Under the Guidance of <strong class="text-brand-400 font-semibold">Dr Korra Sathya Babu</strong> • IIITDM Kurnool</span>
            </div>

            <!-- Quick Action Shortcuts -->
            <div class="flex items-center space-x-2 sm:space-x-3">
                <button onclick="toggleQRModal()" class="flex items-center space-x-1 px-2.5 py-1.5 text-xs font-semibold rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 transition">
                    <span>📱</span>
                    <span class="hidden sm:inline">Connect Phone</span>
                </button>
                <button onclick="toggleChartModal()" class="px-2.5 py-1.5 text-xs font-semibold rounded-lg bg-brand-600 hover:bg-brand-500 text-white shadow-md shadow-brand-600/20 transition">
                    📖 <span class="hidden sm:inline">ISL Chart</span>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Content Grid -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">

        <!-- Left Column: Video Viewport & Mobile Camera Feeds (7 Cols) -->
        <div class="lg:col-span-7 flex flex-col space-y-3 sm:space-y-4">
            
            <!-- Video Container -->
            <div class="relative glass-panel rounded-2xl overflow-hidden aspect-[4/3] sm:aspect-video flex items-center justify-center shadow-2xl border border-slate-800 bg-black">
                
                <!-- MJPEG Video Element (Laptop/Server Camera) -->
                <img id="live-camera-feed" src="/video_feed" alt="Live Sign Language Camera Stream" class="w-full h-full object-cover" onerror="setTimeout(reconnectLiveFeed, 1000)">

                <!-- Client Device Video Element (Phone Camera) -->
                <video id="client-webcam" class="hidden w-full h-full object-cover" autoplay playsinline muted></video>
                <canvas id="client-canvas" class="hidden"></canvas>
                <canvas id="overlay-canvas" class="absolute inset-0 pointer-events-none w-full h-full"></canvas>

                <!-- Interactive Camera Off / Privacy Screen Overlay -->
                <div id="camera-off-overlay" class="hidden absolute inset-0 bg-slate-950/92 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-20">
                    <div class="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-700/80 flex items-center justify-center text-3xl mb-3 shadow-xl">
                        🔒
                    </div>
                    <h3 class="text-base font-bold text-slate-100 mb-1">Camera is Closed</h3>
                    <p class="text-xs text-slate-400 max-w-xs mb-4">Hardware video capture is paused. Click below to start recognition.</p>
                    <button onclick="toggleCameraPower()" class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs shadow-lg shadow-emerald-500/20 active:scale-95 transition flex items-center space-x-2">
                        <span>🟢</span>
                        <span>Open Camera (Press 'C')</span>
                    </button>
                </div>

                <!-- Floating Video HUD Overlay -->
                <div class="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
                    <div class="px-2.5 py-1 sm:px-3 sm:py-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/10 text-[11px] sm:text-xs flex items-center space-x-1.5">
                        <span id="hud-camera-dot" class="w-2 h-2 rounded-full bg-emerald-400 pulse-dot"></span>
                        <span id="hud-camera-label" class="font-semibold text-slate-200">Server Camera</span>
                    </div>

                    <div id="hud-rule-badge" class="px-2.5 py-1 sm:px-3 sm:py-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/10 text-[11px] sm:text-xs font-medium text-slate-300">
                        Detecting...
                    </div>
                </div>

                <!-- Floating Bottom HUD Bar -->
                <div class="absolute bottom-3 left-3 right-3 flex items-center justify-between bg-black/70 backdrop-blur-md px-3 py-2 rounded-xl border border-white/10 text-xs z-10">
                    <div class="flex items-center space-x-2">
                        <span class="text-[10px] sm:text-xs uppercase font-bold text-slate-400">Topology:</span>
                        <span id="hud-hands-count" class="text-[11px] sm:text-xs font-semibold text-brand-400 bg-brand-500/10 px-2 py-0.5 rounded-md border border-brand-500/20">
                            0 Hands
                        </span>
                    </div>

                    <div class="flex items-center space-x-1 sm:space-x-2">
                        <span id="fps-badge" class="font-mono font-semibold text-brand-400 bg-slate-900/80 px-2 py-0.5 rounded border border-slate-700 text-[11px]">60 FPS</span>
                    </div>
                </div>
            </div>

            <!-- Camera Switch & Power Controls Toolbar -->
            <div class="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
                <button id="btn-toggle-cam-power" onclick="toggleCameraPower()" class="p-2.5 rounded-xl bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/40 font-semibold flex items-center justify-center space-x-1.5 transition">
                    <span id="cam-power-icon">🔴</span>
                    <span id="cam-power-text">Close Cam [C]</span>
                </button>
                <button id="btn-toggle-source" onclick="toggleCameraSource()" class="p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold flex items-center justify-center space-x-1.5 transition">
                    <span id="source-icon">📷</span>
                    <span id="source-text">Use Device Cam</span>
                </button>
                <button id="btn-flip-cam" onclick="flipPhoneCamera()" class="p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold flex items-center justify-center space-x-1.5 transition hidden">
                    <span>🔄</span>
                    <span>Flip</span>
                </button>
                <button onclick="triggerAction('delete')" class="p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-semibold flex items-center justify-center space-x-1 transition">
                    <span>⌫</span>
                    <span>Backspace</span>
                </button>
                <button onclick="triggerAction('clear')" class="p-2.5 rounded-xl bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 border border-rose-800/40 font-semibold flex items-center justify-center space-x-1 transition">
                    <span>🗑️</span>
                    <span>Clear All</span>
                </button>
            </div>

            <!-- Voice Language Selector Banner -->
            <div class="flex items-center justify-between p-3 rounded-xl glass-panel text-xs text-slate-300">
                <span class="font-semibold text-slate-400">🗣️ Audio Output Voice:</span>
                <select id="voice-lang-select" onchange="changeVoiceLanguage(this.value)" class="bg-slate-800 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1 text-xs focus:ring-1 focus:ring-brand-500 outline-none">
                    <option value="hi" selected>🇮🇳 Hindi (हिन्दी)</option>
                    <option value="te">🌾 Telugu (తెలుగు)</option>
                    <option value="en">🇬🇧 English</option>
                    <option value="ta">🌴 Tamil (தமிழ்)</option>
                </select>
            </div>

        </div>

        <!-- Right Column: Live Recognition, GRU Autocomplete & Vernacular Voice (5 Cols) -->
        <div class="lg:col-span-5 flex flex-col space-y-3 sm:space-y-4">
            
            <!-- Live Sign & Confidence Card -->
            <div class="p-4 sm:p-5 rounded-2xl glass-panel border border-slate-800 flex items-center justify-between shadow-lg">
                <div>
                    <span class="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-400">Detected Sign</span>
                    <div class="flex items-baseline space-x-3 mt-0.5 sm:mt-1">
                        <span id="current-sign-badge" class="text-4xl sm:text-5xl font-black text-brand-400 font-mono tracking-tighter">-</span>
                        <span id="confidence-percentage" class="text-xs sm:text-sm font-semibold text-slate-400">0% Conf</span>
                    </div>
                </div>

                <div class="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-brand-500/10 border border-brand-500/30 flex flex-col items-center justify-center">
                    <span class="text-xl sm:text-2xl">🖐️</span>
                    <span id="hand-count-mini" class="text-[9px] sm:text-[10px] font-bold text-brand-400 mt-0.5">0 Hand</span>
                </div>
            </div>

            <!-- Assembled Sentence Box -->
            <div class="p-4 sm:p-5 rounded-2xl glass-panel border border-slate-800 flex flex-col space-y-2.5 sm:space-y-3 shadow-lg">
                <div class="flex items-center justify-between">
                    <span class="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-400">English Text</span>
                    <button onclick="triggerAction('speak_sentence')" class="flex items-center space-x-1 px-2.5 py-1 text-xs font-semibold rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 transition">
                        <span>🔊</span>
                        <span>Speak [S]</span>
                    </button>
                </div>

                <div class="min-h-[56px] sm:min-h-[64px] p-3 rounded-xl bg-slate-900/90 border border-slate-800 font-mono text-base sm:text-lg text-emerald-400 flex items-center break-all">
                    <span id="assembled-text-display">Start signing...</span>
                    <span class="w-2 h-5 bg-emerald-400 ml-1.5 animate-pulse shrink-0"></span>
                </div>

                <!-- Fuzzy Spell Correction Badge -->
                <div id="spell-correction-wrapper" class="hidden items-center space-x-2 text-xs text-purple-400 bg-purple-500/10 px-3 py-1.5 rounded-lg border border-purple-500/20">
                    <span class="font-bold">✨ Suggestion:</span>
                    <span id="spell-correction-text" class="font-semibold"></span>
                </div>
            </div>

            <!-- Vernacular Native Translation Cards (Hindi & Telugu) -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 sm:gap-3">
                
                <!-- Hindi Card -->
                <div class="p-3.5 sm:p-4 rounded-xl glass-panel border border-amber-500/20 bg-gradient-to-br from-amber-500/5 to-transparent flex flex-col justify-between space-y-1.5">
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] sm:text-[11px] font-bold uppercase tracking-wide text-amber-400">🇮🇳 हिन्दी Translation</span>
                        <button onclick="speakDirect(state.translated_hi, 'hi')" class="text-xs p-1 hover:scale-110 transition text-amber-300">🔊</button>
                    </div>
                    <div id="trans-hindi-display" class="font-hindi text-sm sm:text-base font-semibold text-amber-200 min-h-[28px] flex items-center">
                        प्रतीक्षा...
                    </div>
                </div>

                <!-- Telugu Card -->
                <div class="p-3.5 sm:p-4 rounded-xl glass-panel border border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-transparent flex flex-col justify-between space-y-1.5">
                    <div class="flex items-center justify-between">
                        <span class="text-[10px] sm:text-[11px] font-bold uppercase tracking-wide text-emerald-400">🌾 తెలుగు Translation</span>
                        <button onclick="speakDirect(state.translated_te, 'te')" class="text-xs p-1 hover:scale-110 transition text-emerald-300">🔊</button>
                    </div>
                    <div id="trans-telugu-display" class="font-telugu text-sm sm:text-base font-semibold text-emerald-200 min-h-[28px] flex items-center">
                        వేచి ఉంది...
                    </div>
                </div>

            </div>

            <!-- Deep Learning GRU Neural Autocomplete & Suggestions -->
            <div class="p-4 rounded-2xl glass-panel border border-slate-800 flex flex-col space-y-2.5">
                <div class="flex items-center justify-between">
                    <span class="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-400">🧠 PyTorch GRU Predictions</span>
                    <span class="text-[10px] text-slate-500 font-mono">Top-2</span>
                </div>

                <!-- Next Word Chips -->
                <div id="next-word-chips-container" class="flex flex-wrap gap-1.5 sm:gap-2">
                    <span class="text-xs text-slate-500">Sign words to see predictions...</span>
                </div>

                <!-- Full Auto-Complete Banner -->
                <div id="autocomplete-banner" class="hidden p-2.5 sm:p-3 rounded-xl bg-slate-900 border border-slate-700/60 flex items-center justify-between">
                    <div class="flex items-center space-x-1.5 text-xs overflow-hidden">
                        <span class="text-amber-400 shrink-0">⚡</span>
                        <span id="autocomplete-full-text" class="font-semibold text-slate-200 truncate"></span>
                    </div>
                    <button onclick="triggerAction('accept_autocomplete')" class="ml-2 px-2.5 py-1 text-xs font-bold rounded-lg bg-brand-600 hover:bg-brand-500 text-white shrink-0 transition">
                        Accept [P]
                    </button>
                </div>
            </div>

            <!-- Main Big Action Button (Desktop) -->
            <button onclick="triggerAction('space')" class="hidden lg:flex w-full py-3.5 rounded-xl bg-gradient-to-r from-brand-600 via-emerald-600 to-teal-600 hover:from-brand-500 hover:to-teal-500 text-white font-bold text-sm shadow-lg shadow-brand-500/20 active:scale-[0.99] transition items-center justify-center space-x-2">
                <span>␣</span>
                <span>Space & Speak Current Word</span>
            </button>

        </div>

    </main>

    <!-- Mobile Fixed Bottom Action Dock (Sticky for thumb accessibility) -->
    <div class="lg:hidden fixed bottom-0 left-0 right-0 z-40 bg-[#0f172a]/95 backdrop-blur-lg border-t border-slate-800 px-3 py-2 flex items-center justify-between space-x-2 shadow-2xl">
        <button onclick="triggerAction('delete')" class="flex-1 py-2.5 rounded-xl bg-slate-800 active:bg-slate-700 text-slate-300 font-bold text-xs border border-slate-700 flex flex-col items-center">
            <span>⌫</span>
            <span class="text-[9px]">Del</span>
        </button>
        <button onclick="triggerAction('space')" class="flex-[2.5] py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-emerald-600 active:from-brand-500 active:to-emerald-500 text-white font-bold text-xs shadow-md shadow-brand-500/20 flex flex-col items-center">
            <span>␣ SPACE</span>
            <span class="text-[9px] font-normal opacity-90">Speak Word</span>
        </button>
        <button onclick="triggerAction('accept_autocomplete')" class="flex-1 py-2.5 rounded-xl bg-slate-800 active:bg-slate-700 text-amber-300 font-bold text-xs border border-slate-700 flex flex-col items-center">
            <span>⚡</span>
            <span class="text-[9px]">Auto</span>
        </button>
        <button onclick="triggerAction('speak_sentence')" class="flex-1 py-2.5 rounded-xl bg-blue-600/30 active:bg-blue-600/40 text-blue-300 font-bold text-xs border border-blue-500/40 flex flex-col items-center">
            <span>🔊</span>
            <span class="text-[9px]">All</span>
        </button>
    </div>

    <!-- Modal: Phone Pairing QR Code -->
    <div id="qr-modal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm hidden items-center justify-center p-4">
        <div class="max-w-sm w-full glass-panel rounded-2xl border border-slate-700 p-6 flex flex-col items-center space-y-4 text-center">
            <div class="w-full flex justify-between items-center border-b border-slate-800 pb-3">
                <h3 class="font-bold text-base text-slate-100 flex items-center space-x-2">
                    <span>📱</span>
                    <span>Pair Phone via Wi-Fi</span>
                </h3>
                <button onclick="toggleQRModal()" class="text-slate-400 hover:text-white">✕</button>
            </div>

            <p class="text-xs text-slate-300">
                Connect your phone to the same Wi-Fi or hotspot and scan this QR code with your phone camera:
            </p>

            <!-- QR Code Render Container -->
            <div id="qrcode" class="p-3 bg-white rounded-xl shadow-xl flex items-center justify-center"></div>

            <div id="qr-url-text" class="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-center font-mono text-xs text-brand-400 select-all">
                http://{LAN_IP}:{DEFAULT_PORT}
            </div>

            <p class="text-[11px] text-slate-400">
                Tip: You can use your phone camera directly by tapping "Use Phone Cam" inside the webapp on your phone!
            </p>
        </div>
    </div>

    <!-- Modal: ISL Chart Reference -->
    <div id="chart-modal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm hidden items-center justify-center p-4">
        <div class="max-w-2xl w-full glass-panel rounded-2xl border border-slate-700 p-5 sm:p-6 flex flex-col space-y-4 max-h-[85vh] overflow-y-auto">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 class="font-bold text-base sm:text-lg text-slate-100 flex items-center space-x-2">
                    <span>📖</span>
                    <span>Indian Sign Language (ISL) Chart</span>
                </h3>
                <button onclick="toggleChartModal()" class="text-slate-400 hover:text-white">✕</button>
            </div>

            <div class="space-y-4 text-xs">
                <div>
                    <h4 class="font-bold text-amber-400 uppercase tracking-wide mb-2">Two-Handed Signs (C_two - 20 Classes)</h4>
                    <p class="text-slate-400 mb-2">Requires both hands in camera view:</p>
                    <div class="flex flex-wrap gap-1.5 font-mono">
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">A</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">B</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">D</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">E</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">F</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">G</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">H</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">J</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">K</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">M</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">N</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">P</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">Q</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">R</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">S</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">T</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">W</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">X</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">Y</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-amber-300 font-bold">Z</span>
                    </div>
                </div>

                <div>
                    <h4 class="font-bold text-emerald-400 uppercase tracking-wide mb-2">Single-Handed Signs (C_one - 15 Classes)</h4>
                    <p class="text-slate-400 mb-2">Executed strictly with one hand:</p>
                    <div class="flex flex-wrap gap-1.5 font-mono">
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">C</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">I</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">L</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">O</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">U</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">V</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">1</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">2 (Trigger)</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">3</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">4</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">5</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">6</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">7</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">8</span>
                        <span class="px-2 py-1 rounded bg-slate-800 text-emerald-300 font-bold">9</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Client-Side JavaScript Logic -->
    <script>
        let ws = null;
        let isPhoneCameraMode = false;
        let currentFacingMode = "user"; // 'user' (selfie) or 'environment' (rear)
        let phoneStream = null;
        let frameInterval = null;

        // Auto-detect if user is visiting from mobile phone
        const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

        // Initialize WebSockets for real-time telemetry
        function initWebSocket() {{
            const loc = window.location;
            const wsUri = (loc.protocol === "https:" ? "wss://" : "ws://") + loc.host + "/ws";
            ws = new WebSocket(wsUri);

            ws.onmessage = (event) => {{
                const data = JSON.parse(event.data);
                updateUI(data);
            }};

            ws.onclose = () => {{
                setTimeout(initWebSocket, 1500);
            }};
        }}

        // Update UI DOM with incoming telemetry
        function updateUI(data) {{
            document.getElementById("current-sign-badge").innerText = data.sign || "-";
            document.getElementById("confidence-percentage").innerText = data.confidence > 0 ? (data.confidence + "% Conf") : "Searching...";
            document.getElementById("hand-count-mini").innerText = data.hands + (data.hands === 1 ? " Hand" : " Hands");
            document.getElementById("hud-hands-count").innerText = data.hands + (data.hands === 1 ? " Hand" : " Hands");
            document.getElementById("hud-rule-badge").innerText = data.rule_message || "Active";
            document.getElementById("fps-badge").innerText = (data.fps || 30) + " FPS";

            // Assembled Text
            const txt = data.word || "";
            document.getElementById("assembled-text-display").innerText = txt || "Start signing...";

            // Spell Correction
            const corrWrap = document.getElementById("spell-correction-wrapper");
            const corrTxt = document.getElementById("spell-correction-text");
            if (data.suggestion && data.suggestion.toLowerCase() !== txt.toLowerCase().trim()) {{
                corrWrap.classList.remove("hidden");
                corrWrap.classList.add("flex");
                corrTxt.innerText = data.suggestion;
            }} else {{
                corrWrap.classList.add("hidden");
                corrWrap.classList.remove("flex");
            }}

            // Vernacular Translations
            document.getElementById("trans-hindi-display").innerText = data.translated_hi || "प्रतीक्षा...";
            document.getElementById("trans-telugu-display").innerText = data.translated_te || "వేచి ఉంది...";

            // Next-Word GRU Chips
            const chipsCont = document.getElementById("next-word-chips-container");
            if (data.next_word_preds && data.next_word_preds.length > 0) {{
                chipsCont.innerHTML = data.next_word_preds.map(item => {{
                    const word = item[0];
                    const prob = Math.round(item[1] * 100);
                    return `<button onclick="appendWord('${{word}}')" class="px-2.5 py-1 text-xs font-semibold rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 transition flex items-center space-x-1">
                                <span>${{word}}</span>
                                <span class="text-[10px] text-emerald-400/80 font-mono">(${{prob}}%)</span>
                            </button>`;
                }}).join("");
            }} else {{
                chipsCont.innerHTML = `<span class="text-xs text-slate-500">Sign words to see predictions...</span>`;
            }}

            // Autocomplete Banner
            const autoBanner = document.getElementById("autocomplete-banner");
            const autoTxt = document.getElementById("autocomplete-full-text");
            if (data.completed_sentence && data.completed_sentence.toLowerCase() !== txt.toLowerCase().trim()) {{
                autoBanner.classList.remove("hidden");
                autoTxt.innerText = data.completed_sentence;
            }} else {{
                autoBanner.classList.add("hidden");
            }}

            // Camera Power State UI Sync
            const isCamActive = (data.camera_active !== false);
            const offOverlay = document.getElementById("camera-off-overlay");
            const btnPower = document.getElementById("btn-toggle-cam-power");
            const iconPower = document.getElementById("cam-power-icon");
            const textPower = document.getElementById("cam-power-text");
            const camDot = document.getElementById("hud-camera-dot");
            const camLabel = document.getElementById("hud-camera-label");

            if (isCamActive) {{
                if (offOverlay) offOverlay.classList.add("hidden");
                if (iconPower) iconPower.innerText = "🔴";
                if (textPower) textPower.innerText = "Close Cam [C]";
                if (btnPower) {{
                    btnPower.classList.remove("bg-emerald-600/20", "text-emerald-300", "border-emerald-500/40");
                    btnPower.classList.add("bg-rose-600/20", "text-rose-300", "border-rose-500/40");
                }}
                if (camDot) {{
                    camDot.className = "w-2 h-2 rounded-full bg-emerald-400 pulse-dot";
                }}
                if (camLabel && !isPhoneCameraMode) {{
                    camLabel.innerText = "Server Camera";
                }}
            }} else {{
                if (offOverlay) offOverlay.classList.remove("hidden");
                if (iconPower) iconPower.innerText = "🟢";
                if (textPower) textPower.innerText = "Open Cam [C]";
                if (btnPower) {{
                    btnPower.classList.remove("bg-rose-600/20", "text-rose-300", "border-rose-500/40");
                    btnPower.classList.add("bg-emerald-600/20", "text-emerald-300", "border-emerald-500/40");
                }}
                if (camDot) {{
                    camDot.className = "w-2 h-2 rounded-full bg-slate-500";
                }}
                if (camLabel && !isPhoneCameraMode) {{
                    camLabel.innerText = "Camera Closed";
                }}
            }}
        }}

        // Toggle camera hardware capture on/off
        async function toggleCameraPower() {{
            if (navigator.vibrate) navigator.vibrate(30);
            if (isPhoneCameraMode) {{
                stopPhoneCamera();
            }}
            try {{
                const resp = await fetch("/api/action", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ action: "toggle_camera" }})
                }});
                if (resp.ok) {{
                    const res = await resp.json();
                    if (res.camera_active) {{
                        reconnectLiveFeed();
                    }}
                }}
            }} catch (e) {{
                console.error("Camera toggle error:", e);
            }}
        }}

        // Trigger action via REST API
        async function triggerAction(actionName, payload = "") {{
            if (navigator.vibrate) navigator.vibrate(30);
            try {{
                await fetch("/api/action", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ action: actionName, payload: payload }})
                }});
            }} catch (e) {{
                console.error("Action error:", e);
            }}
        }}

        function appendWord(word) {{
            triggerAction("append_prediction", word);
        }}

        function changeVoiceLanguage(lang) {{
            triggerAction("set_language", lang);
        }}

        // Direct browser voice synthesis (plays through phone speaker)
        function speakDirect(text, lang) {{
            if (!text || text === "प्रतीक्षा..." || text === "వేచి ఉంది...") return;
            if (navigator.vibrate) navigator.vibrate(40);
            if ('speechSynthesis' in window) {{
                const utterance = new SpeechSynthesisUtterance(text);
                if (lang === 'hi') utterance.lang = 'hi-IN';
                else if (lang === 'te') utterance.lang = 'te-IN';
                else utterance.lang = 'en-US';
                window.speechSynthesis.speak(utterance);
            }}
        }}

        // --------------------------------------------------------
        // PHONE CAMERA CAPTURE (WebRTC & Canvas Streaming)
        // --------------------------------------------------------
        function reconnectLiveFeed() {{
            const liveFeed = document.getElementById("live-camera-feed");
            const videoEl = document.getElementById("client-webcam");
            if (videoEl) videoEl.classList.add("hidden");
            if (liveFeed) {{
                liveFeed.classList.remove("hidden");
                liveFeed.src = "/video_feed?t=" + new Date().getTime();
            }}
        }}

        async function toggleCameraSource() {{
            if (navigator.vibrate) navigator.vibrate(30);

            if (!isPhoneCameraMode) {{
                // Check if browser supports mediaDevices
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
                    alert("Notice: Direct device camera access requires HTTPS or localhost.\\n\\nYou can continue using the Live Server Camera view!");
                    reconnectLiveFeed();
                    return;
                }}

                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{
                            facingMode: {{ ideal: currentFacingMode }},
                            width: {{ ideal: 480 }},
                            height: {{ ideal: 360 }}
                        }},
                        audio: false
                    }});

                    phoneStream = stream;
                    const videoEl = document.getElementById("client-webcam");
                    videoEl.srcObject = stream;
                    videoEl.muted = true;
                    videoEl.playsInline = true;
                    videoEl.setAttribute("playsinline", "");
                    videoEl.setAttribute("webkit-playsinline", "");
                    await videoEl.play();

                    videoEl.classList.remove("hidden");
                    document.getElementById("live-camera-feed").classList.add("hidden");
                    document.getElementById("hud-camera-label").innerText = "Device Camera (" + (currentFacingMode === 'user' ? 'Front' : 'Rear') + ")";

                    document.getElementById("source-icon").innerText = "💻";
                    document.getElementById("source-text").innerText = "Use Server Cam";
                    document.getElementById("btn-flip-cam").classList.remove("hidden");
                    isPhoneCameraMode = true;

                    startPhoneFrameLoop();
                }} catch (err) {{
                    console.error("Camera access error:", err);
                    alert("Could not access device camera (" + (err.name || "Error") + ": " + (err.message || "denied") + ").\\n\\nReturning to Server Camera feed.");
                    stopPhoneCamera();
                }}
            }} else {{
                // Switch back to Server Camera Mode
                stopPhoneCamera();
            }}
        }}

        function stopPhoneCamera() {{
            if (phoneStream) {{
                try {{
                    phoneStream.getTracks().forEach(track => track.stop());
                }} catch (e) {{}}
                phoneStream = null;
            }}
            if (frameInterval) {{
                clearInterval(frameInterval);
                frameInterval = null;
            }}

            const videoEl = document.getElementById("client-webcam");
            if (videoEl) {{
                videoEl.classList.add("hidden");
                try {{
                    videoEl.pause();
                    videoEl.srcObject = null;
                }} catch (e) {{}}
            }}

            reconnectLiveFeed();

            document.getElementById("hud-camera-label").innerText = "Server Camera Feed";
            document.getElementById("source-icon").innerText = "📷";
            document.getElementById("source-text").innerText = "Use Device Cam";
            document.getElementById("btn-flip-cam").classList.add("hidden");
            isPhoneCameraMode = false;

            // Clear overlay canvas
            const overlay = document.getElementById("overlay-canvas");
            if (overlay) {{
                const ctx = overlay.getContext("2d");
                ctx.clearRect(0, 0, overlay.width, overlay.height);
            }}
        }}

        async function flipPhoneCamera() {{
            if (!isPhoneCameraMode) return;
            currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
            stopPhoneCamera();
            await toggleCameraSource();
        }}

        function startPhoneFrameLoop() {{
            const videoEl = document.getElementById("client-webcam");
            const canvasEl = document.getElementById("client-canvas");
            const ctx = canvasEl.getContext("2d");
            canvasEl.width = 320;
            canvasEl.height = 240;

            if (frameInterval) clearInterval(frameInterval);
            let inFlight = false;

            frameInterval = setInterval(async () => {{
                if (!isPhoneCameraMode || inFlight || videoEl.paused || videoEl.ended || videoEl.videoWidth === 0) return;
                inFlight = true;

                try {{
                    ctx.drawImage(videoEl, 0, 0, 320, 240);
                    const b64 = canvasEl.toDataURL("image/jpeg", 0.60);

                    const resp = await fetch("/api/process_browser_frame", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify({{ 
                            image_base64: b64,
                            is_front_camera: (currentFacingMode === "user")
                        }})
                    }});

                    if (resp.ok) {{
                        const data = await resp.json();
                        if (data && !data.error) {{
                            updateUI(data);
                            drawLandmarksOverlay(data.landmarks);
                        }}
                    }}
                }} catch (e) {{
                    console.error("Frame send error:", e);
                }} finally {{
                    inFlight = false;
                }}
            }}, 90);
        }}

        function drawLandmarksOverlay(landmarks) {{
            const overlay = document.getElementById("overlay-canvas");
            const ctx = overlay.getContext("2d");
            overlay.width = overlay.clientWidth;
            overlay.height = overlay.clientHeight;
            ctx.clearRect(0, 0, overlay.width, overlay.height);

            if (!landmarks || landmarks.length === 0) return;

            landmarks.forEach(hand => {{
                if (hand.points) {{
                    ctx.fillStyle = "#10b981";
                    ctx.strokeStyle = "#34d399";
                    ctx.lineWidth = 2;

                    hand.points.forEach(pt => {{
                        const px = pt.x * overlay.width;
                        const py = pt.y * overlay.height;
                        ctx.beginPath();
                        ctx.arc(px, py, 4, 0, 2 * Math.PI);
                        ctx.fill();
                    }});
                }}
            }});
        }}

        // --------------------------------------------------------
        // MODALS & QR CODE GENERATION
        // --------------------------------------------------------
        function toggleQRModal() {{
            if (navigator.vibrate) navigator.vibrate(30);
            const m = document.getElementById("qr-modal");
            m.classList.toggle("hidden");
            m.classList.toggle("flex");

            // Generate QR code if not already generated
            const qrContainer = document.getElementById("qrcode");
            const urlText = document.getElementById("qr-url-text");
            const targetUrl = window.location.origin;
            if (urlText) urlText.innerText = targetUrl;
            qrContainer.innerHTML = "";
            new QRCode(qrContainer, {{
                text: targetUrl,
                width: 180,
                height: 180,
                colorDark: "#0f172a",
                colorLight: "#ffffff",
                correctLevel: QRCode.CorrectLevel.M
            }});
        }}

        function toggleChartModal() {{
            if (navigator.vibrate) navigator.vibrate(30);
            const m = document.getElementById("chart-modal");
            m.classList.toggle("hidden");
            m.classList.toggle("flex");
        }}

        // Keyboard Shortcuts
        window.addEventListener("keydown", (e) => {{
            if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
            if (e.key === "c" || e.key === "C") {{
                e.preventDefault();
                toggleCameraPower();
            }} else if (e.key === " " || e.code === "Space") {{
                e.preventDefault();
                triggerAction("space");
            }} else if (e.key === "Backspace") {{
                e.preventDefault();
                triggerAction("delete");
            }} else if (e.key === "s" || e.key === "S") {{
                e.preventDefault();
                triggerAction("speak_sentence");
            }} else if (e.key === "p" || e.key === "P") {{
                e.preventDefault();
                triggerAction("accept_autocomplete");
            }}
        }});

        // If opened on mobile phone, suggest camera access directly
        if (isMobileDevice) {{
            document.getElementById("btn-toggle-source").classList.add("ring-2", "ring-brand-500");
        }}

        // Start WebSocket
        initWebSocket();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the main single-page application dashboard."""
    return HTML_CONTENT


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge WebApp - Mobile & Desktop")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (0.0.0.0 for all devices)")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS using self-signed cert")
    args = parser.parse_args()

    ssl_config = {}
    proto = "http"
    if args.ssl and os.path.exists("cert.pem") and os.path.exists("key.pem"):
        ssl_config = {
            "ssl_keyfile": "key.pem",
            "ssl_certfile": "cert.pem"
        }
        proto = "https"

    print("\n==================================================================")
    print(" 🤟 SignBridge Mobile & Desktop Web Application Starting...")
    print(" Author: Md Dilnawaz Ahmad")
    print(" Guidance: Dr Korra Sathya Babu (IIITDM Kurnool)")
    print(f" Protocol:      {proto.upper()}")
    print(f" Local PC URL:  {proto}://127.0.0.1:{args.port}")
    print(f" Mobile Phone:  {proto}://{LAN_IP}:{args.port}")
    print("==================================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **ssl_config)
