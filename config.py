import os

# --- TEMEL DİZİNLER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- MODEL YOLLARI ---
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")

# --- VERİTABANI AYARLARI ---
SQLITE_DB_PATH = os.path.join(DATA_DIR, "face_records.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")

# --- KAMERA & ALGORİTMA PARAMETRELERİ ---
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CONFIDENCE_THRESHOLD = 0.70  # Yüz tespiti eşiği
NMS_THRESHOLD = 0.3          # Çakışan kutu eleme
COSINE_SIMILARITY_THRESHOLD = 0.42  # SFace için benzerlik eşiği (>= 0.42 ise aynı kişi)

# --- DUYGU VE MORAL AĞIRLIKLARI (0.0 - 100.0 Endeksi) ---
# Ağırlık faktörleri: +1.0 (en pozitif), -1.0 (en stresli/negatif)
EMOTION_MAP = {
    0: {"name": "Notr", "tr": "Nötr", "color": (200, 200, 200), "weight": 0.0},
    1: {"name": "Happiness", "tr": "Mutlu", "color": (0, 255, 0), "weight": 1.0},
    2: {"name": "Surprise", "tr": "Şaşkın", "color": (0, 255, 255), "weight": 0.2},
    3: {"name": "Sadness", "tr": "Üzgün", "color": (255, 100, 0), "weight": -0.8},
    4: {"name": "Anger", "tr": "Öfkeli", "color": (0, 0, 255), "weight": -1.0},
    5: {"name": "Disgust", "tr": "İğrenmiş", "color": (128, 0, 128), "weight": -0.5},
    6: {"name": "Fear", "tr": "Korkmuş", "color": (255, 0, 255), "weight": -0.7},
    7: {"name": "Contempt", "tr": "Mesafeli", "color": (180, 180, 100), "weight": -0.4},
}

# Kaç saniyede bir aynı işçinin duygu durumu veritabanına loglansın?
EMOTION_LOG_INTERVAL_SEC = 1.0

# Fabrika Departmanları
DEPARTMENTS = [
    "Genel",
    "Üretim / Montaj",
    "Pres Hattı",
    "Kaynak Atölyesi",
    "Boyahane",
    "Depo & Lojistik",
    "Kalite Kontrol",
    "Bakım & Onarım"
]

# Web Dashboard Portu
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
