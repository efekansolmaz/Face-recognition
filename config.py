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

# CLAHE (Işık dengeleme) ayarları
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Esneme / Yanıltıcı Mimik Filtresi
ENABLE_YAWN_FILTER = True
YAWN_MOUTH_RATIO_THRESHOLD = 0.55

# --- FABRİKA ALANLARI / ÇOKLU KAMERA BÖLGELERİ ---
CAMERA_ZONES = [
    "Giriş Turnikesi",
    "Pres Hattı",
    "Montaj Hattı",
    "Kaynak Atölyesi",
    "Boyahane",
    "Yemekhane / Mola Alanı",
    "Depo & Sevkiyat"
]
DEFAULT_CAMERA_ZONE = os.getenv("CAMERA_ZONE", "Pres Hattı")

# RTSP Kamera Akışları (Opsiyonel IP kameralar)
RTSP_STREAMS = {
    "Pres Hattı": os.getenv("RTSP_PRES", ""),
    "Montaj Hattı": os.getenv("RTSP_MONTAJ", ""),
    "Yemekhane / Mola Alanı": os.getenv("RTSP_MOLA", "")
}

# --- DUYGU VE MORAL AĞIRLIKLARI (0.0 - 100.0 Endeksi) ---
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

# --- İSG VE ERKEN UYARI ALARM AYARLARI ---
CRITICAL_MORALE_THRESHOLD = 45.0   # Bu skorun altı "Kritik / Kötü Gün"
SAFETY_RISK_THRESHOLD = 65.0       # İş kazası risk skoru > 65 ise acil mola uyarısı
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # Slack / Telegram / Teams Webhook

# --- KVKK VE GİZLİLİK AYARLARI ---
DEFAULT_PRIVACY_MODE = False        # İsimleri ve yüzleri maskeleme modu
DATA_RETENTION_DAYS = 30           # 30 günden eski yüz fotoğraflarını temizleme

# --- WEB DASHBOARD AYARLARI ---
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
