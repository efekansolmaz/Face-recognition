import os

# --- TEMEL DİZİNLER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# --- MODEL YOLLARI ---
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")

# --- VERİTABANI AYARLARI ---
# Varsayılan olarak SQLite kullanılır (data/face_records.db dosyasına kalıcı kaydeder).
# PostgreSQL kullanmak isterseniz ortam değişkenine (DATABASE_URL) ya da buraya Postgres URI girebilirsiniz.
# Örnek: "postgresql+psycopg2://postgres:sifre@localhost:5432/facedb"
SQLITE_DB_PATH = os.path.join(DATA_DIR, "face_records.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")

# --- YÜZ VE DUYGU ALGORİTMA PARAMETRELERİ ---
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CONFIDENCE_THRESHOLD = 0.70  # Yüz tespiti eşik değeri (0.0 - 1.0)
NMS_THRESHOLD = 0.3          # Çakışan kutuları eleme eşiği
COSINE_SIMILARITY_THRESHOLD = 0.42  # SFace için yüz eşleşme eşiği (>= 0.42 ise aynı kişi)

# Duyguların Türkçe karşılıkları ve renk kodları (BGR formatında)
EMOTION_MAP = {
    0: {"name": "Notr", "tr": "Nötr", "color": (200, 200, 200)},
    1: {"name": "Happiness", "tr": "Mutlu", "color": (0, 255, 0)},
    2: {"name": "Surprise", "tr": "Şaşkın", "color": (0, 255, 255)},
    3: {"name": "Sadness", "tr": "Üzgün", "color": (255, 100, 0)},
    4: {"name": "Anger", "tr": "Öfkeli", "color": (0, 0, 255)},
    5: {"name": "Disgust", "tr": "İğrenmiş", "color": (128, 0, 128)},
    6: {"name": "Fear", "tr": "Korkmuş", "color": (255, 0, 255)},
    7: {"name": "Contempt", "tr": "Mesafeli", "color": (180, 180, 100)},
}

# Kaç saniyede bir aynı kişinin duygu durumu veritabanına loglansın?
EMOTION_LOG_INTERVAL_SEC = 1.0
