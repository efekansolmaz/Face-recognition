import cv2
import numpy as np
from config import (
    EMOTION_MODEL_PATH,
    EMOTION_MAP,
    ENABLE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    ENABLE_YAWN_FILTER
)

class EmotionRecognizer:
    def __init__(self, model_path: str = EMOTION_MODEL_PATH):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        if ENABLE_CLAHE:
            self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        else:
            self.clahe = None

    def check_yawn_or_fatigue(self, face_landmarks: np.ndarray) -> bool:
        """
        YuNet 5 nirengi noktası üzerinden esneme ve aşırı açık ağız kontrolü yapar.
        landmarks: [x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc]
        """
        if not ENABLE_YAWN_FILTER or face_landmarks is None or len(face_landmarks) < 10:
            return False

        try:
            # Gözler arası mesafe
            eye_dist = np.linalg.norm(face_landmarks[0:2] - face_landmarks[2:4])
            # Burun ucu ile ağız merkezi dikey mesafesi
            nose_y = face_landmarks[5]
            mouth_y = (face_landmarks[7] + face_landmarks[9]) / 2.0
            vertical_mouth_open = abs(mouth_y - nose_y)

            if eye_dist > 0:
                ratio = vertical_mouth_open / eye_dist
                # Oran normalden belirgin yüksekse esniyor/ağız aşırı açık
                if ratio > 0.85:
                    return True
            return False
        except Exception:
            return False

    def predict(self, face_bgr: np.ndarray, face_raw: np.ndarray = None) -> tuple[int, str, float, tuple, bool]:
        """
        Kırpılmış yüz görüntüsünden duygu durumunu tahmin eder.
        CLAHE ve esneme/yorgunluk filtrelerini uygular.
        Dönüş: (emotion_idx, emotion_tr, confidence, bgr_color, is_yawn)
        """
        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            return 0, EMOTION_MAP[0]["tr"], 0.0, EMOTION_MAP[0]["color"], False

        is_yawn = False
        if face_raw is not None and len(face_raw) >= 14:
            # face_raw içindeki landmark koordinatları: indeks 4..13
            landmarks = face_raw[4:14]
            is_yawn = self.check_yawn_or_fatigue(landmarks)

        try:
            # 1. Grayscale dönüşümü
            if len(face_bgr.shape) == 3:
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_bgr

            # 2. CLAHE Işık dengelemesi (Fabrika aydınlatma optimizasyonu)
            if self.clahe is not None:
                gray = self.clahe.apply(gray)

            # 3. 64x64 boyutlandırma
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

            # 4. Blob oluştur
            blob = resized.astype(np.float32).reshape(1, 1, 64, 64)

            # 5. Model tahmini
            self.net.setInput(blob)
            preds = self.net.forward()

            # 6. Softmax olasılık
            probs = self._softmax(preds[0])
            emotion_idx = int(np.argmax(probs))
            confidence = float(probs[emotion_idx])

            # Eğer esneme tespit edildiyse yanlış şaşkınlık/korku yerine yorgunluk işaretle
            if is_yawn:
                return emotion_idx, "Yorgun / Esniyor", confidence, (245, 158, 11), True

            info = EMOTION_MAP.get(emotion_idx, EMOTION_MAP[0])
            return emotion_idx, info["tr"], confidence, info["color"], False

        except Exception as e:
            print(f"[DUYGU UYARI] Tahmin sırasında hata: {e}")
            return 0, EMOTION_MAP[0]["tr"], 0.0, EMOTION_MAP[0]["color"], False

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
