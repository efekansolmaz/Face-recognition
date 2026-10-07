import cv2
import numpy as np
from config import EMOTION_MODEL_PATH, EMOTION_MAP

class EmotionRecognizer:
    def __init__(self, model_path: str = EMOTION_MODEL_PATH):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        # OpenCV DNN backend ve target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, face_bgr: np.ndarray) -> tuple[int, str, float, tuple]:
        """
        Kırpılmış yüz görüntüsünden (BGR) duygu durumunu tahmin eder.
        Dönüş: (emotion_idx, emotion_tr, confidence, bgr_color)
        """
        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            return 0, EMOTION_MAP[0]["tr"], 0.0, EMOTION_MAP[0]["color"]

        try:
            # 1. Grayscale'e dönüştür
            if len(face_bgr.shape) == 3:
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_bgr

            # 2. 64x64 boyutuna getir
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            
            # 3. Shape: (1, 1, 64, 64) float32
            blob = resized.astype(np.float32).reshape(1, 1, 64, 64)

            # 4. Model tahmini
            self.net.setInput(blob)
            preds = self.net.forward()

            # 5. Softmax olasılık hesabı
            probs = self._softmax(preds[0])
            emotion_idx = int(np.argmax(probs))
            confidence = float(probs[emotion_idx])

            info = EMOTION_MAP.get(emotion_idx, EMOTION_MAP[0])
            return emotion_idx, info["tr"], confidence, info["color"]

        except Exception as e:
            print(f"[DUYGU UYARI] Tahmin sirasinda hata: {e}")
            return 0, EMOTION_MAP[0]["tr"], 0.0, EMOTION_MAP[0]["color"]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
