import os
import cv2
import numpy as np
import database
from config import (
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    COSINE_SIMILARITY_THRESHOLD,
    FACES_DIR
)

class FaceEngine:
    def __init__(self):
        # 1. Yuz Tespit Modeli (YuNet)
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            top_k=5000
        )
        self.detector_input_size = (320, 320)

        # 2. Yuz Tanima Modeli (SFace)
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_MODEL_PATH,
            config=""
        )

        # 3. Bilinen kullanicilari veritabanindan hafizaya yukle
        self.known_users: list[dict] = []
        self.reload_known_users()

    def reload_known_users(self):
        """Veritabanindaki tum kayitli kisileri yukler."""
        self.known_users = database.load_all_users()
        print(f"[YUZ MOTORU] {len(self.known_users)} kayitli kisi hafizaya yuklendi.")

    def detect_faces(self, frame: np.ndarray) -> list[np.ndarray]:
        """Karedeki yuzleri tespit eder ve 15 elemanli yuz vektoru listesi doner."""
        h, w = frame.shape[:2]
        if self.detector_input_size != (w, h):
            self.detector.setInputSize((w, h))
            self.detector_input_size = (w, h)

        ret, faces = self.detector.detect(frame)
        if ret == 0 or faces is None:
            return []
        return [f for f in faces]

    def extract_feature(self, frame: np.ndarray, face_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Yuzu landmark noktalarina gore hizalar ve 128 boyutlu embedding cikarir.
        Donus: (aligned_crop, feature_128d)
        """
        aligned = self.recognizer.alignCrop(frame, face_raw)
        feature = self.recognizer.feature(aligned)
        return aligned, feature

    def recognize_or_enroll(self, frame: np.ndarray, face_raw: np.ndarray) -> tuple[int, str, float, bool]:
        """
        Yuzu tanir. Eger biliniyorsa ID ve ismini doner.
        Bilinmiyorsa veritabanina otomatik yeni ID ile kaydeder.
        Donus: (user_id, name, similarity_score, is_new_user)
        """
        aligned, feature = self.extract_feature(frame, face_raw)

        best_match = None
        best_sim = -1.0

        for user in self.known_users:
            sim = self.recognizer.match(feature, user["embedding"], cv2.FaceRecognizerSF_FR_COSINE)
            if sim > best_sim:
                best_sim = sim
                best_match = user

        # Eger en yuksek benzerlik esik degerinden buyukse eslesme basarilidir
        if best_match is not None and best_sim >= COSINE_SIMILARITY_THRESHOLD:
            # Arka planda gorulme zamanini guncelle
            database.update_user_seen(best_match["id"], feature)
            return best_match["id"], best_match["name"], float(best_sim), False

        # Eslesme bulunamadi -> Otomatik Yeni Kisi Kaydi
        temp_id = len(self.known_users) + 1
        photo_filename = f"user_{temp_id}.jpg"
        photo_path = os.path.join(FACES_DIR, photo_filename)
        cv2.imwrite(photo_path, aligned)

        new_user = database.create_user(
            embedding=feature,
            photo_path=photo_path
        )
        
        # Hafizadaki listeye ekle
        self.known_users.append({
            "id": new_user["id"],
            "name": new_user["name"],
            "embedding": feature,
            "photo_path": photo_path
        })

        print(f"[YENI KISI ALGILANDI] ID: {new_user['id']} - Isim: {new_user['name']} kaydedildi.")
        return new_user["id"], new_user["name"], float(best_sim if best_sim > 0 else 0.0), True
