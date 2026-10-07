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

        # 3. Bilinen iscileri hafizaya yukle
        self.known_workers: list[dict] = []
        self.reload_known_workers()

    def reload_known_workers(self):
        """Veritabanindaki tum iscileri hafizaya alir."""
        self.known_workers = database.load_all_workers()
        print(f"[YUZ MOTORU] {len(self.known_workers)} kayitli isci hafizaya yuklendi.")

    def detect_faces(self, frame: np.ndarray) -> list[np.ndarray]:
        """Karedeki tum yuzleri tespit eder ve liste doner."""
        h, w = frame.shape[:2]
        if self.detector_input_size != (w, h):
            self.detector.setInputSize((w, h))
            self.detector_input_size = (w, h)

        ret, faces = self.detector.detect(frame)
        if ret == 0 or faces is None:
            return []
        return [f for f in faces]

    def extract_feature(self, frame: np.ndarray, face_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Yuzu landmark noktalarina gore hizalar ve 128D embedding cikarir."""
        aligned = self.recognizer.alignCrop(frame, face_raw)
        feature = self.recognizer.feature(aligned)
        return aligned, feature

    def recognize_or_enroll(self, frame: np.ndarray, face_raw: np.ndarray) -> tuple[int, str, str, str, float, bool]:
        """
        Yuzu tanir. Eger biliniyorsa bilgilerini doner.
        Bilinmiyorsa veritabanina yeni Isci (otomatik sicil no ile) kaydeder.
        Donus: (worker_id, name, worker_code, department, similarity, is_new)
        """
        aligned, feature = self.extract_feature(frame, face_raw)

        best_match = None
        best_sim = -1.0

        for worker in self.known_workers:
            # Sifir embedding kontrolu (CSV'den on kayit yapilmissa)
            if np.all(worker["embedding"] == 0):
                continue

            sim = self.recognizer.match(feature, worker["embedding"], cv2.FaceRecognizerSF_FR_COSINE)
            if sim > best_sim:
                best_sim = sim
                best_match = worker

        # Eger en yuksek benzerlik esik degerinden buyukse eslesme basarilidir
        if best_match is not None and best_sim >= COSINE_SIMILARITY_THRESHOLD:
            database.update_worker_seen(best_match["id"], feature)
            return (
                best_match["id"],
                best_match["name"],
                best_match["worker_code"],
                best_match["department"],
                float(best_sim),
                False
            )

        # Eslesme bulunamadi -> Otomatik Yeni Isci Kaydi
        temp_id = len(self.known_workers) + 1
        photo_filename = f"worker_{temp_id}.jpg"
        photo_path = os.path.join(FACES_DIR, photo_filename)
        cv2.imwrite(photo_path, aligned)

        new_worker = database.create_worker(
            embedding=feature,
            photo_path=photo_path
        )

        # Hafizadaki listeye ekle
        self.known_workers.append({
            "id": new_worker["id"],
            "worker_code": new_worker["worker_code"],
            "name": new_worker["name"],
            "department": new_worker["department"],
            "shift": new_worker["shift"],
            "embedding": feature,
            "photo_path": photo_path,
            "is_identified": False
        })

        print(f"[YENI ISCI KAYDEDILDI] {new_worker['worker_code']} - {new_worker['name']}")
        return (
            new_worker["id"],
            new_worker["name"],
            new_worker["worker_code"],
            new_worker["department"],
            float(best_sim if best_sim > 0 else 0.0),
            True
        )
