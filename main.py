import time
import cv2
import numpy as np
import database
from config import CAMERA_INDEX, EMOTION_LOG_INTERVAL_SEC
from face_engine import FaceEngine
from emotion_engine import EmotionRecognizer
from tracker import FaceTracker

def draw_corner_rect(img, bbox, color, thickness=2, d=15):
    """Modern kose cizgili bounding box cizer."""
    x, y, w, h = bbox
    # Ana kutu (ince)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

    # Koseler (kalin ve belirgin)
    # Sol Ust
    cv2.line(img, (x, y), (x + d, y), color, thickness)
    cv2.line(img, (x, y), (x, y + d), color, thickness)
    # Sag Ust
    cv2.line(img, (x + w, y), (x + w - d, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + d), color, thickness)
    # Sol Alt
    cv2.line(img, (x, y + h), (x + d, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - d), color, thickness)
    # Sag Alt
    cv2.line(img, (x + w, y + h), (x + w - d, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - d), color, thickness)

def draw_hud(frame, fps, enrolled_count, detected_count):
    """Ekranin sol ustune yari saydam modern bilgi paneli ekler."""
    h_panel, w_panel = 100, 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + w_panel, 10 + h_panel), (20, 20, 20), -1)
    # %75 seffaflik
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + w_panel, 10 + h_panel), (60, 60, 60), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(frame, f"Kayitli Kisi: {enrolled_count}", (20, 58), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Ekrandaki: {detected_count}", (20, 81), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1)
    cv2.putText(frame, "[Q]: Cikis | [S]: Ozet Rapor", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

def main():
    print("=" * 60)
    print("  GERCEK ZAMANLI YUZ TANIMA VE DUYGU ANALIZI SISTEMI")
    print("=" * 60)

    # 1. Bilesenleri baslat
    face_engine = FaceEngine()
    emotion_recognizer = EmotionRecognizer()
    tracker = FaceTracker()

    # 2. Kamerayi ac
    print(f"[KAMERA] Kamera index {CAMERA_INDEX} baslatiliyor...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[HATA] Kamera açılamadı! (Index: {CAMERA_INDEX})")
        print("Lutfen kameranizin takili oldugundan ve baska program tarafindan kullanilmadigindan emin olun.")
        return

    # Kamera cozunurlugunu ayarla (opsiyonel 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    fps = 0.0

    print("[SISTEM] Hazir! Cikmak icin video penceresindeyken 'q' tusuna basin.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[UYARI] Kameradan goruntu alinamadi.")
            break

        # Ayna goruntusu icin yatay cevir
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        curr_time = time.time()

        # FPS hesaplama
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        # 3. Yuz tespiti (YuNet)
        detected_faces = face_engine.detect_faces(frame)

        # 4. Kareler arasi takipciyi guncelle
        tracked_faces = tracker.update(detected_faces)

        # 5. Her yuz icin tanima ve duygu analizi yap
        for track in tracked_faces:
            x, y, fw, fh = track.bbox

            # Goruntu sinirlari disina tasmasini engelle
            x = max(0, x)
            y = max(0, y)
            fw = min(w - x, fw)
            fh = min(h - y, fh)

            if fw <= 10 or fh <= 10:
                continue

            # A. Eger henuz tanimlanmamissa veya periyodik dogrulama gerekiyorsa
            if track.user_id is None:
                user_id, name, sim, is_new = face_engine.recognize_or_enroll(frame, track.face_raw)
                track.user_id = user_id
                track.user_name = name

            # B. Duygu Analizi (FERPlus)
            face_crop = frame[y:y+fh, x:x+fw]
            _, emotion_tr, conf, emotion_color = emotion_recognizer.predict(face_crop)
            track.emotion = emotion_tr
            track.emotion_conf = conf
            track.emotion_color = emotion_color

            # C. Veritabanina periyodik duygu kaydi (ornek: saniyede 1 kez)
            if curr_time - track.last_log_time >= EMOTION_LOG_INTERVAL_SEC:
                database.log_emotion(track.user_id, emotion_tr, conf)
                track.last_log_time = curr_time

            # D. Gorsellestirme
            draw_corner_rect(frame, (x, y, fw, fh), emotion_color, thickness=2, d=15)

            # Etiket arka plan seridi
            header_text = f"ID:{track.user_id} {track.user_name}"
            sub_text = f"{track.emotion} %{int(track.emotion_conf * 100)}"
            
            # Etiket kutusu
            label_y = max(25, y - 10)
            cv2.rectangle(frame, (x, label_y - 22), (x + max(len(header_text), len(sub_text)) * 11 + 10, label_y + 16), (25, 25, 25), -1)
            cv2.rectangle(frame, (x, label_y - 22), (x + max(len(header_text), len(sub_text)) * 11 + 10, label_y + 16), emotion_color, 1)

            cv2.putText(frame, header_text, (x + 5, label_y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1)
            cv2.putText(frame, sub_text, (x + 5, label_y + 12), cv2.FONT_HERSHEY_DUPLEX, 0.45, emotion_color, 1)

        # 6. HUD Bilgi Paneli
        draw_hud(frame, fps, len(face_engine.known_users), len(tracked_faces))

        # 7. Ekrana bas
        cv2.imshow("Gercek Zamanli Yuz Tanima & Duygu Analizi", frame)

        # Klavye kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' veya ESC
            print("\n[BILGI] Program kullanıcı tarafından sonlandırıldı.")
            break
        elif key == ord('s'):  # Istatistik raporu
            stats = database.get_stats()
            print("\n" + "="*40)
            print("         VERITABANI OZET RAPORU")
            print("="*40)
            print(f"Toplam Kayitli Kisi : {stats['user_count']}")
            print(f"Toplam Duygu Kaydi  : {stats['log_count']}")
            print("Duygu Dagilimi:")
            for emo, cnt in stats["top_emotions"].items():
                print(f"  - {emo}: {cnt} kez")
            print("="*40 + "\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
