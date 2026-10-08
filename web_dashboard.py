import os
import datetime
import time
import cv2
import threading
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
import analytics
import database
import pdf_report
from config import (
    FACES_DIR,
    REPORTS_DIR,
    DASHBOARD_PORT,
    DASHBOARD_HOST,
    CAMERA_INDEX,
    DEFAULT_CAMERA_ZONE,
    CAMERA_ZONES
)
from face_engine import FaceEngine
from emotion_engine import EmotionRecognizer
from tracker import FaceTracker

app = Flask(__name__)

# --- CANLI KAMERA YAYIN MOTORU (MJPEG Stream) ---
class VideoCamera:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.frame = None
        self.face_engine = None
        self.emotion_engine = None
        self.tracker = None
        self.active_zone = DEFAULT_CAMERA_ZONE

    def start(self, zone: str = DEFAULT_CAMERA_ZONE):
        with self.lock:
            self.active_zone = zone
            if not self.is_running:
                self.cap = cv2.VideoCapture(CAMERA_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.face_engine = FaceEngine()
                self.emotion_engine = EmotionRecognizer()
                self.tracker = FaceTracker()
                self.is_running = True
                threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        last_log_times = {}
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            curr_time = time.time()

            detected_faces = self.face_engine.detect_faces(frame)
            tracked_faces = self.tracker.update(detected_faces)

            for track in tracked_faces:
                x, y, fw, fh = track.bbox
                x = max(0, x)
                y = max(0, y)
                fw = min(w - x, fw)
                fh = min(h - y, fh)
                if fw <= 10 or fh <= 10:
                    continue

                if track.worker_id is None:
                    wid, wname, wcode, wdept, sim, is_new = self.face_engine.recognize_or_enroll(frame, track.face_raw)
                    track.worker_id = wid
                    track.worker_name = wname
                    track.worker_code = wcode
                    track.worker_department = wdept

                # Duygu Analizi (CLAHE ve esneme/yorgunluk filtreli)
                face_crop = frame[y:y+fh, x:x+fw]
                _, emotion_tr, conf, emotion_color, is_yawn = self.emotion_engine.predict(face_crop, track.face_raw)
                track.emotion = emotion_tr
                track.emotion_conf = conf
                track.emotion_color = emotion_color

                # Veritabanı Duygu & Bölge Kaydı
                last_time = last_log_times.get(track.worker_id, 0.0)
                if curr_time - last_time >= 1.0:
                    database.log_emotion(
                        worker_id=track.worker_id,
                        emotion=emotion_tr,
                        confidence=conf,
                        camera_zone=self.active_zone,
                        is_yawn=is_yawn
                    )
                    last_log_times[track.worker_id] = curr_time

                # Bounding Box ve HUD
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), emotion_color, 2)
                badge_text = f"{track.worker_code} {track.worker_name} [{self.active_zone}]"
                sub_text = f"{track.emotion} %{int(conf * 100)}"
                
                label_y = max(20, y - 8)
                cv2.rectangle(frame, (x, label_y - 20), (x + len(badge_text) * 9 + 10, label_y + 14), (20, 20, 20), -1)
                cv2.putText(frame, badge_text, (x + 5, label_y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1)
                cv2.putText(frame, sub_text, (x + 5, label_y + 10), cv2.FONT_HERSHEY_DUPLEX, 0.42, emotion_color, 1)

            with self.lock:
                self.frame = frame

            time.sleep(0.03)

    def get_frame_bytes(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes() if ret else None

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()

camera_instance = VideoCamera()

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("dashboard.html", camera_zones=CAMERA_ZONES)

@app.route("/api/summary")
def api_summary():
    date_str = request.args.get("date")
    target_date = datetime.date.today()
    if date_str:
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            pass
    summary = analytics.get_daily_factory_summary(target_date)
    return jsonify(summary)

@app.route("/api/worker/<int:worker_id>/weekly")
def api_worker_weekly(worker_id):
    trend = analytics.calculate_weekly_burnout_trend(worker_id)
    return jsonify(trend)

@app.route("/api/alerts")
def api_alerts():
    alerts = database.get_active_alerts(limit=15)
    return jsonify(alerts)

@app.route("/api/worker/<int:worker_id>/edit", methods=["POST"])
def api_worker_edit(worker_id):
    data = request.json or {}
    name = data.get("name", "")
    code = data.get("worker_code")
    dept = data.get("department")
    shift = data.get("shift")
    notes = data.get("notes")

    success = database.update_worker_profile(worker_id, name, code, dept, shift, notes)
    return jsonify({"success": success})

@app.route("/api/workers/import-csv", methods=["POST"])
def api_import_csv():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Dosya bulunamadı"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Dosya seçilmedi"}), 400

    temp_path = os.path.join(REPORTS_DIR, "temp_import.csv")
    file.save(temp_path)
    added, skipped = database.import_workers_from_csv(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return jsonify({"success": True, "added": added, "skipped": skipped})

@app.route("/api/report/pdf")
def api_report_pdf():
    date_str = request.args.get("date")
    target_date = datetime.date.today()
    if date_str:
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            pass

    pdf_path = pdf_report.generate_daily_pdf(target_date)
    return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))

@app.route("/api/report/excel")
def api_report_excel():
    date_str = request.args.get("date")
    target_date = datetime.date.today()
    if date_str:
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            pass

    excel_path = analytics.export_daily_excel(target_date)
    return send_file(excel_path, as_attachment=True, download_name=os.path.basename(excel_path))

@app.route("/api/mock-demo-data", methods=["POST"])
def api_mock_demo():
    analytics.generate_demo_factory_data()
    return jsonify({"success": True})

@app.route("/api/kvkk/cleanup", methods=["POST"])
def api_kvkk_cleanup():
    deleted = database.cleanup_old_photos()
    return jsonify({"success": True, "deleted_photos": deleted})

@app.route("/faces/<filename>")
def serve_face(filename):
    return send_from_directory(FACES_DIR, filename)

def gen_frames(zone):
    camera_instance.start(zone)
    while True:
        frame_bytes = camera_instance.get_frame_bytes()
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    zone = request.args.get("zone", DEFAULT_CAMERA_ZONE)
    return Response(gen_frames(zone), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print(f"\n" + "=" * 65)
    print(f"  KURUMSAL FABRİKA YÖNETİM DASHBOARD'U BAŞLATILIYOR")
    print(f"  Tarayıcınızdan açın: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"=" * 65 + "\n")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, threaded=True)
