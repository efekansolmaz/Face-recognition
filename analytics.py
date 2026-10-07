import datetime
from sqlalchemy import func
from database import SessionLocal, Worker, EmotionLog
from config import EMOTION_MAP

def get_target_date_range(target_date: datetime.date = None):
    """Verilen günün başlangıç ve bitiş saatlerini döner."""
    if target_date is None:
        target_date = datetime.date.today()
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    return start_dt, end_dt

def calculate_worker_morale(worker_id: int, target_date: datetime.date = None) -> dict:
    """
    Belirli bir işçinin seçilen gündeki duygu ve moral analizini yapar.
    0 - 100 arası Morale Score, baskın duygular ve saatlik trend döner.
    """
    session = SessionLocal()
    try:
        worker = session.query(Worker).filter(Worker.id == worker_id).first()
        if not worker:
            return None

        start_dt, end_dt = get_target_date_range(target_date)

        logs = session.query(EmotionLog).filter(
            EmotionLog.worker_id == worker_id,
            EmotionLog.timestamp >= start_dt,
            EmotionLog.timestamp <= end_dt
        ).order_by(EmotionLog.timestamp.asc()).all()

        total_logs = len(logs)
        if total_logs == 0:
            return {
                "worker_id": worker.id,
                "worker_code": worker.worker_code,
                "name": worker.name,
                "department": worker.department,
                "shift": worker.shift,
                "photo_path": worker.photo_path,
                "is_identified": worker.is_identified,
                "total_logs": 0,
                "first_seen_today": None,
                "last_seen_today": None,
                "active_duration_minutes": 0,
                "morale_score": 60.0,
                "status_label": "Kayıt Yok",
                "status_color": "secondary",
                "dominant_emotion": "Bilinmiyor",
                "emotion_breakdown": {},
                "hourly_morale": {},
                "is_at_risk": False,
                "risk_reason": ""
            }

        first_seen = logs[0].timestamp
        last_seen = logs[-1].timestamp
        duration_mins = max(1, int((last_seen - first_seen).total_seconds() / 60))

        # Duygu dağılımı ve ağırlıklı toplam
        emotion_counts = {}
        weighted_sum = 0.0

        hourly_buckets = {}  # saat -> [morale_weights]

        for log in logs:
            emotion_counts[log.emotion] = emotion_counts.get(log.emotion, 0) + 1
            weighted_sum += (log.morale_weight * log.confidence)

            hour_key = log.timestamp.strftime("%H:00")
            if hour_key not in hourly_buckets:
                hourly_buckets[hour_key] = []
            hourly_buckets[hour_key].append(log.morale_weight)

        # 0.0 - 100.0 Morale Score Formülü
        avg_weight = weighted_sum / max(total_logs, 1)
        raw_score = 60.0 + (avg_weight * 40.0)
        morale_score = round(max(0.0, min(100.0, raw_score)), 1)

        # Durum sınıflandırması
        if morale_score >= 80.0:
            status_label = "Yüksek Motivasyon"
            status_color = "success"  # Yeşil
            is_at_risk = False
            risk_reason = ""
        elif morale_score >= 60.0:
            status_label = "Normal / Dengeli"
            status_color = "primary"  # Mavi
            is_at_risk = False
            risk_reason = ""
        elif morale_score >= 45.0:
            status_label = "Hafif Düşük / Yorgun"
            status_color = "warning"  # Sarı/Turuncu
            is_at_risk = False
            risk_reason = "Gün genelinde hafif moralsizlik veya yorgunluk gözlemlendi."
        else:
            status_label = "🚨 Yüksek Stres / Kötü Gün"
            status_color = "danger"  # Kırmızı
            is_at_risk = True
            risk_reason = "Belirgin öfke, aşırı stres veya üzüntü tespit edildi. İK/Amir görüşmesi önerilir."

        # Baskın duygu
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

        # Yüzdelik duygu dağılımı
        emotion_breakdown = {
            k: {
                "count": v,
                "percent": round((v / total_logs) * 100, 1)
            } for k, v in emotion_counts.items()
        }

        # Saatlik moral trendi
        hourly_morale = {}
        for h, weights in hourly_buckets.items():
            h_avg = sum(weights) / len(weights)
            h_score = round(max(0.0, min(100.0, 60.0 + (h_avg * 40.0))), 1)
            hourly_morale[h] = h_score

        return {
            "worker_id": worker.id,
            "worker_code": worker.worker_code,
            "name": worker.name,
            "department": worker.department,
            "shift": worker.shift,
            "photo_path": worker.photo_path,
            "is_identified": worker.is_identified,
            "total_logs": total_logs,
            "first_seen_today": first_seen.strftime("%H:%M"),
            "last_seen_today": last_seen.strftime("%H:%M"),
            "active_duration_minutes": duration_mins,
            "morale_score": morale_score,
            "status_label": status_label,
            "status_color": status_color,
            "dominant_emotion": dominant_emotion,
            "emotion_breakdown": emotion_breakdown,
            "hourly_morale": hourly_morale,
            "is_at_risk": is_at_risk,
            "risk_reason": risk_reason
        }
    finally:
        session.close()

def get_daily_factory_summary(target_date: datetime.date = None) -> dict:
    """
    Tüm fabrikanın seçilen gündeki genel özetini ve departman kıyaslamasını üretir.
    """
    session = SessionLocal()
    try:
        start_dt, end_dt = get_target_date_range(target_date)

        # Bugün logu olan işçi ID'leri
        active_worker_ids = session.query(EmotionLog.worker_id).filter(
            EmotionLog.timestamp >= start_dt,
            EmotionLog.timestamp <= end_dt
        ).distinct().all()

        worker_ids = [w[0] for w in active_worker_ids]

        worker_analyses = []
        for w_id in worker_ids:
            analysis = calculate_worker_morale(w_id, target_date)
            if analysis:
                worker_analyses.append(analysis)

        total_active_workers = len(worker_analyses)
        if total_active_workers == 0:
            return {
                "date": (target_date or datetime.date.today()).strftime("%Y-%m-%d"),
                "total_active_workers": 0,
                "average_factory_morale": 60.0,
                "morale_status": "Veri Yok",
                "risk_count": 0,
                "at_risk_workers": [],
                "department_stats": {},
                "top_positive_workers": [],
                "all_workers": []
            }

        # Ortalama fabrika morali
        avg_morale = round(sum(w["morale_score"] for w in worker_analyses) / total_active_workers, 1)

        # Risk altındaki işçiler (Morali < 45)
        at_risk_workers = [w for w in worker_analyses if w["is_at_risk"]]

        # En pozitif işçiler
        top_positive_workers = sorted(worker_analyses, key=lambda x: x["morale_score"], reverse=True)[:5]

        # Departman bazlı özet
        department_stats = {}
        for w in worker_analyses:
            dept = w["department"]
            if dept not in department_stats:
                department_stats[dept] = {"workers": 0, "scores": [], "emotions": {}}
            department_stats[dept]["workers"] += 1
            department_stats[dept]["scores"].append(w["morale_score"])

        dept_summary = {}
        for dept, data in department_stats.items():
            dept_summary[dept] = {
                "worker_count": data["workers"],
                "avg_morale": round(sum(data["scores"]) / len(data["scores"]), 1)
            }

        return {
            "date": (target_date or datetime.date.today()).strftime("%Y-%m-%d"),
            "total_active_workers": total_active_workers,
            "average_factory_morale": avg_morale,
            "morale_status": "Yüksek" if avg_morale >= 75 else ("Dengeli" if avg_morale >= 60 else "Düşük / Stresli"),
            "risk_count": len(at_risk_workers),
            "at_risk_workers": at_risk_workers,
            "department_stats": dept_summary,
            "top_positive_workers": top_positive_workers,
            "all_workers": sorted(worker_analyses, key=lambda x: x["morale_score"])
        }
    finally:
        session.close()

def generate_demo_factory_data():
    """Fabrika senaryosu icin gercekci ornek vardiya verileri olusturur."""
    import random
    session = SessionLocal()
    try:
        sample_workers = [
            {"code": "SICIL-101", "name": "Ahmet Yılmaz", "dept": "Pres Hattı", "shift": "Gündüz (08:00 - 16:00)", "mood": "normal"},
            {"code": "SICIL-102", "name": "Mehmet Demir", "dept": "Kaynak Atölyesi", "shift": "Gündüz (08:00 - 16:00)", "mood": "bad"},
            {"code": "SICIL-103", "name": "Ayşe Kaya", "dept": "Kalite Kontrol", "shift": "Gündüz (08:00 - 16:00)", "mood": "good"},
            {"code": "SICIL-104", "name": "Canan Çelik", "dept": "Üretim / Montaj", "shift": "Gündüz (08:00 - 16:00)", "mood": "good"},
            {"code": "SICIL-105", "name": "Ali Öztürk", "dept": "Pres Hattı", "shift": "Gündüz (08:00 - 16:00)", "mood": "bad"},
            {"code": "SICIL-106", "name": "Fatma Şahin", "dept": "Boyahane", "shift": "Gündüz (08:00 - 16:00)", "mood": "normal"},
            {"code": "SICIL-107", "name": "Burak Yurt", "dept": "Depo & Lojistik", "shift": "Gündüz (08:00 - 16:00)", "mood": "normal"},
            {"code": "SICIL-108", "name": "İşçi #108 (Tanımsız)", "dept": "Genel", "shift": "Gündüz (08:00 - 16:00)", "mood": "normal", "unid": True},
        ]

        today = datetime.date.today()
        base_time = datetime.datetime.combine(today, datetime.time(8, 0))

        for sw in sample_workers:
            existing = session.query(Worker).filter(Worker.worker_code == sw["code"]).first()
            if not existing:
                import numpy as np
                w = Worker(
                    worker_code=sw["code"],
                    name=sw["name"],
                    department=sw["dept"],
                    shift=sw["shift"],
                    first_seen=base_time,
                    last_seen=base_time + datetime.timedelta(hours=8),
                    total_detected_frames=500,
                    is_identified=not sw.get("unid", False)
                )
                w.set_embedding(np.random.rand(1, 128).astype(np.float32))
                session.add(w)
                session.commit()
                worker_id = w.id
            else:
                worker_id = existing.id

            # Eski bugünkü logları temizle ve yeniden oluştur
            start_dt, end_dt = get_target_date_range(today)
            session.query(EmotionLog).filter(
                EmotionLog.worker_id == worker_id,
                EmotionLog.timestamp >= start_dt,
                EmotionLog.timestamp <= end_dt
            ).delete()

            # Vardiya boyunca saat saat loglar ekle
            for h in range(8):
                hour_time = base_time + datetime.timedelta(hours=h)
                for _ in range(15):  # Saatte 15 log
                    log_time = hour_time + datetime.timedelta(minutes=random.randint(0, 55))
                    
                    if sw["mood"] == "good":
                        emo, weight = random.choices(
                            [("Mutlu", 1.0), ("Nötr", 0.0), ("Şaşkın", 0.2)],
                            weights=[0.65, 0.30, 0.05]
                        )[0]
                    elif sw["mood"] == "bad":
                        emo, weight = random.choices(
                            [("Öfkeli", -1.0), ("Üzgün", -0.8), ("Nötr", 0.0), ("Korkmuş", -0.7)],
                            weights=[0.45, 0.30, 0.20, 0.05]
                        )[0]
                    else:
                        emo, weight = random.choices(
                            [("Nötr", 0.0), ("Mutlu", 1.0), ("Üzgün", -0.8), ("Mesafeli", -0.4)],
                            weights=[0.70, 0.15, 0.10, 0.05]
                        )[0]

                    log = EmotionLog(
                        worker_id=worker_id,
                        emotion=emo,
                        confidence=round(random.uniform(0.75, 0.98), 2),
                        morale_weight=weight,
                        timestamp=log_time
                    )
                    session.add(log)
            session.commit()
        return True
    finally:
        session.close()

