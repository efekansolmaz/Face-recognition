import datetime
import requests
import numpy as np
import pandas as pd
from sqlalchemy import func
from database import SessionLocal, Worker, EmotionLog, AlertEvent, create_alert
from config import (
    EMOTION_MAP,
    CRITICAL_MORALE_THRESHOLD,
    SAFETY_RISK_THRESHOLD,
    WEBHOOK_URL,
    CAMERA_ZONES
)

def get_target_date_range(target_date: datetime.date = None):
    if target_date is None:
        target_date = datetime.date.today()
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = datetime.datetime.combine(target_date, datetime.time.max)
    return start_dt, end_dt

def calculate_safety_risk(morale_score: float, negative_percent: float, active_hours: float, yawn_count: int) -> tuple[float, str]:
    """
    İSG İş Kazası Risk Skoru (0 - 100):
    Yüksek stres, öfke, aşırı yorgunluk/esneme ve uzun çalışma saatlerine göre hesaplanır.
    """
    risk = 15.0  # Temel düşük risk

    # 1. Düşük moral ve negatif duygu etkisi (Max +45 puan)
    if morale_score < 45.0:
        risk += (45.0 - morale_score) * 1.0
    risk += (negative_percent * 0.3)

    # 2. Yorgunluk / Esneme etkisi (Max +20 puan)
    risk += min(20.0, yawn_count * 4.0)

    # 3. Fazla mesai / Uzun çalışma süresi etkisi (Max +20 puan)
    if active_hours > 6.0:
        risk += min(20.0, (active_hours - 6.0) * 8.0)

    safety_score = round(max(0.0, min(100.0, risk)), 1)

    if safety_score >= 70.0:
        label = "🚨 Yüksek Kaza Riski (Acil Mola Önerilir)"
    elif safety_score >= 45.0:
        label = "⚠️ Orta Risk (Dikkat Dağınıklığı / Yorgunluk)"
    else:
        label = "✅ Düşük Risk (Güvenli Çalışma Durumu)"

    return safety_score, label

def trigger_webhook_alert(worker_name: str, worker_code: str, alert_type: str, message: str):
    """Slack, Telegram veya Teams webhook servisine anlık bildirim atar."""
    if not WEBHOOK_URL:
        return
    try:
        payload = {
            "text": f"🏭 *Fabrika İSG & Ruh Hali Uyarısı*\n*Personel:* {worker_name} ({worker_code})\n*Tür:* {alert_type}\n*Mesaj:* {message}"
        }
        requests.post(WEBHOOK_URL, json=payload, timeout=3)
    except Exception as e:
        print(f"[WEBHOOK UYARI] Bildirim gönderilemedi: {e}")

def calculate_worker_morale(worker_id: int, target_date: datetime.date = None) -> dict:
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
                "safety_risk_score": 15.0,
                "safety_risk_label": "Düşük Risk",
                "dominant_emotion": "Bilinmiyor",
                "emotion_breakdown": {},
                "hourly_morale": {},
                "zone_distribution": {},
                "is_at_risk": False,
                "risk_reason": ""
            }

        first_seen = logs[0].timestamp
        last_seen = logs[-1].timestamp
        duration_mins = max(1, int((last_seen - first_seen).total_seconds() / 60))
        active_hours = duration_mins / 60.0

        emotion_counts = {}
        weighted_sum = 0.0
        yawn_count = 0
        zone_counts = {}
        hourly_buckets = {}

        for log in logs:
            emotion_counts[log.emotion] = emotion_counts.get(log.emotion, 0) + 1
            weighted_sum += (log.morale_weight * log.confidence)

            if log.is_yawn:
                yawn_count += 1

            z = log.camera_zone or "Genel"
            zone_counts[z] = zone_counts.get(z, 0) + 1

            hour_key = log.timestamp.strftime("%H:00")
            if hour_key not in hourly_buckets:
                hourly_buckets[hour_key] = []
            hourly_buckets[hour_key].append(log.morale_weight)

        avg_weight = weighted_sum / max(total_logs, 1)
        raw_score = 60.0 + (avg_weight * 40.0)
        morale_score = round(max(0.0, min(100.0, raw_score)), 1)

        # Negatif duygu oranı
        negative_emotions = ["Öfkeli", "Üzgün", "Korkmuş", "İğrenmiş", "Yorgun / Esniyor"]
        neg_count = sum(emotion_counts.get(e, 0) for e in negative_emotions)
        neg_percent = round((neg_count / total_logs) * 100, 1)

        # İSG İş Kazası Riski Hesabı
        safety_score, safety_label = calculate_safety_risk(morale_score, neg_percent, active_hours, yawn_count)

        # Durum sınıflandırması
        if morale_score >= 80.0:
            status_label = "Yüksek Motivasyon"
            status_color = "success"
            is_at_risk = False
            risk_reason = ""
        elif morale_score >= 60.0:
            status_label = "Normal / Dengeli"
            status_color = "primary"
            is_at_risk = False
            risk_reason = ""
        elif morale_score >= 45.0:
            status_label = "Hafif Düşük / Yorgun"
            status_color = "warning"
            is_at_risk = False
            risk_reason = "Günün genelinde hafif moralsizlik veya yorgunluk gözlemlendi."
        else:
            status_label = "🚨 Yüksek Stres / Kötü Gün"
            status_color = "danger"
            is_at_risk = True
            risk_reason = "Belirgin öfke, stres veya aşırı yorgunluk tespit edildi. İK/Amir görüşmesi önerilir."

        # Eğer kaza riski çok yüksekse veritabanına alarm kaydet ve webhook tetikle
        if safety_score >= SAFETY_RISK_THRESHOLD:
            msg = f"{worker.name} ({worker.worker_code}) için Kaza Riski Skoru {safety_score}/100 seviyesine ulaştı. Aşırı stres/yorgunluk gözlemleniyor."
            create_alert(worker.id, "SAFETY_RISK", msg, logs[-1].camera_zone)
            trigger_webhook_alert(worker.name, worker.worker_code, "İSG Kaza Riski Alarmı", msg)

        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

        emotion_breakdown = {
            k: {
                "count": v,
                "percent": round((v / total_logs) * 100, 1)
            } for k, v in emotion_counts.items()
        }

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
            "active_duration_hours": round(active_hours, 1),
            "morale_score": morale_score,
            "status_label": status_label,
            "status_color": status_color,
            "safety_risk_score": safety_score,
            "safety_risk_label": safety_label,
            "dominant_emotion": dominant_emotion,
            "emotion_breakdown": emotion_breakdown,
            "hourly_morale": hourly_morale,
            "zone_distribution": zone_counts,
            "is_at_risk": is_at_risk,
            "risk_reason": risk_reason
        }
    finally:
        session.close()

def calculate_weekly_burnout_trend(worker_id: int, days: int = 7) -> dict:
    """
    Bir personelin son 7 gündeki moral değişimini hesaplayarak kronik stres ve tükenmişlik tespiti yapar.
    """
    today = datetime.date.today()
    daily_scores = []
    labels = []

    for i in range(days - 1, -1, -1):
        d = today - datetime.timedelta(days=i)
        day_analysis = calculate_worker_morale(worker_id, d)
        labels.append(d.strftime("%d %b"))
        daily_scores.append(day_analysis["morale_score"] if day_analysis and day_analysis["total_logs"] > 0 else None)

    valid_scores = [s for s in daily_scores if s is not None]
    if not valid_scores:
        avg_weekly = 60.0
        trend_status = "Veri Yetersiz"
    else:
        avg_weekly = round(sum(valid_scores) / len(valid_scores), 1)
        if len(valid_scores) >= 3 and valid_scores[-1] < valid_scores[0] - 15:
            trend_status = "⚠️ Belirgin Düşüş Eğilimi (Tükenmişlik Riski)"
        elif avg_weekly < 45.0:
            trend_status = "🚨 Kronik Yüksek Stres"
        elif avg_weekly >= 75.0:
            trend_status = "✅ Yüksek & Kararlı Motivasyon"
        else:
            trend_status = "Dengeli Rutin"

    return {
        "labels": labels,
        "scores": daily_scores,
        "avg_weekly_score": avg_weekly,
        "trend_status": trend_status
    }

def get_daily_factory_summary(target_date: datetime.date = None) -> dict:
    session = SessionLocal()
    try:
        start_dt, end_dt = get_target_date_range(target_date)

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
                "safety_alert_count": 0,
                "at_risk_workers": [],
                "department_stats": {},
                "zone_stats": {},
                "top_positive_workers": [],
                "all_workers": []
            }

        avg_morale = round(sum(w["morale_score"] for w in worker_analyses) / total_active_workers, 1)

        at_risk_workers = [w for w in worker_analyses if w["is_at_risk"]]
        safety_alert_workers = [w for w in worker_analyses if w["safety_risk_score"] >= SAFETY_RISK_THRESHOLD]
        top_positive_workers = sorted(worker_analyses, key=lambda x: x["morale_score"], reverse=True)[:5]

        # Departman bazlı özet
        department_stats = {}
        for w in worker_analyses:
            dept = w["department"]
            if dept not in department_stats:
                department_stats[dept] = {"workers": 0, "scores": []}
            department_stats[dept]["workers"] += 1
            department_stats[dept]["scores"].append(w["morale_score"])

        dept_summary = {
            dept: {
                "worker_count": data["workers"],
                "avg_morale": round(sum(data["scores"]) / len(data["scores"]), 1)
            } for dept, data in department_stats.items()
        }

        # Fabrika Bölge / Kamera Lokasyonu Bazlı Stres Dağılımı
        zone_records = session.query(
            EmotionLog.camera_zone,
            func.avg(EmotionLog.morale_weight),
            func.count(EmotionLog.id)
        ).filter(
            EmotionLog.timestamp >= start_dt,
            EmotionLog.timestamp <= end_dt
        ).group_by(EmotionLog.camera_zone).all()

        zone_stats = {}
        for z_name, avg_w, cnt in zone_records:
            z_score = round(max(0.0, min(100.0, 60.0 + (float(avg_w or 0.0) * 40.0))), 1)
            zone_stats[z_name or "Genel"] = {
                "morale_score": z_score,
                "log_count": cnt,
                "status": "Stresli" if z_score < 50 else ("Dengeli" if z_score < 75 else "Pozitif")
            }

        return {
            "date": (target_date or datetime.date.today()).strftime("%Y-%m-%d"),
            "total_active_workers": total_active_workers,
            "average_factory_morale": avg_morale,
            "morale_status": "Yüksek" if avg_morale >= 75 else ("Dengeli" if avg_morale >= 60 else "Düşük / Stresli"),
            "risk_count": len(at_risk_workers),
            "safety_alert_count": len(safety_alert_workers),
            "at_risk_workers": at_risk_workers,
            "department_stats": dept_summary,
            "zone_stats": zone_stats,
            "top_positive_workers": top_positive_workers,
            "all_workers": sorted(worker_analyses, key=lambda x: x["morale_score"])
        }
    finally:
        session.close()

def export_daily_excel(target_date: datetime.date = None, output_path: str = None) -> str:
    """
    Günün tüm fabrika analizini çok sayfalı profesyonel bir Excel tablosuna aktarır.
    """
    from config import REPORTS_DIR
    import os

    if target_date is None:
        target_date = datetime.date.today()

    if output_path is None:
        date_str = target_date.strftime("%Y-%m-%d")
        output_path = os.path.join(REPORTS_DIR, f"Fabrika_Vardiya_Raporu_{date_str}.xlsx")

    summary = get_daily_factory_summary(target_date)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sayfa 1: Personel Karnesi
        worker_rows = []
        for w in summary["all_workers"]:
            worker_rows.append({
                "Sicil No": w["worker_code"],
                "Adı Soyadı": w["name"],
                "Departman": w["department"],
                "Vardiya": w["shift"],
                "İlk Görülme": w["first_seen_today"],
                "Son Görülme": w["last_seen_today"],
                "Çalışma (dk)": w["active_duration_minutes"],
                "Baskın Duygu": w["dominant_emotion"],
                "Moral Skoru (0-100)": w["morale_score"],
                "Ruh Hali Durumu": w["status_label"],
                "İSG Kaza Risk Skoru (0-100)": w["safety_risk_score"],
                "İSG Risk Değerlendirmesi": w["safety_risk_label"],
                "Amir / İK Notu": w["risk_reason"]
            })
        df_workers = pd.DataFrame(worker_rows)
        df_workers.to_excel(writer, sheet_name="Personel Karnesi", index=False)

        # Sayfa 2: İSG ve Kötü Gün Geçirenler
        risk_rows = [w for w in worker_rows if w["Moral Skoru (0-100)"] < CRITICAL_MORALE_THRESHOLD or w["İSG Kaza Risk Skoru (0-100)"] >= SAFETY_RISK_THRESHOLD]
        df_risk = pd.DataFrame(risk_rows)
        df_risk.to_excel(writer, sheet_name="İSG & Riskli Personel", index=False)

        # Sayfa 3: Departman ve Alan Dağılımları
        dept_rows = [{"Departman": k, "İşçi Sayısı": v["worker_count"], "Ortalama Moral Skoru": v["avg_morale"]} for k, v in summary["department_stats"].items()]
        df_dept = pd.DataFrame(dept_rows)
        df_dept.to_excel(writer, sheet_name="Departman Dağılımı", index=False)

        zone_rows = [{"Fabrika Alanı / Bölge": k, "Bölge Moral Skoru": v["morale_score"], "Log Adedi": v["log_count"], "Bölge Durumu": v["status"]} for k, v in summary["zone_stats"].items()]
        df_zone = pd.DataFrame(zone_rows)
        df_zone.to_excel(writer, sheet_name="Kamera Bölgeleri", index=False)

    print(f"[EXCEL] Rapor başarıyla oluşturuldu: {output_path}")
    return output_path

def generate_demo_factory_data():
    """Fabrika senaryosu için zengin çoklu gün, çoklu bölge ve İSG kaza riski verileri oluşturur."""
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
        zones = ["Pres Hattı", "Montaj Hattı", "Kaynak Atölyesi", "Yemekhane / Mola Alanı", "Giriş Turnikesi"]

        # Son 7 günün verilerini üret
        for day_offset in range(7):
            cur_date = today - datetime.timedelta(days=day_offset)
            base_time = datetime.datetime.combine(cur_date, datetime.time(8, 0))

            for sw in sample_workers:
                existing = session.query(Worker).filter(Worker.worker_code == sw["code"]).first()
                if not existing:
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

                # Eski o güne ait logları temizle
                start_dt, end_dt = get_target_date_range(cur_date)
                session.query(EmotionLog).filter(
                    EmotionLog.worker_id == worker_id,
                    EmotionLog.timestamp >= start_dt,
                    EmotionLog.timestamp <= end_dt
                ).delete()

                # Vardiya boyunca saat saat loglar ekle
                for h in range(8):
                    hour_time = base_time + datetime.timedelta(hours=h)
                    # Öğle saati yemekhane, diğer saatler kendi departmanı
                    zone = "Yemekhane / Mola Alanı" if h in [4, 5] else sw["dept"]
                    if zone not in zones:
                        zone = "Montaj Hattı"

                    for _ in range(12):
                        log_time = hour_time + datetime.timedelta(minutes=random.randint(0, 55))
                        is_yawn = False

                        if sw["mood"] == "good":
                            emo, weight = random.choices(
                                [("Mutlu", 1.0), ("Nötr", 0.0), ("Şaşkın", 0.2)],
                                weights=[0.70, 0.25, 0.05]
                            )[0]
                        elif sw["mood"] == "bad":
                            # Kötü gün / yorgun işçi
                            emo, weight = random.choices(
                                [("Öfkeli", -1.0), ("Üzgün", -0.8), ("Nötr", 0.0), ("Yorgun / Esniyor", -0.6)],
                                weights=[0.45, 0.30, 0.15, 0.10]
                            )[0]
                            if emo == "Yorgun / Esniyor" or (h >= 6 and random.random() < 0.25):
                                is_yawn = True
                        else:
                            emo, weight = random.choices(
                                [("Nötr", 0.0), ("Mutlu", 1.0), ("Üzgün", -0.8), ("Mesafeli", -0.4)],
                                weights=[0.70, 0.15, 0.10, 0.05]
                            )[0]
                            if h >= 6 and random.random() < 0.10:
                                is_yawn = True

                        log = EmotionLog(
                            worker_id=worker_id,
                            camera_zone=zone,
                            emotion=emo,
                            confidence=round(random.uniform(0.75, 0.98), 2),
                            morale_weight=weight,
                            is_yawn=is_yawn,
                            timestamp=log_time
                        )
                        session.add(log)
                session.commit()
        return True
    finally:
        session.close()
