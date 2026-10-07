import database
from database import SessionLocal, User, EmotionLog
import pandas as pd

def print_detailed_report():
    session = SessionLocal()
    try:
        users = session.query(User).all()
        print("\n" + "=" * 60)
        print("         KAYITLI KULLANICILAR VE GORULME GECMISI")
        print("=" * 60)
        if not users:
            print("Henuz hicbir kisi kaydedilmemis.")
        for u in users:
            log_count = session.query(EmotionLog).filter(EmotionLog.user_id == u.id).count()
            print(f"ID: {u.id:2d} | İsim: {u.name:<15} | İlk Görülme: {u.first_seen.strftime('%Y-%m-%d %H:%M:%S')} | Son Görülme: {u.last_seen.strftime('%Y-%m-%d %H:%M:%S')} | Duygu Kaydı: {log_count}")
        
        print("\n" + "=" * 60)
        print("                   DUYGU OZETLERI")
        print("=" * 60)
        stats = database.get_stats()
        print(f"Toplam Duygu Logu: {stats['log_count']}")
        for emo, count in stats["top_emotions"].items():
            print(f"  * {emo:<10}: {count} adet")
        print("=" * 60 + "\n")
    finally:
        session.close()

def export_logs_csv(output_file="data/emotion_logs.csv"):
    session = SessionLocal()
    try:
        logs = session.query(EmotionLog).all()
        if not logs:
            print("[BILGI] Kaydedilecek duygu logu bulunamadi.")
            return
        data = [{
            "id": l.id,
            "user_id": l.user_id,
            "emotion": l.emotion,
            "confidence": l.confidence,
            "timestamp": l.timestamp
        } for l in logs]
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"[OK] Loglar '{output_file}' dosyasina CSV olarak aktarildi.")
    finally:
        session.close()

if __name__ == "__main__":
    print_detailed_report()
