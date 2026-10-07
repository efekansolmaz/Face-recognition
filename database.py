import datetime
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, ForeignKey, Boolean, Text, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import DATABASE_URL, EMOTION_MAP

Base = declarative_base()

class Worker(Base):
    __tablename__ = "workers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_code = Column(String(50), unique=True, nullable=False)   # Sicil No / Kart No
    name = Column(String(100), nullable=False)                     # Ad Soyad
    department = Column(String(100), default="Genel")               # Departman / Hat
    shift = Column(String(50), default="Gündüz (08:00 - 16:00)")    # Vardiya
    first_seen = Column(DateTime, default=datetime.datetime.now)
    last_seen = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
    total_detected_frames = Column(Integer, default=1)
    photo_path = Column(String(255), nullable=True)
    embedding = Column(LargeBinary, nullable=False)                # 128-D float numpy array
    is_identified = Column(Boolean, default=False)                 # Isimlendirildi mi?
    notes = Column(Text, nullable=True)

    emotion_logs = relationship("EmotionLog", back_populates="worker", cascade="all, delete-orphan")
    shift_sessions = relationship("ShiftSession", back_populates="worker", cascade="all, delete-orphan")

    def get_embedding(self) -> np.ndarray:
        return pickle.loads(self.embedding)

    def set_embedding(self, emb: np.ndarray):
        self.embedding = pickle.dumps(emb)

    def __repr__(self):
        return f"<Worker(id={self.id}, code='{self.worker_code}', name='{self.name}', dept='{self.department}')>"


class ShiftSession(Base):
    __tablename__ = "shift_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(Integer, ForeignKey("workers.id"), nullable=False)
    start_time = Column(DateTime, default=datetime.datetime.now)
    end_time = Column(DateTime, default=datetime.datetime.now)
    duration_seconds = Column(Float, default=0.0)

    worker = relationship("Worker", back_populates="shift_sessions")


class EmotionLog(Base):
    __tablename__ = "emotion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(Integer, ForeignKey("workers.id"), nullable=False)
    emotion = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    morale_weight = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.datetime.now)

    worker = relationship("Worker", back_populates="emotion_logs")

    def __repr__(self):
        return f"<EmotionLog(worker_id={self.worker_id}, emotion='{self.emotion}', conf={self.confidence:.2f})>"


# --- ENGINE & SESSION ---
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Fabrika veritabani hazirlandi: {DATABASE_URL}")

init_db()

# --- CRUD VE YARDIMCI FONKSIYONLAR ---

def load_all_workers():
    session = SessionLocal()
    try:
        workers = session.query(Worker).all()
        result = []
        for w in workers:
            try:
                emb = w.get_embedding()
                result.append({
                    "id": w.id,
                    "worker_code": w.worker_code,
                    "name": w.name,
                    "department": w.department,
                    "shift": w.shift,
                    "embedding": emb,
                    "photo_path": w.photo_path,
                    "is_identified": w.is_identified
                })
            except Exception as e:
                print(f"[DB HATA] Worker {w.id} embedding hatasi: {e}")
        return result
    finally:
        session.close()

def create_worker(embedding: np.ndarray, name: str = None, worker_code: str = None, department: str = "Genel", shift: str = "Gündüz (08:00 - 16:00)", photo_path: str = None, is_identified: bool = False) -> dict:
    session = SessionLocal()
    try:
        count = session.query(Worker).count() + 1
        assigned_code = worker_code or f"ISC-{count:03d}"
        assigned_name = name or f"İşçi #{count}"

        new_worker = Worker(
            worker_code=assigned_code,
            name=assigned_name,
            department=department,
            shift=shift,
            first_seen=datetime.datetime.now(),
            last_seen=datetime.datetime.now(),
            total_detected_frames=1,
            photo_path=photo_path,
            is_identified=is_identified
        )
        new_worker.set_embedding(embedding)

        session.add(new_worker)
        session.commit()
        session.refresh(new_worker)

        return {
            "id": new_worker.id,
            "worker_code": new_worker.worker_code,
            "name": new_worker.name,
            "department": new_worker.department,
            "shift": new_worker.shift,
            "embedding": embedding,
            "photo_path": photo_path,
            "is_identified": new_worker.is_identified
        }
    finally:
        session.close()

def update_worker_seen(worker_id: int, new_embedding: np.ndarray = None):
    session = SessionLocal()
    try:
        worker = session.query(Worker).filter(Worker.id == worker_id).first()
        if worker:
            now = datetime.datetime.now()
            worker.last_seen = now
            worker.total_detected_frames += 1

            if new_embedding is not None:
                current_emb = worker.get_embedding()
                updated_emb = (0.95 * current_emb) + (0.05 * new_embedding)
                worker.set_embedding(updated_emb)

            session.commit()
    finally:
        session.close()

def update_worker_profile(worker_id: int, name: str, worker_code: str = None, department: str = None, shift: str = None, notes: str = None) -> bool:
    session = SessionLocal()
    try:
        worker = session.query(Worker).filter(Worker.id == worker_id).first()
        if worker:
            worker.name = name.strip()
            if worker_code:
                worker.worker_code = worker_code.strip()
            if department:
                worker.department = department.strip()
            if shift:
                worker.shift = shift.strip()
            if notes is not None:
                worker.notes = notes.strip()
            worker.is_identified = True
            session.commit()
            return True
        return False
    finally:
        session.close()

def log_emotion(worker_id: int, emotion: str, confidence: float):
    session = SessionLocal()
    try:
        # Ağırlığı Emotion Map'ten bul
        weight = 0.0
        for info in EMOTION_MAP.values():
            if info["tr"] == emotion or info["name"] == emotion:
                weight = info.get("weight", 0.0)
                break

        log_entry = EmotionLog(
            worker_id=worker_id,
            emotion=emotion,
            confidence=float(confidence),
            morale_weight=weight,
            timestamp=datetime.datetime.now()
        )
        session.add(log_entry)
        session.commit()
    finally:
        session.close()

def import_workers_from_csv(csv_path: str) -> tuple[int, int]:
    """
    CSV dosyasından personel listesini içeri aktarır.
    Format: worker_code, name, department, shift
    """
    session = SessionLocal()
    added, skipped = 0, 0
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            code = str(row.get("worker_code", "")).strip()
            name = str(row.get("name", "")).strip()
            dept = str(row.get("department", "Genel")).strip()
            shift = str(row.get("shift", "Gündüz (08:00 - 16:00)")).strip()

            if not code or not name:
                continue

            existing = session.query(Worker).filter(Worker.worker_code == code).first()
            if existing:
                skipped += 1
                continue

            # Dummy embedding ile ön kayıt açılabilir (Kamera ilk gördüğünde embedding atanır)
            dummy_emb = np.zeros((1, 128), dtype=np.float32)
            w = Worker(
                worker_code=code,
                name=name,
                department=dept,
                shift=shift,
                is_identified=True
            )
            w.set_embedding(dummy_emb)
            session.add(w)
            added += 1

        session.commit()
        return added, skipped
    finally:
        session.close()
