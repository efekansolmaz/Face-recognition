import datetime
import pickle
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import DATABASE_URL

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    first_seen = Column(DateTime, default=datetime.datetime.now)
    last_seen = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
    total_frames = Column(Integer, default=1)
    photo_path = Column(String(255), nullable=True)
    embedding = Column(LargeBinary, nullable=False)  # 128-D float numpy array

    emotion_logs = relationship("EmotionLog", back_populates="user", cascade="all, delete-orphan")

    def get_embedding(self) -> np.ndarray:
        return pickle.loads(self.embedding)

    def set_embedding(self, emb: np.ndarray):
        self.embedding = pickle.dumps(emb)

    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', first_seen='{self.first_seen}')>"


class EmotionLog(Base):
    __tablename__ = "emotion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    emotion = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now)

    user = relationship("User", back_populates="emotion_logs")

    def __repr__(self):
        return f"<EmotionLog(user_id={self.user_id}, emotion='{self.emotion}', confidence={self.confidence:.2f})>"


# --- ENGINE & SESSION ---
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Veritabani hazirlandi: {DATABASE_URL}")

# Modul yuklendiginde tablolari otomatik hazirla
init_db()

def load_all_users():
    session = SessionLocal()
    try:
        users = session.query(User).all()
        user_list = []
        for u in users:
            try:
                emb = u.get_embedding()
                user_list.append({
                    "id": u.id,
                    "name": u.name,
                    "embedding": emb,
                    "photo_path": u.photo_path
                })
            except Exception as e:
                print(f"[DB UYARI] User {u.id} embedding okunamadi: {e}")
        return user_list
    finally:
        session.close()

def create_user(embedding: np.ndarray, name: str = None, photo_path: str = None) -> dict:
    session = SessionLocal()
    try:
        user_count = session.query(User).count()
        assigned_name = name or f"Kisi_{user_count + 1}"
        
        new_user = User(
            name=assigned_name,
            first_seen=datetime.datetime.now(),
            last_seen=datetime.datetime.now(),
            total_frames=1,
            photo_path=photo_path
        )
        new_user.set_embedding(embedding)
        
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
        
        return {
            "id": new_user.id,
            "name": new_user.name,
            "embedding": embedding,
            "photo_path": photo_path
        }
    finally:
        session.close()

def update_user_seen(user_id: int, new_embedding: np.ndarray = None):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if user:
            user.last_seen = datetime.datetime.now()
            user.total_frames += 1
            
            # Dinamik hafif embedding guncellemesi (yuz acilari degistikce daha guclu temsil saglar)
            if new_embedding is not None:
                current_emb = user.get_embedding()
                # 0.95 eski + 0.05 yeni agirlikli hareketli ortalama
                updated_emb = (0.95 * current_emb) + (0.05 * new_embedding)
                user.set_embedding(updated_emb)
                
            session.commit()
    finally:
        session.close()

def update_user_name(user_id: int, new_name: str) -> bool:
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if user:
            user.name = new_name
            session.commit()
            return True
        return False
    finally:
        session.close()

def log_emotion(user_id: int, emotion: str, confidence: float):
    session = SessionLocal()
    try:
        log_entry = EmotionLog(
            user_id=user_id,
            emotion=emotion,
            confidence=float(confidence),
            timestamp=datetime.datetime.now()
        )
        session.add(log_entry)
        session.commit()
    finally:
        session.close()

def get_stats():
    session = SessionLocal()
    try:
        user_count = session.query(User).count()
        log_count = session.query(EmotionLog).count()
        top_emotions = session.query(
            EmotionLog.emotion, func.count(EmotionLog.id)
        ).group_by(EmotionLog.emotion).all()
        
        return {
            "user_count": user_count,
            "log_count": log_count,
            "top_emotions": dict(top_emotions)
        }
    finally:
        session.close()
