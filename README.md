# Real-time Yüz Tanıma, Dinamik ID Atama ve Duygu Analizi

Bu proje; web kamerasından alınan canlı video akışında insan yüzlerini derin öğrenme modelleriyle tespit eder, kişiye özel 128 boyutlu yüz öznitelik vektörü (embedding) çıkararak benzersiz bir ID atar, sonraki gelişlerinde aynı ID ile tanır, anlık duygu durumunu analiz eder ve tüm bu verileri kalıcı bir veritabanına kaydeder.

---

## 🚀 Özellikler

1. **Modern Yüz Tespiti (YuNet):**
   * Klasik ve hatalı Haar Cascade yerine OpenCV'nin resmi CNN tabanlı `YuNet` modeli kullanılır. Farklı açılardan ve ışık koşullarından etkilenmez.
2. **Derin Öğrenme Tabanlı Yüz Tanıma (SFace):**
   * Eski LBPH algoritması yerine `SFace` (128-D Cosine Embedding) kullanılır.
   * Model yeniden eğitime ihtiyaç duymaz; yeni bir yüz görüldüğünde milisaniyeler içinde yeni ID oluşturulur ve sonraki karelerde hemen tanınır.
3. **Kareler Arası Kararlı Takip (Face Tracker):**
   * IoU tabanlı takipçi sayesinde ekrandaki yüzlerin ID'si titreme (flicker) yapmaz.
4. **Gerçek Zamanlı Duygu Analizi (FERPlus ONNX):**
   * 8 temel duygu sınıfı (Nötr, Mutlu, Şaşkın, Üzgün, Öfkeli, İğrenmiş, Korkmuş, Mesafeli) tespit edilir.
   * Her duygu için özel renk kodlu dinamik HUD sınır çizgileri çizilir.
5. **Kalıcı Veritabanı ve Loglama:**
   * Kişi profilleri (`users`: id, isim, ilk görülme, son görülme, embedding, yüz fotoğrafı) kalıcı olarak saklanır.
   * Duygu akışları (`emotion_logs`: user_id, duygu, güven skoru, zaman damgası) saniyede bir veritabanına loglanır.
   * Varsayılan olarak yerel `data/face_records.db` (SQLite) dosyasına yazılır. İstenirse tek satırla PostgreSQL'e bağlanabilir.

---

## 📂 Proje Yapısı

```text
Face-recognition/
├── models/                     # ONNX derin öğrenme modelleri
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── emotion-ferplus-8.onnx
├── data/                       # Yüz fotoğrafları ve yerel veritabanı
│   ├── faces/                  # Kaydedilen ilk yüz fotoğrafları
│   └── face_records.db         # Kalıcı SQLite veritabanı
├── config.py                   # Eşik değerleri, dosya yolları ve DB bağlantısı
├── database.py                 # SQLAlchemy modelleri (User, EmotionLog) ve CRUD
├── face_engine.py              # YuNet ve SFace entegrasyonu, dinamik kayıt motoru
├── emotion_engine.py           # FERPlus duygu çıkarım motoru
├── tracker.py                  # Kararlı yüz takip modülü
├── model_downloader.py         # Modelleri otomatik indiren yardımcı script
├── main.py                     # Gerçek zamanlı kamera döngüsü ve modern HUD
├── report.py                   # Kayıtlı kişileri ve duygu istatistiklerini raporlama
└── requirements.txt            # Bağımlılıklar
```

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Modelleri İndirin (Zaten indirilmişse kontrol eder):
```bash
python model_downloader.py
```

### 3. Uygulamayı Başlatın:
```bash
python main.py
```

---

## 🎮 Klavye Kontrolleri

* **`Q` veya `ESC`**: Kamerayı ve programı güvenli şekilde sonlandırır.
* **`S`**: Konsol ekranına anlık veritabanı özetini (toplam kişi ve duygu dağılımı) yazdırır.

---

## 📊 Raporlama ve Dışa Aktarma

Kayıtlı kişileri ve duygu geçmişini terminalde görmek veya CSV olarak dışa aktarmak için:
```bash
python report.py
```

---

## 🗄️ PostgreSQL Entegrasyonu (Opsiyonel)

Varsayılan SQLite yerine PostgreSQL kullanmak için `config.py` dosyasında veya ortam değişkeninde `DATABASE_URL` değerini değiştirmeniz yeterlidir:

```python
# config.py veya .env
DATABASE_URL = "postgresql+psycopg2://postgres:SIFRENIZ@localhost:5432/facedb"
```
SQLAlchemy sayesinde hiçbir kod değişikliği yapmadan tablolar PostgreSQL'de otomatik açılacaktır.
