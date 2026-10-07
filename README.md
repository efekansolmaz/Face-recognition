# Fabrika Personel Ruh Hali, Vardiya & Yüz Tanıma Analitiği

Bu sistem; çok sayıda işçinin çalıştığı fabrika, atölye ve üretim tesislerinde insan yüzlerini gerçek zamanlı olarak tespit edip tanır, vardiya boyu çalışma sürelerini ve ruh hallerini analiz eder. Gün sonunda İK (İnsan Kaynakları) ve İSG yöneticilerine hem **interaktif bir Web Dashboard** hem de yazdırılabilir **Gün Sonu PDF Raporu** sunar.

---

## 🚀 Öne Çıkan Özellikler

1. **Çoklu İşçi Tespiti & Tanıma:**
   * **OpenCV YuNet (CNN):** Kalabalık kadrajlarda aynı anda çok sayıda yüzü kaçırmadan tespit eder.
   * **SFace (128D Embedding):** Yeni giren işçiye anında `ISC-XXX` kodu atar, yüz vektörünü kaydeder ve sonraki geçişlerinde otomatik tanır.
2. **Kişi Bazlı "Günlük Moral Skoru" (0 - 100):**
   * Ham duygular yerine ağırlıklı bir formülle hesaplanır:
     * **Pozitif (+):** Mutlu (`+1.0`)
     * **Nötr (0.0):** Normal çalışma hali
     * **Negatif (-):** Öfkeli (`-1.0`), Üzgün (`-0.8`), Korku (`-0.7`), İğrenme (`-0.5`)
   * **Skor Aralıkları:**
     * `80 - 100`: Yüksek Motivasyon / Pozitif
     * `60 - 79`: Normal / Dengeli
     * `45 - 59`: Hafif Düşük / Yorgun
     * `< 45`: 🚨 **Yüksek Stres / Kötü Gün (Kritik İnceleme Listesi)**
3. **Hibrit Personel Yönetimi:**
   * **Sıfır Kurulumla Başlama:** Kamera ilk gördüğü işçiye ID verir ve verileri toplamaya başlar.
   * **Dashboard'dan İsimlendirme:** Yönetici, kameranın çektiği fotoğrafın altına tıklayarak tek tıkla gerçek Ad Soyad, Sicil No ve Departman atayabilir.
   * **CSV İçe Aktarma:** İK'nın elindeki personel listesi (Sicil No, Ad Soyad, Departman, Vardiya) tek tıkla sisteme yüklenebilir.
4. **Yönetici Web Dashboard'u (`http://127.0.0.1:5000`):**
   * Yönetici KPI kartları (Aktif işçi, fabrika ortalama morali, kötü gün geçiren işçi sayısı).
   * ⚠️ **Kötü Gün / Riskli Personel Uyarı Bandı**.
   * Vardiya boyu saatlik moral trendi çizgi grafiği (Chart.js).
   * Departman bazlı karşılaştırma çubuk grafiği.
   * Arama, filtreleme ve personele özel düzenleme özellikli karne tablosu.
   * Tarayıcıdan izlenebilen canlı kamera önizleme penceresi.
5. **Kurumsal Gün Sonu PDF Raporu:**
   * Tek tıkla indirilebilir A4 formatında, tam Türkçe Unicode destekli kurumsal PDF.
   * Yönetici özeti, departman karşılaştırmaları, kötü gün geçirenlerin analiz ve tavsiye tablosu ve tüm işçilerin detaylı gün sonu karnesi.

---

## 📂 Proje Yapısı

```text
Face-recognition/
├── models/                     # ONNX modelleri (YuNet, SFace, FERPlus)
├── data/                       # Veriler ve Raporlar
│   ├── faces/                  # İşçilerin yakalanan profil fotoğrafları
│   ├── reports/                # Üretilen Gün Sonu PDF raporları
│   └── face_records.db         # Kalıcı SQLite veritabanı (veya PostgreSQL)
├── templates/
│   └── dashboard.html          # Modern, responsive koyu modlu Web Paneli
├── web_dashboard.py            # Flask Web Sunucusu ve REST API'ler
├── pdf_report.py               # ReportLab tabanlı otomatik PDF rapor motoru
├── analytics.py                # Moral hesaplayıcı ve fabrika istatistik motoru
├── database.py                 # Worker ve EmotionLog SQLAlchemy modelleri
├── face_engine.py              # YuNet + SFace derin öğrenme yüz motoru
├── emotion_engine.py           # FERPlus 8 sınıflı duygu analiz motoru
├── tracker.py                  # Çoklu işçi IoU takipçisi
├── main.py                     # Doğrudan masaüstü kamera penceresi HUD
├── baslat.bat                  # Kolay başlatma menüsü (Dashboard / Kamera / PDF)
└── requirements.txt            # Bağımlılıklar
```

---

## 🛠️ Nasıl Çalıştırılır?

### 1. En Kolay Yol:
Proje dizinindeki **`baslat.bat`** dosyasına çift tıklayın ve menüden seçiminizi yapın:
* **`[1]`**: Web Yönetici Dashboard'unu başlatır ve tarayıcınızda açar (`http://127.0.0.1:5000`).
* **`[2]`**: Doğrudan masaüstü kamera penceresini açar.
* **`[3]`**: O güne ait Gün Sonu PDF Raporunu derler.

### 2. Terminalden Çalıştırma:
* **Dashboard'u Açmak İçin:**
  ```bash
  python web_dashboard.py
  ```
  Tarayıcınızdan `http://127.0.0.1:5000` adresine gidin.

* **PDF Raporu Üretmek İçin:**
  ```bash
  python pdf_report.py
  ```
