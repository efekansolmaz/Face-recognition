import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import analytics
from config import REPORTS_DIR

# Türkçe karakter destekli font kaydı (Windows sistem fontları)
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

try:
    if os.path.exists("C:/Windows/Fonts/arial.ttf") and os.path.exists("C:/Windows/Fonts/arialbd.ttf"):
        pdfmetrics.registerFont(TTFont("ArialTR", "C:/Windows/Fonts/arial.ttf"))
        pdfmetrics.registerFont(TTFont("ArialTR-Bold", "C:/Windows/Fonts/arialbd.ttf"))
        FONT_NAME = "ArialTR"
        FONT_BOLD = "ArialTR-Bold"
        print("[PDF] ArialTR fontu yuklendi (Tam Turkce destegi).")
except Exception as e:
    print(f"[PDF UYARI] Ozel font yuklenemedi, varsayilan kullaniliyor: {e}")

class NumberedCanvas(canvas.Canvas):
    """Sayfa numarası ve alt bilgi ekleyen canvas."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont(FONT_NAME, 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Alt Bilgi Çizgisi
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 35, 555, 35)
        
        # Sol ve Sağ Metin
        now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        self.drawString(40, 24, f"Fabrika İSG & İK Yönetim Raporu | Oluşturulma: {now_str}")
        self.drawRightString(555, 24, f"Sayfa {self._pageNumber} / {page_count}")
        self.restoreState()

def generate_daily_pdf(target_date: datetime.date = None, output_path: str = None) -> str:
    """
    Seçilen gün için profesyonel Gün Sonu PDF raporunu oluşturur.
    """
    if target_date is None:
        target_date = datetime.date.today()

    if output_path is None:
        date_str = target_date.strftime("%Y-%m-%d")
        output_path = os.path.join(REPORTS_DIR, f"Gun_Sonu_Ruh_Hali_Raporu_{date_str}.pdf")

    summary = analytics.get_daily_factory_summary(target_date)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Özel Stiller
    title_style = ParagraphStyle(
        "DocTitle",
        fontName=FONT_BOLD,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName=FONT_NAME,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        fontName=FONT_BOLD,
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=14,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        "Body",
        fontName=FONT_NAME,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        fontName=FONT_BOLD,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # 1. Başlık Alanı
    story.append(Paragraph("FABRİKA PERSONEL RUH HALİ & VARDİYA GÜN SONU RAPORU", title_style))
    date_formatted = target_date.strftime("%d %B %Y")
    story.append(Paragraph(f"Tarih: <b>{date_formatted}</b> | Kapsam: Günlük Çalışma Periyodu | Yapay Zeka Yüz ve Duygu Analitiği", subtitle_style))

    # 2. Yönetici Özet KPI Kutuları
    kpi_data = [
        [
            Paragraph("<b>TOPLAM AKTİF İŞÇİ</b>", body_style),
            Paragraph("<b>FABRİKA GENEL MORALİ</b>", body_style),
            Paragraph("<b>RİSKLİ / DESTEK GEREKEN</b>", body_style),
            Paragraph("<b>GENEL DURUM</b>", body_style)
        ],
        [
            Paragraph(f"<font size=16><b>{summary['total_active_workers']}</b></font> Kişi", body_bold),
            Paragraph(f"<font size=16 color='#2B6CB0'><b>{summary['average_factory_morale']} / 100</b></font>", body_bold),
            Paragraph(f"<font size=16 color='{('#E53E3E' if summary['risk_count'] > 0 else '#38A169')}'><b>{summary['risk_count']}</b></font> Kişi", body_bold),
            Paragraph(f"<font size=13><b>{summary['morale_status']}</b></font>", body_bold)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[128, 128, 128, 131])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # 3. İK ve Amirler İçin Kritik Risk Tablosu (Kötü Gün Geçirenler)
    story.append(Paragraph("⚠️ ÖNCELİKLİ İNCELEME & DESTEK LİSTESİ (Günü Kötü / Stresli Geçenler)", section_heading))
    if summary["risk_count"] == 0:
        story.append(Paragraph("<i>Bugün kritik seviyede moralsiz veya aşırı stresli işçi tespit edilmedi. Tüm fabrika dengeli/normal çalışma temposunda seyretti.</i>", body_style))
    else:
        risk_rows = [
            [
                Paragraph("<b>Sicil No</b>", body_bold),
                Paragraph("<b>Adı Soyadı</b>", body_bold),
                Paragraph("<b>Departman</b>", body_bold),
                Paragraph("<b>Moral Skoru</b>", body_bold),
                Paragraph("<b>Baskın Duygu</b>", body_bold),
                Paragraph("<b>Tespit & Tavsiye</b>", body_bold)
            ]
        ]
        for w in summary["at_risk_workers"]:
            risk_rows.append([
                Paragraph(w["worker_code"], body_style),
                Paragraph(f"<b>{w['name']}</b>", body_style),
                Paragraph(w["department"], body_style),
                Paragraph(f"<font color='#E53E3E'><b>{w['morale_score']}/100</b></font>", body_bold),
                Paragraph(w["dominant_emotion"], body_style),
                Paragraph(w["risk_reason"], body_style)
            ])
        risk_table = Table(risk_rows, colWidths=[65, 95, 90, 65, 75, 125])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FED7D7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#9B2C2C")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#FEB2B2")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FED7D7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(risk_table)

    story.append(Spacer(1, 10))

    # 4. Departman Bazlı Ruh Hali Dağılımı
    story.append(Paragraph("📊 DEPARTMAN BAZLI ORTALAMA MORAL", section_heading))
    dept_rows = [[Paragraph("<b>Departman</b>", body_bold), Paragraph("<b>İşçi Sayısı</b>", body_bold), Paragraph("<b>Ortalama Moral</b>", body_bold), Paragraph("<b>Durum</b>", body_bold)]]
    for dept, data in summary["department_stats"].items():
        m_score = data["avg_morale"]
        status = "İyi" if m_score >= 75 else ("Normal" if m_score >= 60 else "Düşük / Stresli")
        dept_rows.append([
            Paragraph(dept, body_style),
            Paragraph(str(data["worker_count"]), body_style),
            Paragraph(f"<b>{m_score} / 100</b>", body_style),
            Paragraph(status, body_style)
        ])
    dept_table = Table(dept_rows, colWidths=[180, 110, 110, 115])
    dept_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(dept_table)

    story.append(Spacer(1, 10))

    # 5. Tüm Personel Gün Sonu Karnesi (Detaylı Liste)
    story.append(Paragraph("📋 TÜM PERSONEL GÜN SONU KARNESİ", section_heading))
    all_rows = [
        [
            Paragraph("<b>Sicil No</b>", body_bold),
            Paragraph("<b>Adı Soyadı</b>", body_bold),
            Paragraph("<b>Departman</b>", body_bold),
            Paragraph("<b>Süre (dk)</b>", body_bold),
            Paragraph("<b>Baskın Duygu</b>", body_bold),
            Paragraph("<b>Moral Skoru</b>", body_bold),
            Paragraph("<b>Durum</b>", body_bold)
        ]
    ]

    for w in summary["all_workers"]:
        score = w["morale_score"]
        score_color = "#38A169" if score >= 80 else ("#3182CE" if score >= 60 else ("#D69E2E" if score >= 45 else "#E53E3E"))
        all_rows.append([
            Paragraph(w["worker_code"], body_style),
            Paragraph(f"<b>{w['name']}</b>", body_style),
            Paragraph(w["department"], body_style),
            Paragraph(f"{w['active_duration_minutes']} dk", body_style),
            Paragraph(w["dominant_emotion"], body_style),
            Paragraph(f"<font color='{score_color}'><b>{score} / 100</b></font>", body_bold),
            Paragraph(w["status_label"], body_style)
        ])

    all_table = Table(all_rows, colWidths=[65, 100, 95, 55, 75, 65, 60])
    all_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(all_table)

    # Belgeyi derle
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF] Rapor basariyla olusturuldu: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_daily_pdf()
