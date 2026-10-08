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

FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

try:
    if os.path.exists("C:/Windows/Fonts/arial.ttf") and os.path.exists("C:/Windows/Fonts/arialbd.ttf"):
        pdfmetrics.registerFont(TTFont("ArialTR", "C:/Windows/Fonts/arial.ttf"))
        pdfmetrics.registerFont(TTFont("ArialTR-Bold", "C:/Windows/Fonts/arialbd.ttf"))
        FONT_NAME = "ArialTR"
        FONT_BOLD = "ArialTR-Bold"
except Exception:
    pass

class NumberedCanvas(canvas.Canvas):
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
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(40, 35, 555, 35)
        now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        self.drawString(40, 24, f"Fabrika İSG & İK Yönetim Raporu | Oluşturulma: {now_str}")
        self.drawRightString(555, 24, f"Sayfa {self._pageNumber} / {page_count}")
        self.restoreState()

def generate_daily_pdf(target_date: datetime.date = None, output_path: str = None) -> str:
    if target_date is None:
        target_date = datetime.date.today()

    if output_path is None:
        date_str = target_date.strftime("%Y-%m-%d")
        output_path = os.path.join(REPORTS_DIR, f"Gun_Sonu_Ruh_Hali_Raporu_{date_str}.pdf")

    summary = analytics.get_daily_factory_summary(target_date)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle", fontName=FONT_BOLD, fontSize=18, leading=22, textColor=colors.HexColor("#1A365D"), spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", fontName=FONT_NAME, fontSize=9, leading=12, textColor=colors.HexColor("#4A5568"), spaceAfter=12
    )
    section_heading = ParagraphStyle(
        "SectionHeading", fontName=FONT_BOLD, fontSize=11, leading=15, textColor=colors.HexColor("#2D3748"), spaceBefore=10, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", fontName=FONT_NAME, fontSize=8, leading=11, textColor=colors.HexColor("#2D3748")
    )
    body_bold = ParagraphStyle(
        "BodyBold", fontName=FONT_BOLD, fontSize=8, leading=11, textColor=colors.HexColor("#1A202C")
    )

    story = []

    # 1. Başlık Alanı
    story.append(Paragraph("FABRİKA PERSONEL RUH HALİ, VARDİYA & İSG GÜN SONU RAPORU", title_style))
    date_formatted = target_date.strftime("%d.%m.%Y")
    story.append(Paragraph(f"Tarih: <b>{date_formatted}</b> | Vardiya Analizi | Yapay Zeka Yüz Tanıma, Duygu & İş Kazası Risk Skoru", subtitle_style))

    # 2. KPI Tablosu
    kpi_data = [
        [
            Paragraph("<b>TOPLAM AKTİF İŞÇİ</b>", body_style),
            Paragraph("<b>FABRİKA GENEL MORALİ</b>", body_style),
            Paragraph("<b>KÖTÜ GÜN GEÇİREN</b>", body_style),
            Paragraph("<b>YÜKSEK KAZA RİSKİ</b>", body_style)
        ],
        [
            Paragraph(f"<font size=14><b>{summary['total_active_workers']}</b></font> Kişi", body_bold),
            Paragraph(f"<font size=14 color='#2B6CB0'><b>{summary['average_factory_morale']} / 100</b></font>", body_bold),
            Paragraph(f"<font size=14 color='{('#E53E3E' if summary['risk_count'] > 0 else '#38A169')}'><b>{summary['risk_count']}</b></font> Kişi", body_bold),
            Paragraph(f"<font size=14 color='{('#C53030' if summary.get('safety_alert_count', 0) > 0 else '#38A169')}'><b>{summary.get('safety_alert_count', 0)}</b></font> Kişi", body_bold)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 133])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    # 3. İSG ve Kötü Gün Geçirenler Tablosu
    story.append(Paragraph("⚠️ ÖNCELİKLİ İNCELEME & DESTEK LİSTESİ (Kötü Gün / Kaza Riski Taşıyanlar)", section_heading))
    if summary["risk_count"] == 0 and summary.get("safety_alert_count", 0) == 0:
        story.append(Paragraph("<i>Bugün kritik seviyede moralsiz veya aşırı stresli işçi tespit edilmedi. Tüm fabrika güvenli ve dengeli çalışma temposunda seyretti.</i>", body_style))
    else:
        risk_rows = [
            [
                Paragraph("<b>Sicil No</b>", body_bold),
                Paragraph("<b>Adı Soyadı</b>", body_bold),
                Paragraph("<b>Departman</b>", body_bold),
                Paragraph("<b>Moral Skoru</b>", body_bold),
                Paragraph("<b>İSG Kaza Riski</b>", body_bold),
                Paragraph("<b>Durum & İSG Tavsiyesi</b>", body_bold)
            ]
        ]
        combined_risks = summary["at_risk_workers"]
        for w in summary["all_workers"]:
            if w.get("safety_risk_score", 0) >= 70.0 and w not in combined_risks:
                combined_risks.append(w)

        for w in combined_risks:
            risk_rows.append([
                Paragraph(w["worker_code"], body_style),
                Paragraph(f"<b>{w['name']}</b>", body_style),
                Paragraph(w["department"], body_style),
                Paragraph(f"<font color='#E53E3E'><b>{w['morale_score']}/100</b></font>", body_bold),
                Paragraph(f"<font color='#C53030'><b>{w['safety_risk_score']}/100</b></font>", body_bold),
                Paragraph(f"{w['status_label']} - {w['risk_reason']}", body_style)
            ])
        risk_table = Table(risk_rows, colWidths=[60, 95, 90, 65, 75, 138])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FED7D7")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#9B2C2C")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#FEB2B2")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FED7D7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(risk_table)

    story.append(Spacer(1, 8))

    # 4. Bölge / Kamera Lokasyonları Analizi
    if summary.get("zone_stats"):
        story.append(Paragraph("📍 FABRİKA BÖLGE / KAMERA LOKASYONLARI ANALİZİ", section_heading))
        zone_rows = [[Paragraph("<b>Kamera Bölgesi / Alan</b>", body_bold), Paragraph("<b>Ortalama Moral</b>", body_bold), Paragraph("<b>Toplam Tespit</b>", body_bold), Paragraph("<b>Bölge Durumu</b>", body_bold)]]
        for z_name, z_data in summary["zone_stats"].items():
            zone_rows.append([
                Paragraph(z_name, body_style),
                Paragraph(f"<b>{z_data['morale_score']} / 100</b>", body_style),
                Paragraph(f"{z_data['log_count']} kare", body_style),
                Paragraph(z_data["status"], body_style)
            ])
        zone_table = Table(zone_rows, colWidths=[180, 110, 110, 123])
        zone_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(zone_table)
        story.append(Spacer(1, 8))

    # 5. Tüm Personel Gün Sonu Karnesi
    story.append(Paragraph("📋 TÜM PERSONEL GÜN SONU & İSG KARNESİ", section_heading))
    all_rows = [
        [
            Paragraph("<b>Sicil No</b>", body_bold),
            Paragraph("<b>Adı Soyadı</b>", body_bold),
            Paragraph("<b>Departman</b>", body_bold),
            Paragraph("<b>Süre</b>", body_bold),
            Paragraph("<b>Baskın Duygu</b>", body_bold),
            Paragraph("<b>Moral Skoru</b>", body_bold),
            Paragraph("<b>İSG Kaza Riski</b>", body_bold)
        ]
    ]

    for w in summary["all_workers"]:
        score = w["morale_score"]
        score_color = "#38A169" if score >= 80 else ("#3182CE" if score >= 60 else ("#D69E2E" if score >= 45 else "#E53E3E"))
        safety = w.get("safety_risk_score", 15.0)
        safety_color = "#E53E3E" if safety >= 70 else ("#D69E2E" if safety >= 45 else "#38A169")
        all_rows.append([
            Paragraph(w["worker_code"], body_style),
            Paragraph(f"<b>{w['name']}</b>", body_style),
            Paragraph(w["department"], body_style),
            Paragraph(f"{w['active_duration_minutes']} dk", body_style),
            Paragraph(w["dominant_emotion"], body_style),
            Paragraph(f"<font color='{score_color}'><b>{score} / 100</b></font>", body_bold),
            Paragraph(f"<font color='{safety_color}'><b>{safety} / 100</b></font>", body_bold)
        ])

    all_table = Table(all_rows, colWidths=[60, 95, 95, 50, 75, 75, 73])
    all_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(all_table)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF] Rapor başarıyla oluşturuldu: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_daily_pdf()
