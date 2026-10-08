@echo off
title Fabrika Personel Ruh Hali, ISG & Vardiya Analitigi
cd /d "%~dp0"
echo ============================================================
echo   FABRIKA PERSONEL RUH HALI, ISG & VARDIYA ANALITIGI
echo ============================================================
echo.
echo  [1] Web Yonetici Dashboard'unu Baslat (Tavsiye Edilen)
echo  [2] Dogrudan Kamera Penceresini Baslat (Masaustu HUD)
echo  [3] Gun Sonu PDF Raporu Olustur
echo  [4] Gun Sonu Excel (.xlsx) Raporu Olustur
echo.
set /p secim="Lutfen bir secim yapin (1/2/3/4) [Varsayilan: 1]: "
if "%secim%"=="" set secim=1

if "%secim%"=="1" (
    echo.
    echo Web Dashboard baslatiliyor...
    echo Tarayicinizda aciliyor: http://127.0.0.1:5000
    start http://127.0.0.1:5000
    python web_dashboard.py
)
if "%secim%"=="2" (
    echo.
    echo Kamera penceresi baslatiliyor...
    python main.py
)
if "%secim%"=="3" (
    echo.
    echo PDF Raporu olusturuluyor...
    python pdf_report.py
    pause
)
if "%secim%"=="4" (
    echo.
    echo Excel Raporu olusturuluyor...
    python -c "import analytics; analytics.export_daily_excel()"
    pause
)
pause
