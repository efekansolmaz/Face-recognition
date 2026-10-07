@echo off
title Yuz Tanima ve Duygu Analizi
cd /d "%~dp0"
echo ============================================================
echo   GERCEK ZAMANLI YUZ TANIMA VE DUYGU ANALIZI
echo ============================================================
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Bir hata olustu! Lutfen yukaridaki hata mesajini kontrol edin.
)
pause
