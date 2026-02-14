@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo 執行實驗 01: Bankruptcy Baseline
python experiments\01_bankruptcy_baseline.py
pause
