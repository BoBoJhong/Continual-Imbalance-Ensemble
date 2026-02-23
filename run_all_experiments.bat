@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo 一鍵執行 01~10 實驗（含進階 DES 09、比例實驗 10）...
python scripts\run_all_experiments.py
echo.
echo 完成。可再執行: python scripts\compare_all_results.py
pause
