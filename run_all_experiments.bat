@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo 一鍵執行 01~08 實驗...
python scripts\run_all_experiments.py
echo.
echo 完成。可再執行: python scripts\compare_all_results.py
pause
