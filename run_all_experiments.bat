@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo.
echo ============================================================
echo  Continual-Imbalance-Ensemble  一鍵執行所有實驗
echo ============================================================
echo  Phase 1: Baseline          (bankruptcy / stock / medical)
echo  Phase 2: Ensemble          (static combinations)
echo  Phase 3: Dynamic           (DES / DCS)
echo  Phase 4: Feature (Study II)(feature selection sweep)
echo  Phase 5: Analysis          (proportion / split / cost)
echo ============================================================
echo.
python scripts\run\run_all_experiments.py
echo.
echo 完成。可再執行: python scripts\analysis\compare_all_results.py
pause
