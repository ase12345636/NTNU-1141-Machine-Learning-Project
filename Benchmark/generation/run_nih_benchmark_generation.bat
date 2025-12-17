@echo off
REM =============================================================================
REM CXReasonBench 一鍵執行腳本 (Windows)
REM =============================================================================
REM 此批次檔會在WSL中執行bash腳本
REM =============================================================================

echo.
echo ========================================================
echo   CXReasonBench NIH Dataset Benchmark Generation
echo ========================================================
echo.

REM 檢查WSL是否可用
where wsl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [錯誤] WSL 未安裝或未啟用
    echo.
    echo 請安裝 WSL: https://docs.microsoft.com/zh-tw/windows/wsl/install
    pause
    exit /b 1
)

echo [檢查] WSL 已安裝 ✓
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo [執行] 啟動 WSL bash 腳本...
echo.

REM 將Windows路徑轉換為WSL路徑並執行
wsl bash -c "cd /mnt/d/CXReasonBench/Benchmark/generation && bash run_nih_benchmark_generation.sh"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   執行成功！
    echo ========================================================
    echo.
    echo 輸出位置: d:\CXReasonBench\output_nih\
    echo.
) else (
    echo.
    echo ========================================================
    echo   執行失敗！請檢查錯誤信息
    echo ========================================================
    echo.
)

pause
