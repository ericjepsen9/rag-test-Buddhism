@echo off
chcp 65001 >nul
setlocal

:: ========== 配置区 ==========
:: 项目目录（修改为你的实际路径）
set "PROJECT_DIR=C:\Users\ericj\rag-test-Buddhism"
:: Git 分支名（按需修改）
set "BRANCH=main"
:: 环境变量
set "ADMIN_API_KEY=123456"
:: 启动命令
set "START_CMD=uvicorn api_server:app --host 0.0.0.0 --port 8000"
:: ========== 配置区结束 ==========

echo.
echo ============================================
echo   RAG 项目一键更新 ^& 重启
echo ============================================
echo.

cd /d "%PROJECT_DIR%" || (
    echo [错误] 无法进入目录: %PROJECT_DIR%
    pause
    exit /b 1
)

:: 1. 停止正在运行的项目进程
echo [1/4] 停止现有进程...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING" 2^>nul') do (
    echo       终止进程 PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)
:: 同时按进程名查找
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq api_server*" >nul 2>&1
echo       完成

:: 2. 拉取最新代码（强制覆盖本地修改）
echo.
echo [2/4] 拉取最新代码 (分支: %BRANCH%)...
git fetch origin %BRANCH%
if errorlevel 1 (
    echo [错误] git fetch 失败，请检查网络连接
    pause
    exit /b 1
)
git reset --hard origin/%BRANCH%
if errorlevel 1 (
    echo [错误] git reset 失败
    pause
    exit /b 1
)
echo       完成

:: 3. 安装/更新依赖（如果 requirements.txt 有变化）
echo.
echo [3/4] 检查依赖...
pip install -r requirements.txt -q
echo       完成

:: 4. 重启项目
echo.
echo [4/4] 启动项目...
echo       命令: %START_CMD%
echo.
echo ============================================
echo   更新完成，项目启动中...
echo   按 Ctrl+C 可停止
echo ============================================
echo.

%START_CMD%

endlocal
