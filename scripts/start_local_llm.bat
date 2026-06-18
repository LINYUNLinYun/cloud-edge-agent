@echo off
REM ============================================================
REM 启动本地 LLM 服务 (Ollama)
REM ============================================================
REM 如果没有安装 Ollama，请先下载安装：
REM https://ollama.com/download/windows
REM ============================================================

echo.
echo ╔═══════════════════════════════════════════════════╗
echo ║   启动本地 LLM 服务 (Ollama)                     ║
echo ╚═══════════════════════════════════════════════════╝
echo.

REM 检查 Ollama 是否安装
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Ollama，请先安装：
    echo https://ollama.com/download/windows
    pause
    exit /b 1
)

REM 检查 Ollama 是否已运行
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ollama 已在运行
) else (
    echo [启动] 正在启动 Ollama 服务...
    start /b ollama serve
    timeout /t 3 >nul
)

REM 检查模型是否存在
ollama list | findstr "qwen2.5" >nul 2>&1
if %errorlevel% neq 0 (
    echo [下载] 正在下载 Qwen 2.5 模型...
    ollama pull qwen2.5:1.5b
)

echo.
echo ============================================================
echo   本地 LLM 服务已就绪！
echo   API 地址: http://localhost:11434/v1
echo   模型: qwen2.5:1.5b
echo ============================================================
echo.
echo 测试命令:
echo   curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen2.5:1.5b\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
echo.
pause
