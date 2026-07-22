@echo off
rem Запуск веб-сервера RAG-поиска через python из .venv.
rem Аргументы пробрасываются в src.server, например:
rem     run_server.bat --port 9000 --host 0.0.0.0
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ОШИБКА] Не найден .venv\Scripts\python.exe
    echo Создайте окружение по инструкции из README.md.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m src.server %*
endlocal
