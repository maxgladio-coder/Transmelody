@echo off
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "melody_audition_ui.py" %*
