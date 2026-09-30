@echo off
REM Starts the OcuGrade web app. Run from the project folder.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m webapp.server --open
pause
