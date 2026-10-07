@echo off
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run.ps1"
