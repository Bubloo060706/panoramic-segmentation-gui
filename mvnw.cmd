@echo off
set SCRIPT_DIR=%~dp0
set MAVEN_CMD=C:\Users\balaj\.maven\apache-maven-3.9.9\bin\mvn.cmd
if exist "%MAVEN_CMD%" (
    "%MAVEN_CMD%" -s "%SCRIPT_DIR%.mvn\settings.xml" %*
) else (
    mvn -s "%SCRIPT_DIR%.mvn\settings.xml" %*
)
