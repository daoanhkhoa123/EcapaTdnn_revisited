@echo off
REM Usage: gitpush.bat "your commit message"
REM If no commit message is provided, it defaults to "fix"

set msg=%1
if "%msg%"=="" set msg=fix

git add .
git commit -m "%msg%"
git push origin 
