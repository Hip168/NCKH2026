@echo off
rem Tat toan bo he thong, giu lai image va du lieu.
cd /d "%~dp0infra"
docker compose --profile cham-diem down --remove-orphans
pause
