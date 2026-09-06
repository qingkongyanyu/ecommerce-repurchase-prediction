@echo off
chcp 65001 >nul
setlocal
title 电商复购预测系统 - 一键启动
set "ROOT=%~dp0"

echo ============================================
echo  电商复购预测系统 - 一键启动
echo ============================================
echo.

echo [1/2] 正在启动后端服务 (http://127.0.0.1:8000) ...
start "ecommerce-backend" /D "%ROOT%backend" cmd /k "python main.py"

echo [2/2] 正在启动前端大屏 (http://127.0.0.1:3000) ...
start "ecommerce-frontend" /D "%ROOT%frontend" cmd /k "npm run dev"

echo.
echo 已启动，请稍候几秒后访问:
echo   接口文档:   http://127.0.0.1:8000/docs
echo   可视化大屏: http://127.0.0.1:3000
echo.
echo 关闭本窗口不会停止服务，如需停止请直接关闭对应命令行窗口。
pause
endlocal
