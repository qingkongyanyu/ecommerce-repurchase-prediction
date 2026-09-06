#!/usr/bin/env bash
# 电商复购预测系统 - 一键启动 (Linux / Git Bash)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " 电商复购预测系统 - 一键启动"
echo "============================================"

echo "[1/2] 启动后端服务 (http://127.0.0.1:8000) ..."
(cd "$ROOT/backend" && python main.py) &
BACKEND_PID=$!

echo "[2/2] 启动前端大屏 (http://127.0.0.1:3000) ..."
(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo
echo "已启动:"
echo "  接口文档:   http://127.0.0.1:8000/docs"
echo "  可视化大屏: http://127.0.0.1:3000"
echo "  Ctrl+C 停止全部服务"
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" INT TERM
wait
