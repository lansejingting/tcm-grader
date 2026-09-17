@echo off
rem 中药材智能定级 - 人工复核台启动脚本
rem 双击运行即可：会自动启动本地服务并打开浏览器
rem 关闭服务：直接关闭本窗口
chcp 65001 >nul
cd /d %~dp0
echo 正在启动人工复核台...
python 10_review_demo.py
echo.
echo 服务已退出。如果上方有红色报错，请截图反馈。
pause
