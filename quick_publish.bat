@echo off
REM Surfacia 快速发布脚本 (Windows)

echo 🚀 Surfacia 快速发布到 PyPI
echo ================================

REM 检查是否在正确的目录
if not exist "setup.py" (
    echo ❌ 错误: 请在 Surfacia 项目根目录运行此脚本
    pause
    exit /b 1
)

if not exist "surfacia" (
    echo ❌ 错误: 请在 Surfacia 项目根目录运行此脚本
    pause
    exit /b 1
)

REM 安装必要工具
echo 📦 安装/更新构建工具...
pip install --upgrade build twine

REM 清理旧文件
echo 🧹 清理旧构建文件...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "surfacia.egg-info" rmdir /s /q "surfacia.egg-info"

REM 构建包
echo 🔨 构建包...
python -m build

REM 检查包
echo 🔍 检查包完整性...
twine check dist/*

if %errorlevel% neq 0 (
    echo ❌ 包检查失败，请修复错误后重试
    pause
    exit /b 1
)

echo ✅ 包构建成功！
echo.
echo 📋 接下来的步骤：
echo 1. 测试发布: twine upload --repository testpypi dist/*
echo 2. 测试安装: pip install -i https://test.pypi.org/simple/ surfacia
echo 3. 正式发布: twine upload dist/*
echo.
echo 或者运行 Python 脚本进行交互式发布:
echo python publish_to_pypi.py

pause