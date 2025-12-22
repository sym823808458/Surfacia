#!/bin/bash
# Surfacia 快速发布脚本 (Linux/Mac)

echo "🚀 Surfacia 快速发布到 PyPI"
echo "================================"

# 检查是否在正确的目录
if [ ! -f "setup.py" ] || [ ! -d "surfacia" ]; then
    echo "❌ 错误: 请在 Surfacia 项目根目录运行此脚本"
    exit 1
fi

# 安装必要工具
echo "📦 安装/更新构建工具..."
pip install --upgrade build twine

# 清理旧文件
echo "🧹 清理旧构建文件..."
rm -rf build/ dist/ *.egg-info/

# 构建包
echo "🔨 构建包..."
python -m build

# 检查包
echo "🔍 检查包完整性..."
twine check dist/*

if [ $? -ne 0 ]; then
    echo "❌ 包检查失败，请修复错误后重试"
    exit 1
fi

echo "✅ 包构建成功！"
echo ""
echo "📋 接下来的步骤："
echo "1. 测试发布: twine upload --repository testpypi dist/*"
echo "2. 测试安装: pip install -i https://test.pypi.org/simple/ surfacia"
echo "3. 正式发布: twine upload dist/*"
echo ""
echo "或者运行 Python 脚本进行交互式发布:"
echo "python publish_to_pypi.py"