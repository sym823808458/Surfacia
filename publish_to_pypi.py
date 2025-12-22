#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyPI 发布脚本 - Surfacia Package
自动化构建和发布流程
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"\n🔄 {description}")
    print(f"执行命令: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 错误: {description}")
        print(f"标准输出: {result.stdout}")
        print(f"错误输出: {result.stderr}")
        return False
    else:
        print(f"✅ 成功: {description}")
        if result.stdout:
            print(f"输出: {result.stdout}")
        return True

def clean_build_dirs():
    """清理构建目录"""
    dirs_to_clean = ['build', 'dist', 'surfacia.egg-info']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"🧹 清理目录: {dir_name}")
            shutil.rmtree(dir_name)

def check_requirements():
    """检查必要的工具是否安装"""
    required_tools = ['twine', 'build']
    missing_tools = []
    
    for tool in required_tools:
        result = subprocess.run(f"pip show {tool}", shell=True, capture_output=True)
        if result.returncode != 0:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"❌ 缺少必要工具: {', '.join(missing_tools)}")
        print("请先安装: pip install build twine")
        return False
    
    print("✅ 所有必要工具已安装")
    return True

def validate_package_structure():
    """验证包结构"""
    required_files = [
        'setup.py',
        'pyproject.toml', 
        'README.md',
        'LICENSE',
        'requirements.txt',
        'surfacia/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 包结构验证通过")
    return True

def build_package():
    """构建包"""
    print("\n📦 开始构建包...")
    
    # 使用现代构建工具
    if not run_command("python -m build", "构建源码包和wheel包"):
        return False
    
    # 检查构建结果
    if not os.path.exists('dist'):
        print("❌ 构建失败: dist目录不存在")
        return False
    
    dist_files = os.listdir('dist')
    if not dist_files:
        print("❌ 构建失败: dist目录为空")
        return False
    
    print(f"✅ 构建成功，生成文件: {', '.join(dist_files)}")
    return True

def check_package():
    """检查包的完整性"""
    print("\n🔍 检查包完整性...")
    
    if not run_command("twine check dist/*", "检查包完整性"):
        return False
    
    return True

def upload_to_test_pypi():
    """上传到测试PyPI"""
    print("\n🧪 上传到测试PyPI...")
    
    cmd = "twine upload --repository testpypi dist/*"
    print(f"执行命令: {cmd}")
    print("请输入您的TestPyPI用户名和密码...")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def upload_to_pypi():
    """上传到正式PyPI"""
    print("\n🚀 上传到正式PyPI...")
    
    cmd = "twine upload dist/*"
    print(f"执行命令: {cmd}")
    print("请输入您的PyPI用户名和密码...")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    """主函数"""
    print("🎯 Surfacia PyPI 发布工具")
    print("=" * 50)
    
    # 检查当前目录
    if not os.path.exists('surfacia'):
        print("❌ 错误: 请在Surfacia项目根目录运行此脚本")
        sys.exit(1)
    
    # 步骤1: 检查必要工具
    if not check_requirements():
        sys.exit(1)
    
    # 步骤2: 验证包结构
    if not validate_package_structure():
        sys.exit(1)
    
    # 步骤3: 清理旧的构建文件
    clean_build_dirs()
    
    # 步骤4: 构建包
    if not build_package():
        sys.exit(1)
    
    # 步骤5: 检查包完整性
    if not check_package():
        sys.exit(1)
    
    # 步骤6: 选择发布目标
    print("\n📋 选择发布目标:")
    print("1. 测试PyPI (推荐先测试)")
    print("2. 正式PyPI")
    print("3. 两者都发布")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        if upload_to_test_pypi():
            print("\n✅ 成功上传到测试PyPI!")
            print("测试安装: pip install -i https://test.pypi.org/simple/ surfacia")
        else:
            print("\n❌ 上传到测试PyPI失败")
            sys.exit(1)
    
    elif choice == "2":
        confirm = input("⚠️  确认上传到正式PyPI? (yes/no): ").strip().lower()
        if confirm == "yes":
            if upload_to_pypi():
                print("\n🎉 成功发布到PyPI!")
                print("安装命令: pip install surfacia")
            else:
                print("\n❌ 上传到PyPI失败")
                sys.exit(1)
        else:
            print("取消发布")
    
    elif choice == "3":
        # 先上传测试PyPI
        if upload_to_test_pypi():
            print("\n✅ 成功上传到测试PyPI!")
            
            confirm = input("继续上传到正式PyPI? (yes/no): ").strip().lower()
            if confirm == "yes":
                if upload_to_pypi():
                    print("\n🎉 成功发布到PyPI!")
                    print("安装命令: pip install surfacia")
                else:
                    print("\n❌ 上传到PyPI失败")
                    sys.exit(1)
        else:
            print("\n❌ 上传到测试PyPI失败")
            sys.exit(1)
    
    else:
        print("无效选择")
        sys.exit(1)

if __name__ == "__main__":
    main()