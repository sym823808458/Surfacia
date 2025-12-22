"""
文件工具函数
"""
import os
import glob
import shutil
import pandas as pd
from pathlib import Path
import logging

def setup_logging(log_file_path):
    """设置日志记录"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

def ensure_directory(directory_path):
    """确保目录存在，如果不存在则创建"""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def find_files_by_pattern(directory, pattern):
    """在目录中查找匹配模式的文件"""
    search_path = os.path.join(directory, pattern)
    return sorted(glob.glob(search_path))

def find_files_by_extension(directory, extension):
    """在目录中查找指定扩展名的文件"""
    if not extension.startswith('.'):
        extension = '.' + extension
    pattern = f"*{extension}"
    return find_files_by_pattern(directory, pattern)

def check_file_exists_with_extensions(base_name, extensions):
    """检查具有指定扩展名的文件是否存在"""
    for ext in extensions:
        if os.path.exists(f"{base_name}{ext}"):
            return True, f"{base_name}{ext}"
    return False, None

def read_text_file(file_path, encoding='utf-8'):
    """读取文本文件"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return None

def write_text_file(file_path, content, encoding='utf-8'):
    """写入文本文件"""
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logging.error(f"Failed to write {file_path}: {e}")
        return False

def clean_temp_files(directory, temp_patterns=None):
    """清理临时文件"""
    if temp_patterns is None:
        temp_patterns = ['*.tmp', '*.temp', 'xtbrestart', 'xtbtopo.mol', 'charges', 'wbo', 'fort.7']
    
    cleaned_files = []
    for pattern in temp_patterns:
        files = glob.glob(os.path.join(directory, pattern))
        for file in files:
            try:
                os.remove(file)
                cleaned_files.append(file)
            except Exception as e:
                logging.error(f"Failed to remove {file}: {e}")
    
    return cleaned_files