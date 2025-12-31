# Surfacia 文档部署指南

本文档详细说明如何构建和部署 Surfacia 项目的代码手册。

## 📋 目录

- [本地构建文档](#本地构建文档)
- [部署到 Read the Docs](#部署到-read-the-docs)
- [常见问题排查](#常见问题排查)
- [更新文档](#更新文档)

---

## 🖥️ 本地构建文档

### 方法 1: 使用 Python 模块方式（推荐）

```bash
# 进入项目目录
cd C:\Users\YumingSu\Sym_Python_codes\Surfacia

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 安装文档依赖
cd docs
pip install -r requirements.txt

# 构建文档
sphinx-build -b html source _build/html

# 预览文档
# 在浏览器中打开 _build/html/index.html
```

### 方法 2: 使用脚本构建

创建批处理脚本 `build_docs.bat`:

```batch
@echo off
cd /d C:\Users\YumingSu\Sym_Python_codes\Surfacia\docs
sphinx-build -b html source _build/html
echo.
echo 文档构建完成！
echo 请打开 _build/html\index.html 查看文档
pause
```

### 构建输出

成功构建后，文档将生成在 `docs/_build/html/` 目录下。主要文件包括：

- `index.html` - 主页
- `descriptors/` - 描述符文档
- `api/` - API 参考
- `tutorials/` - 教程

---

## 🚀 部署到 Read the Docs

### 步骤 1: 推送代码到 GitHub

```bash
# 添加所有更改
git add .

# 提交更改
git commit -m "更新文档：添加描述符文档和修复配置"

# 推送到 GitHub
git push origin main
```

### 步骤 2: 在 Read the Docs 上注册项目

1. 访问 [Read the Docs](https://readthedocs.org/)
2. 使用 GitHub 账号登录
3. 点击 "Import a Project"
4. 填写项目信息：
   - **Name**: `surfacia`
   - **Repository**: `https://github.com/sym823808458/Surfacia`
   - **Default branch**: `main`
   - **Documentation Type**: `Sphinx`
   - **Requirements file**: `docs/requirements.txt`

### 步骤 3: 配置项目设置

在项目设置中：

1. **Advanced Settings**:
   - Python version: 3.11
   - Python interpreter: CPython
   - Build documentation with: Sphinx
   - Conf file path: `docs/source/conf.py`
   - Build directory: `_build/html`
   - Keep build directory: Yes

2. **Environment Variables** (如果需要):
   - 可以添加任何需要的环境变量

### 步骤 4: 手动触发构建

推送新代码后，可以手动触发构建：

1. 进入项目页面
2. 点击 "Builds" 标签
3. 点击 "Build version" 按钮
4. 选择 `latest` 分支

### 步骤 5: 访问文档

构建成功后，文档将在线访问：

```
https://surfacia.readthedocs.io/
```

---

## 🔧 常见问题排查

### 问题 1: Jinja2 版本冲突

**错误信息**: `ImportError: cannot import name 'environmentfilter' from 'jinja2'`

**解决方案**:

```bash
# 卸载所有相关包
pip uninstall sphinx jinja2 sphinx-autodoc-typehints -y

# 重新安装指定版本
pip install sphinx==7.4.7 jinja2==3.1.6 sphinx-autodoc-typehints==2.5.0
```

### 问题 2: 构建警告

**警告**: `WARNING: document isn't included in any toctree`

**解决方案**: 确保所有文档都在 `index.rst` 的 `toctree` 中被引用。

### 问题 3: 主题未找到

**错误**: `Theme 'furo' not found`

**解决方案**:

```bash
pip install furo
```

### 问题 4: 图表不显示

**问题**: Mermaid 图表不显示

**解决方案**:

1. 确保安装了 `sphinxcontrib-mermaid`
2. 在 `conf.py` 中添加扩展：

```python
extensions = [
    'sphinxcontrib.mermaid',
    # ... 其他扩展
]
```

### 问题 5: API 文档未生成

**问题**: API 参考文档缺少内容

**解决方案**:

```bash
# 生成 API 文档
cd docs
sphinx-apidoc -o source/api ../surfacia

# 重新构建
sphinx-build -b html source _build/html
```

---

## 📝 更新文档

### 更新现有文档

1. 编辑 RST 文件
2. 本地构建预览
3. 测试链接和格式
4. 提交并推送

### 添加新文档

1. 在 `docs/source/` 下创建新的 RST 文件
2. 在 `index.rst` 中添加引用
3. 更新相关文档的交叉引用
4. 本地构建验证
5. 提交并推送

### 更新 API 文档

```bash
# 删除旧的 API 文档
rm -rf source/api/*.rst

# 重新生成 API 文档
sphinx-apidoc -o source/api ../surfacia --force

# 构建文档
sphinx-build -b html source _build/html
```

### 更新依赖

```bash
# 更新 requirements.txt
cd docs
pip install --upgrade sphinx sphinx-rtd-theme furo sphinx-copybutton

# 更新 requirements.txt
pip freeze > requirements.txt

# 测试构建
sphinx-build -b html source _build/html
```

---

## 📊 文档结构

```
docs/
├── source/
│   ├── conf.py                 # Sphinx 配置
│   ├── index.rst               # 主页
│   ├── getting_started/        # 入门指南
│   │   ├── introduction.rst
│   │   ├── installation.rst
│   │   └── quick_start.rst
│   ├── descriptors/            # 描述符文档
│   │   ├── index.rst
│   │   ├── size_and_shape.rst
│   │   ├── electronic_properties.rst
│   │   └── mqsa_modes.rst
│   ├── api/                    # API 参考
│   │   ├── index.rst
│   │   ├── core.rst
│   │   ├── ml.rst
│   │   ├── visualization.rst
│   │   └── utils.rst
│   ├── tutorials/              # 教程
│   │   ├── index.rst
│   │   └── basic_workflow.rst
│   ├── _static/                # 静态文件
│   │   ├── css/
│   │   └── images/
│   └── citation.rst            # 引用信息
├── requirements.txt            # 文档依赖
└── README.md                   # 本文件
```

---

## 🎨 自定义文档样式

### 修改颜色主题

编辑 `docs/source/_static/css/custom.css`:

```css
:root {
  --primary-color: #2563eb;
  --secondary-color: #64748b;
  --background-color: #ffffff;
}
```

### 修改字体

在 `conf.py` 中:

```python
html_theme_options = {
    'fonts': {
        'family': 'sans-serif',
        'language': 'en'
    }
}
```

### 添加 Logo

1. 将 Logo 文件放在 `docs/source/_static/images/`
2. 在 `index.rst` 中引用:

```rst
.. image:: _static/images/logo.png
   :alt: Surfacia Logo
   :class: main-logo
```

---

## 📚 有用的资源

- [Sphinx 官方文档](https://www.sphinx-doc.org/)
- [reStructuredText 语法](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Read the Docs 文档](https://docs.readthedocs.io/)
- [Furo 主题文档](https://pradyunsg.me/furo/)
- [MyST 语法](https://myst-parser.readthedocs.io/)

---

## 🆘 获取帮助

如果遇到问题：

1. 查看 [Sphinx 故障排查](https://www.sphinx-doc.org/en/master/usage/advanced.html#troubleshooting)
2. 搜索 [Read the Docs 论坛](https://forum.readthedocs.io/)
3. 查阅 GitHub Issues: https://github.com/sym823808458/Surfacia/issues

---

## ✅ 检查清单

部署前请确认：

- [ ] 所有文档构建无错误
- [ ] 所有链接可点击
- [ ] 图片正确显示
- [ ] API 文档是最新的
- [ ] 示例代码可以运行
- [ ] 拼写和语法正确
- [ ] 代码注释准确

---

**最后更新**: 2025-12-23
**维护者**: Surfacia Team
