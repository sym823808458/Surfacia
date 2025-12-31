# Read the Docs 部署指南

## ✅ 已完成的工作

您的文档已成功推送到 GitHub！所有文件位置：
- **GitHub 仓库**: https://github.com/sym823808458/Surfacia
- **本地路径**: `C:\Users\YumingSu\Sym_Python_codes\Surfacia`

## 📋 已创建的文档文件清单

### 部署相关文件（4个）
- `.readthedocs.yml` - Read the Docs 自动构建配置
- `DEPLOYMENT_GUIDE.md` - 详细部署指南
- `QUICK_DEPLOY_GUIDE.md` - 快速部署指南
- `README_DEPLOYMENT.md` - 部署说明

### API 文档（8个）
- `docs/source/api/commands.rst` - 命令行工具 API
- `docs/source/api/core.rst` - 核心模块 API
- `docs/source/api/descriptors.rst` - 描述符计算 API
- `docs/source/api/ml.rst` - 机器学习模块 API
- `docs/source/api/shap_viz.rst` - SHAP 可视化 API
- `docs/source/api/utils.rst` - 工具函数 API
- `docs/source/api/visualization.rst` - 可视化 API
- `docs/source/api/workflow.rst` - 工作流程 API

### 其他文档（2个）
- `docs/source/descriptors/mqsa_modes.rst` - MQSA 模式
- `docs/source/tutorials/basic_workflow.rst` - 基础教程

## 🚀 在 Read the Docs 上部署文档

### 步骤 1：注册/登录 Read the Docs

1. 访问 https://readthedocs.org
2. 点击右上角的 **"Sign Up"** 注册或 **"Log in"** 登录
3. 推荐使用 GitHub 账号登录（点击 "Sign up with GitHub"）

### 步骤 2：导入 GitHub 仓库

1. 登录后，点击 **"Import a Project"** 按钮
2. 在 **"Import a Project from a Git Repository"** 页面：
   - **Name**: Surfacia
   - **Repository**: https://github.com/sym823808458/Surfacia.git
   - **Edit advanced project options**: 展开
3. 在高级选项中：
   - **Documentation Type**: Sphinx
   - **Python version**: 3.x（根据您的项目选择）
   - **Requirements file**: `docs/requirements.txt`
4. 点击 **"Create Project"** 或 **"Create"**

### 步骤 3：配置构建设置

项目创建后，Read the Docs 会自动检测您的配置文件：

**配置文件位置**：`.readthedocs.yml`

**自动配置内容**：
```yaml
version: 2

sphinx:
  configuration: docs/source/conf.py

python:
  version: "3.10"
  install:
    - requirements: docs/requirements.txt
```

### 步骤 4：触发首次构建

1. 项目创建后，Read the Docs 会自动开始构建
2. 构建过程可能需要 2-5 分钟
3. 点击 **"Builds"** 标签页查看构建状态
4. 如果构建失败，点击查看详细日志并修复错误

### 步骤 5：验证文档部署

构建成功后：

1. 访问您的文档主页：`https://surfacia.readthedocs.io/`
   - （实际URL可能是 `https://sym823808458-surfacia.readthedocs.io/`）
2. 检查文档页面是否正常显示
3. 验证以下部分：
   - 首页内容
   - API 文档链接
   - 教程和指南
   - 导航菜单

## 🔧 常见问题排查

### 构建失败怎么办？

1. **查看构建日志**：
   - 在 Read the Docs 网站上，点击 "Builds" 标签
   - 点击最新的构建记录
   - 查看详细错误信息

2. **常见错误及解决方案**：

   **错误：ModuleNotFoundError**
   - 原因：缺少依赖包
   - 解决：检查 `docs/requirements.txt` 是否包含所有必需的包

   **错误：ImportError**
   - 原因：包安装路径问题
   - 解决：在 `.readthedocs.yml` 中添加 `sys.path` 配置

   **错误：配置文件错误**
   - 原因：`docs/source/conf.py` 有语法错误
   - 解决：检查 conf.py 中的配置是否正确

### 如何更新文档？

1. **修改本地文件**：
   ```bash
   cd C:\Users\YumingSu\Sym_Python_codes\Surfacia
   # 编辑文档文件
   ```

2. **提交并推送到 GitHub**：
   ```bash
   git add docs/
   git commit -m "Update documentation"
   git push origin master
   ```

3. **自动构建**：
   - Read the Docs 会自动检测到 GitHub 的更新
   - 几分钟后自动重新构建
   - 您的文档网站会自动更新

### 如何自定义主题？

您的项目已配置了自定义 CSS：
- 文件位置：`docs/source/_static/css/custom.css`
- 主题：默认使用 Sphinx Alabaster 主题

如需更改主题，编辑 `docs/source/conf.py`：

```python
html_theme = 'furo'  # 或其他主题
```

推荐主题：
- `furo` - 现代简洁主题
- `sphinx_rtd_theme` - Read the Docs 默认主题
- `pydata_sphinx_theme` - 数据科学项目常用主题

## 📊 文档结构预览

```
Surfacia Documentation
├── Home (首页)
├── Getting Started (入门指南)
│   ├── Installation (安装)
│   └── Quick Start (快速开始)
├── Tutorials (教程)
│   └── Basic Workflow (基础工作流程)
├── API Reference (API 参考)
│   ├── Core (核心模块)
│   ├── Descriptors (描述符计算)
│   ├── Machine Learning (机器学习)
│   ├── Visualization (可视化)
│   ├── SHAP Visualization (SHAP 可视化)
│   ├── Utils (工具函数)
│   ├── Commands (命令行工具)
│   └── Workflow (工作流程)
├── Descriptors (描述符详解)
│   ├── Overview (概览)
│   ├── Electronic Properties (电子性质)
│   ├── Size and Shape (尺寸和形状)
│   └── MQSA Modes (MQSA 模式)
└── Citation (引用)
```

## 🎯 下一步建议

1. **首次部署**：按照上述步骤在 Read the Docs 上创建项目
2. **测试文档**：确保所有链接正常，内容完整
3. **定期更新**：每次代码更新后，同步更新文档
4. **收集反馈**：从用户那里收集文档使用反馈并改进

## 📞 获取帮助

如果遇到问题，可以：
- 查看 Read the Docs 官方文档：https://docs.readthedocs.io/
- 查看 Sphinx 文档：https://www.sphinx-doc.org/
- 检查项目的 GitHub Issues：https://github.com/sym823808458/Surfacia/issues

---

## ✨ 恭喜！

您的文档已经准备就绪，现在可以在 Read the Docs 上部署了！

**GitHub 仓库**: https://github.com/sym823808458/Surfacia
**文档将在**: https://[your-project-name].readthedocs.io/

祝部署顺利！🎉
