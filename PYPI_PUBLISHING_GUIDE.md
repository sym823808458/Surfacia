# Surfacia PyPI 发布指南

本指南将帮助您将 Surfacia 包发布到 PyPI，让其他用户可以通过 `pip install surfacia` 安装。

## 📋 发布前准备

### 1. 注册 PyPI 账户

#### 正式 PyPI
- 访问 https://pypi.org/account/register/
- 注册账户并验证邮箱

#### 测试 PyPI（推荐先测试）
- 访问 https://test.pypi.org/account/register/
- 注册测试账户

### 2. 安装必要工具

```bash
pip install build twine
```

### 3. 配置 API Token（推荐）

#### 为正式 PyPI 创建 API Token
1. 登录 https://pypi.org/
2. 进入 Account settings → API tokens
3. 创建新的 API token
4. 保存 token（只显示一次）

#### 为测试 PyPI 创建 API Token
1. 登录 https://test.pypi.org/
2. 进入 Account settings → API tokens
3. 创建新的 API token

#### 配置 .pypirc 文件
在用户主目录创建 `.pypirc` 文件：

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

## 🚀 发布流程

### 方法一：使用自动化脚本（推荐）

```bash
# 在 Surfacia 项目根目录运行
python publish_to_pypi.py
```

脚本会自动执行以下步骤：
1. 检查必要工具
2. 验证包结构
3. 清理旧构建文件
4. 构建包
5. 检查包完整性
6. 上传到 PyPI

### 方法二：手动发布

#### 1. 清理旧文件
```bash
rm -rf build/ dist/ *.egg-info/
```

#### 2. 构建包
```bash
python -m build
```

#### 3. 检查包完整性
```bash
twine check dist/*
```

#### 4. 上传到测试 PyPI（推荐先测试）
```bash
twine upload --repository testpypi dist/*
```

#### 5. 测试安装
```bash
pip install -i https://test.pypi.org/simple/ surfacia
```

#### 6. 上传到正式 PyPI
```bash
twine upload dist/*
```

## 📦 包结构检查清单

确保以下文件存在且正确：

- [ ] `setup.py` - 包配置文件
- [ ] `pyproject.toml` - 现代包配置
- [ ] `README.md` - 项目说明
- [ ] `LICENSE` - 许可证文件
- [ ] `requirements.txt` - 依赖列表
- [ ] `MANIFEST.in` - 包含文件清单
- [ ] `surfacia/__init__.py` - 包初始化文件

## 🔧 版本管理

### 更新版本号
需要在以下文件中同步更新版本号：

1. `setup.py` 中的 `version="3.0.1"`
2. `pyproject.toml` 中的 `version = "3.0.1"`
3. `surfacia/__init__.py` 中的 `__version__ = "3.0.1"`

### 版本号规范
遵循语义化版本控制（SemVer）：
- `MAJOR.MINOR.PATCH`
- 例如：`3.0.1` → `3.0.2`（补丁）→ `3.1.0`（新功能）→ `4.0.0`（破坏性更改）

## 🧪 测试发布

### 1. 先发布到测试 PyPI
```bash
twine upload --repository testpypi dist/*
```

### 2. 测试安装
```bash
# 创建新的虚拟环境测试
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# test_env\Scripts\activate  # Windows

# 从测试 PyPI 安装
pip install -i https://test.pypi.org/simple/ surfacia

# 测试基本功能
python -c "import surfacia; print(surfacia.__version__)"
surfacia --help
```

### 3. 确认无误后发布到正式 PyPI

## 📝 发布后检查

### 1. 验证 PyPI 页面
- 访问 https://pypi.org/project/surfacia/
- 检查项目描述、版本、依赖等信息

### 2. 测试安装
```bash
pip install surfacia
```

### 3. 测试功能
```bash
surfacia --help
python -c "import surfacia; print('安装成功!')"
```

## ⚠️ 常见问题

### 1. 包名已存在
- 错误：`The name 'surfacia' is already in use`
- 解决：选择不同的包名或联系现有包的维护者

### 2. 版本号已存在
- 错误：`File already exists`
- 解决：更新版本号后重新构建和上传

### 3. 依赖问题
- 确保 `requirements.txt` 中的所有依赖都可以从 PyPI 安装
- 特别注意 RDKit 等特殊依赖的安装说明

### 4. 文件缺失
- 检查 `MANIFEST.in` 文件
- 确保所有必要文件都被包含

## 🔄 更新发布

### 发布新版本的步骤：
1. 更新代码
2. 更新版本号（3个文件）
3. 更新 `CHANGELOG.md`（如果有）
4. 重新构建和发布

```bash
# 清理
rm -rf build/ dist/ *.egg-info/

# 构建
python -m build

# 检查
twine check dist/*

# 发布
twine upload dist/*
```

## 📊 发布统计

发布成功后，您可以在以下位置查看统计信息：
- PyPI 项目页面：https://pypi.org/project/surfacia/
- 下载统计：https://pypistats.org/packages/surfacia

## 🎯 最佳实践

1. **先测试后发布**：始终先发布到测试 PyPI
2. **版本控制**：使用语义化版本号
3. **文档完整**：确保 README.md 详细且准确
4. **依赖管理**：明确指定依赖版本范围
5. **持续集成**：考虑使用 GitHub Actions 自动化发布流程

## 📞 获取帮助

如果遇到问题，可以：
1. 查看 PyPI 官方文档：https://packaging.python.org/
2. 查看 Twine 文档：https://twine.readthedocs.io/
3. 在项目 GitHub 仓库提交 Issue

---

**祝您发布成功！** 🎉

发布后，用户就可以通过以下命令安装 Surfacia：

```bash
pip install surfacia