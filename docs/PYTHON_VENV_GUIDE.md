# Python模块虚拟环境管理指南

## 概述

本项目按照Python模块划分，每个模块都有独立的虚拟环境和requirements.txt文件，确保依赖隔离和版本管理。

## 模块结构

### 1. algo - AI算法核心模块
- **路径**: `algo/`
- **虚拟环境**: `algo/algo_venv/`
- **依赖文件**: `algo/requirements.txt`
- **功能**: RAG、ASR/TTS、Agent图等AI核心功能

### 2. backend/app - FastAPI后端服务
- **路径**: `backend/app/`
- **虚拟环境**: `backend/app/backend_venv/`
- **依赖文件**: `backend/app/requirements.txt`
- **功能**: Web API服务、WebRTC、实时通信

### 3. platforms/admin - 管理后台
- **路径**: `platforms/admin/`
- **虚拟环境**: `platforms/admin/admin_venv/`
- **依赖文件**: `platforms/admin/requirements.txt`
- **功能**: Flask管理界面、数据分析、用户管理

### 4. shared/sdks/python - Python SDK
- **路径**: `shared/sdks/python/`
- **虚拟环境**: `shared/sdks/python/sdk_venv/`
- **依赖文件**: `shared/sdks/python/requirements.txt`
- **功能**: 客户端SDK、API封装

### 5. tools/scripts - 工具脚本
- **路径**: `tools/scripts/`
- **虚拟环境**: `tools/scripts/tools_venv/`
- **依赖文件**: `tools/scripts/requirements.txt`
- **功能**: 部署脚本、测试工具、监控脚本

## 快速设置

### 设置所有模块虚拟环境
```bash
# 在项目根目录执行
./setup_all_venvs.sh
```

### 设置单个模块虚拟环境
```bash
# 进入模块目录
cd algo
./setup_venv.sh

# 或者
cd backend/app
./setup_venv.sh
```

## 使用方法

### 激活虚拟环境
```bash
# algo模块
source algo/algo_venv/bin/activate

# backend/app模块
source backend/app/backend_venv/bin/activate

# platforms/admin模块
source platforms/admin/admin_venv/bin/activate

# shared/sdks/python模块
source shared/sdks/python/sdk_venv/bin/activate

# tools/scripts模块
source tools/scripts/tools_venv/bin/activate
```

### 停用虚拟环境
```bash
deactivate
```

### 安装新依赖
```bash
# 激活对应模块的虚拟环境
source algo/algo_venv/bin/activate

# 安装依赖
pip install new-package

# 更新requirements.txt
pip freeze > algo/requirements.txt
```

## 依赖管理

### 各模块主要依赖

#### algo模块
- fastapi, uvicorn - Web框架
- langchain, sentence-transformers - AI框架
- faiss-cpu - 向量搜索
- openai, azure-cognitiveservices-speech - AI服务
- soundfile, librosa - 音频处理

#### backend/app模块
- fastapi, uvicorn - Web框架
- sqlalchemy, alembic - 数据库ORM
- redis, asyncpg - 缓存和数据库
- aiortc, websockets - 实时通信
- opentelemetry - 可观测性

#### platforms/admin模块
- Flask, Flask-SQLAlchemy - Web框架
- pandas, plotly - 数据分析和可视化
- psycopg2-binary - PostgreSQL驱动

#### shared/sdks/python模块
- requests, aiohttp - HTTP客户端
- websockets - WebSocket客户端
- pydantic - 数据验证

#### tools/scripts模块
- pytest - 测试框架
- click - 命令行工具
- psutil - 系统监控
- httpx - HTTP客户端

## 开发工作流

### 1. 开发新功能
```bash
# 激活对应模块虚拟环境
source algo/algo_venv/bin/activate

# 开发代码
# ...

# 运行测试
pytest

# 停用虚拟环境
deactivate
```

### 2. 添加新依赖
```bash
# 激活虚拟环境
source algo/algo_venv/bin/activate

# 安装依赖
pip install new-package

# 更新requirements.txt
pip freeze > requirements.txt

# 提交代码
git add requirements.txt
git commit -m "Add new-package dependency"
```

### 3. 部署准备
```bash
# 确保所有模块虚拟环境都是最新的
./setup_all_venvs.sh

# 运行测试
cd tools/scripts
source tools_venv/bin/activate
python run_all_tests.py
```

## 故障排除

### 虚拟环境创建失败
```bash
# 检查Python版本
python3 --version

# 确保有venv模块
python3 -m venv --help

# 手动创建虚拟环境
python3 -m venv algo/algo_venv
```

### 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 清理缓存
pip cache purge

# 重新安装
pip install -r requirements.txt
```

### 模块导入错误
```bash
# 确保激活了正确的虚拟环境
which python
which pip

# 检查已安装的包
pip list

# 重新安装依赖
pip install -r requirements.txt
```

## 最佳实践

1. **依赖隔离**: 每个模块使用独立的虚拟环境，避免依赖冲突
2. **版本锁定**: requirements.txt中指定具体版本号
3. **定期更新**: 定期更新依赖包到最新稳定版本
4. **测试覆盖**: 每次更新依赖后运行完整测试套件
5. **文档同步**: 更新依赖时同步更新相关文档

## 注意事项

- 不要在全局环境安装项目依赖
- 每次切换模块时记得激活对应的虚拟环境
- 提交代码前确保requirements.txt是最新的
- 生产环境部署时使用固定版本号的依赖
