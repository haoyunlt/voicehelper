#!/bin/bash

# VoiceHelper Algorithm Service 启动脚本
# 解决Python模块路径问题

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}[INFO]${NC} 启动 VoiceHelper Algorithm Service..."

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置Python路径
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo -e "${GREEN}[INFO]${NC} PYTHONPATH设置为: $PYTHONPATH"

# 检查虚拟环境
if [ -d "../voice_venv" ]; then
    echo -e "${GREEN}[INFO]${NC} 激活虚拟环境..."
    source ../voice_venv/bin/activate
else
    echo -e "${YELLOW}[WARNING]${NC} 虚拟环境不存在，使用系统Python"
fi

# 检查依赖
echo -e "${GREEN}[INFO]${NC} 检查Python依赖..."
python -c "import fastapi, uvicorn, pydantic" 2>/dev/null || {
    echo -e "${RED}[ERROR]${NC} 缺少必要依赖，请运行: pip install -r requirements-basic.txt"
    exit 1
}

# 设置环境变量
export SERVICE_NAME=${SERVICE_NAME:-"voicehelper-algo"}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8000"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export ENVIRONMENT=${ENVIRONMENT:-"development"}
export AI_MODE=${AI_MODE:-"simple"}

echo -e "${GREEN}[INFO]${NC} 服务配置:"
echo -e "  - 服务名: $SERVICE_NAME"
echo -e "  - 监听地址: $HOST:$PORT"
echo -e "  - 日志级别: $LOG_LEVEL"
echo -e "  - 环境: $ENVIRONMENT"
echo -e "  - AI模式: $AI_MODE"

# 选择启动模式
if [ "$AI_MODE" = "full" ]; then
    echo -e "${GREEN}[INFO]${NC} 检查完整AI依赖..."
    python -c "import torch, transformers, sentence_transformers, langchain, faiss" 2>/dev/null && {
        echo -e "${GREEN}[INFO]${NC} 启动完整AI功能版本..."
        exec python app/main_full.py
    } || {
        echo -e "${YELLOW}[WARNING]${NC} 完整AI依赖未安装，回退到简化版本"
        echo -e "${YELLOW}[WARNING]${NC} 要使用完整功能，请运行: pip install -r requirements-full.txt"
        exec python app/main_simple.py
    }
else
    echo -e "${GREEN}[INFO]${NC} 启动简化版本..."
    exec python app/main_simple.py
fi