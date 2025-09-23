#!/bin/bash

# VoiceHelper Algorithm Service 启动脚本
# 解决Python模块路径问题

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}[INFO]${NC} 启动 VoiceHelper Algorithm Service..."

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

# 设置Python路径
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}[INFO]${NC} PYTHONPATH设置为: $PYTHONPATH"

# 检查虚拟环境（优先使用本地虚拟环境）
if [ -d "$SCRIPT_DIR/algo_venv" ]; then
    echo -e "${GREEN}[INFO]${NC} 激活本地虚拟环境..."
    source "$SCRIPT_DIR/algo_venv/bin/activate"
elif [ -d "../voice_venv" ]; then
    echo -e "${GREEN}[INFO]${NC} 激活共享虚拟环境..."
    source ../voice_venv/bin/activate
else
    echo -e "${YELLOW}[WARNING]${NC} 虚拟环境不存在，使用系统Python"
    echo -e "${BLUE}[HINT]${NC} 运行 ./setup_venv.sh 创建虚拟环境"
fi

# 加载环境变量
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo -e "${GREEN}[INFO]${NC} 加载环境变量..."
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
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

# 创建必要目录
mkdir -p logs data models

# 选择启动模式和文件
if [ -f "app/v2_api.py" ]; then
    MAIN_FILE="app/v2_api.py"
    echo -e "${GREEN}[INFO]${NC} 使用V2 API版本..."
elif [ -f "app/main.py" ]; then
    MAIN_FILE="app/main.py"
    echo -e "${GREEN}[INFO]${NC} 使用主程序版本..."
elif [ "$AI_MODE" = "full" ] && [ -f "app/main_full.py" ]; then
    echo -e "${GREEN}[INFO]${NC} 检查完整AI依赖..."
    python -c "import torch, transformers, sentence_transformers, langchain, faiss" 2>/dev/null && {
        MAIN_FILE="app/main_full.py"
        echo -e "${GREEN}[INFO]${NC} 启动完整AI功能版本..."
    } || {
        MAIN_FILE="app/main_simple.py"
        echo -e "${YELLOW}[WARNING]${NC} 完整AI依赖未安装，回退到简化版本"
        echo -e "${YELLOW}[WARNING]${NC} 要使用完整功能，请运行: pip install -r requirements-full.txt"
    }
elif [ -f "app/main_simple.py" ]; then
    MAIN_FILE="app/main_simple.py"
    echo -e "${GREEN}[INFO]${NC} 启动简化版本..."
else
    echo -e "${RED}[ERROR]${NC} 未找到可启动的主程序文件"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} 启动文件: $MAIN_FILE"
echo -e "${GREEN}[INFO]${NC} 服务地址: http://$HOST:$PORT"

# 启动服务
exec python "$MAIN_FILE"