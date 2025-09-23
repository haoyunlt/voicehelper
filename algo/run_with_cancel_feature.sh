#!/bin/bash

# VoiceHelper 算法服务启动脚本（包含取消聊天功能）
# 使用方法: ./run_with_cancel_feature.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_info "启动 VoiceHelper 算法服务（包含聊天取消功能）"
print_info "工作目录: $SCRIPT_DIR"

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    print_error "Python3 未安装或不在 PATH 中"
    exit 1
fi

# 检查必要文件
if [ ! -f "app/v2_api.py" ]; then
    print_error "找不到 app/v2_api.py 文件"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export PORT="${PORT:-8070}"

print_info "设置环境变量:"
print_info "  PYTHONPATH=$PYTHONPATH"
print_info "  PORT=$PORT"

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "端口 $PORT 已被占用，尝试终止占用进程..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 创建日志目录
mkdir -p logs

print_info "启动服务..."
print_info "服务地址: http://localhost:$PORT"
print_info "健康检查: http://localhost:$PORT/api/v1/health"
print_info "API文档: http://localhost:$PORT/docs"

# 启动服务
python3 -m uvicorn app.v2_api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --log-level info \
    --access-log \
    2>&1 | tee logs/service.log &

SERVICE_PID=$!

# 等待服务启动
print_info "等待服务启动..."
sleep 3

# 检查服务是否启动成功
if kill -0 $SERVICE_PID 2>/dev/null; then
    print_success "服务启动成功 (PID: $SERVICE_PID)"
    
    # 进行健康检查
    print_info "进行健康检查..."
    if curl -s "http://localhost:$PORT/api/v1/health" > /dev/null; then
        print_success "健康检查通过"
        
        # 显示服务信息
        echo
        print_info "=== 服务信息 ==="
        curl -s "http://localhost:$PORT/api/v1/health" | python3 -m json.tool
        
        echo
        print_info "=== 可用的API端点 ==="
        echo "  POST /api/v1/chat/stream     - 流式聊天"
        echo "  POST /api/v1/chat/cancel     - 取消聊天"
        echo "  GET  /api/v1/chat/session/{id} - 获取会话状态"
        echo "  GET  /api/v1/chat/sessions   - 列出所有会话"
        echo "  GET  /api/v1/health          - 健康检查"
        echo "  WS   /api/v1/voice/stream    - 语音WebSocket"
        
        echo
        print_info "=== 测试命令 ==="
        echo "  # 运行功能测试"
        echo "  python3 test_cancel_chat.py"
        echo
        echo "  # 运行使用示例"
        echo "  python3 examples/cancel_chat_example.py"
        echo
        echo "  # 手动测试取消功能"
        echo "  curl -X POST http://localhost:$PORT/api/v1/chat/cancel -H 'Content-Type: application/json' -d '{\"session_id\":\"test_session\"}'"
        
    else
        print_error "健康检查失败"
        kill $SERVICE_PID 2>/dev/null || true
        exit 1
    fi
else
    print_error "服务启动失败"
    exit 1
fi

# 设置信号处理
cleanup() {
    print_info "正在关闭服务..."
    kill $SERVICE_PID 2>/dev/null || true
    wait $SERVICE_PID 2>/dev/null || true
    print_success "服务已关闭"
    exit 0
}

trap cleanup SIGINT SIGTERM

print_info "服务正在运行，按 Ctrl+C 停止"
print_info "日志文件: logs/service.log"

# 等待服务进程
wait $SERVICE_PID
