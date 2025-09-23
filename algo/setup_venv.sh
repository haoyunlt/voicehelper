#!/bin/bash

# VoiceHelper 算法服务虚拟环境设置脚本
# 使用方法: ./setup_venv.sh

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
VENV_DIR="$SCRIPT_DIR/algo_venv"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "设置 VoiceHelper 算法服务虚拟环境"
print_info "脚本目录: $SCRIPT_DIR"
print_info "项目根目录: $PROJECT_ROOT"
print_info "虚拟环境目录: $VENV_DIR"

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    print_error "Python3 未安装或不在 PATH 中"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python 版本: $PYTHON_VERSION"

# 删除旧的虚拟环境
if [ -d "$VENV_DIR" ]; then
    print_warning "删除旧的虚拟环境..."
    rm -rf "$VENV_DIR"
fi

# 创建虚拟环境
print_info "创建虚拟环境..."
python3 -m venv "$VENV_DIR"
print_success "虚拟环境创建完成"

# 激活虚拟环境
print_info "激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 升级pip
print_info "升级pip..."
pip install --upgrade pip

# 安装依赖
print_info "安装依赖..."
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt"
elif [ -f "$SCRIPT_DIR/requirements-basic.txt" ]; then
    print_warning "requirements.txt 不存在，使用基础依赖"
    pip install -r "$SCRIPT_DIR/requirements-basic.txt"
else
    print_error "未找到依赖文件"
    exit 1
fi

# 创建环境变量配置文件
print_info "创建环境变量配置..."
cat > "$SCRIPT_DIR/.env" << EOF
# VoiceHelper 算法服务环境变量配置

# Python路径配置
PYTHONPATH=$SCRIPT_DIR:$PROJECT_ROOT

# 算法服务配置
ALGO_SERVICE_HOST=0.0.0.0
ALGO_SERVICE_PORT=8070
ALGO_SERVICE_DEBUG=true

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=$SCRIPT_DIR/logs

# 模型配置
MODEL_CACHE_DIR=$SCRIPT_DIR/models
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
LLM_MODEL=gpt-3.5-turbo

# 数据配置
DATA_DIR=$SCRIPT_DIR/data
FAISS_INDEX_DIR=$SCRIPT_DIR/data/faiss

# API配置
OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here

# 后端服务配置
BACKEND_SERVICE_URL=http://localhost:8080
EOF

# 创建激活脚本
print_info "创建激活脚本..."
cat > "$SCRIPT_DIR/activate.sh" << 'EOF'
#!/bin/bash

# VoiceHelper 算法服务激活脚本
# 使用方法: source ./activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 激活虚拟环境
if [ -f "$SCRIPT_DIR/algo_venv/bin/activate" ]; then
    source "$SCRIPT_DIR/algo_venv/bin/activate"
    echo "✅ 虚拟环境已激活"
else
    echo "❌ 虚拟环境不存在，请先运行 ./setup_venv.sh"
    return 1
fi

# 加载环境变量
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a  # 自动导出变量
    source "$SCRIPT_DIR/.env"
    set +a
    echo "✅ 环境变量已加载"
fi

# 设置PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:$PYTHONPATH"
echo "✅ PYTHONPATH已设置: $PYTHONPATH"

# 创建必要目录
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/models"

echo "🚀 VoiceHelper 算法服务环境已准备就绪！"
echo "📂 工作目录: $SCRIPT_DIR"
echo "🐍 Python路径: $PYTHONPATH"
EOF

chmod +x "$SCRIPT_DIR/activate.sh"

print_success "VoiceHelper 算法服务虚拟环境设置完成！"
print_info "使用方法："
print_info "1. 激活环境: source ./activate.sh"
print_info "2. 启动服务: python app/v2_api.py"
print_info "3. 或使用: ./start.sh"