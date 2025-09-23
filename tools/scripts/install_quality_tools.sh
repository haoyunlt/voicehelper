#!/bin/bash

# VoiceHelper 代码质量工具安装脚本
# 一键安装所有代码质量检查工具

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查工具是否存在
check_tool() {
    local tool=$1
    if command -v "$tool" &> /dev/null; then
        return 0
    fi
    return 1
}

log_info "开始安装代码质量工具..."

# 检查系统类型
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    log_info "检测到 macOS 系统"
    
    # 检查 Homebrew
    if ! check_tool "brew"; then
        log_error "未安装 Homebrew，请先安装 Homebrew"
        exit 1
    fi
    
    # 安装基础工具
    log_info "安装基础工具..."
    brew install cloc fdupes
    
    # 安装 Go 工具
    log_info "安装 Go 代码质量工具..."
    brew install golangci-lint
    go install github.com/securego/gosec/v2/cmd/gosec@latest
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    log_info "检测到 Linux 系统"
    
    # 检查包管理器
    if check_tool "apt-get"; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y cloc fdupes
        
        # 安装 golangci-lint
        curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin latest
        
    elif check_tool "yum"; then
        # CentOS/RHEL
        sudo yum install -y cloc fdupes
        
        # 安装 golangci-lint
        curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin latest
        
    else
        log_warn "未识别的 Linux 发行版，请手动安装工具"
    fi
    
    # 安装 gosec
    go install github.com/securego/gosec/v2/cmd/gosec@latest
    
else
    log_warn "未识别的操作系统: $OSTYPE"
fi

# 安装 Python 工具
log_info "安装 Python 代码质量工具..."

# 检查虚拟环境
VENV_PATH=""
if [ -d "voice_venv" ]; then
    VENV_PATH="voice_venv"
elif [ -d "venv" ]; then
    VENV_PATH="venv"
elif [ -d ".venv" ]; then
    VENV_PATH=".venv"
fi

if [ -n "$VENV_PATH" ]; then
    log_info "使用虚拟环境: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    log_warn "未找到虚拟环境，使用全局 Python"
fi

# 安装 Python 工具
pip install --upgrade pip
pip install flake8 pylint bandit mypy safety pytest-cov

# 安装 Node.js 工具（如果存在 Node.js）
if check_tool "npm"; then
    log_info "安装 Node.js 代码质量工具..."
    npm install -g eslint prettier typescript
fi

# 验证安装
log_info "验证工具安装..."

echo -e "\n${GREEN}=== 工具版本信息 ===${NC}"

# 验证基础工具
if check_tool "cloc"; then
    echo "✅ cloc: $(cloc --version)"
else
    echo "❌ cloc: 未安装"
fi

if check_tool "fdupes"; then
    echo "✅ fdupes: $(fdupes --version)"
else
    echo "❌ fdupes: 未安装"
fi

# 验证 Go 工具
if check_tool "golangci-lint"; then
    echo "✅ golangci-lint: $(golangci-lint --version | head -1)"
else
    echo "❌ golangci-lint: 未安装"
fi

if check_tool "gosec"; then
    echo "✅ gosec: $(gosec --version | head -1)"
else
    echo "❌ gosec: 未安装"
fi

# 验证 Python 工具
if check_tool "flake8"; then
    echo "✅ flake8: $(flake8 --version | head -1)"
else
    echo "❌ flake8: 未安装"
fi

if check_tool "pylint"; then
    echo "✅ pylint: $(pylint --version | head -1)"
else
    echo "❌ pylint: 未安装"
fi

if check_tool "bandit"; then
    echo "✅ bandit: $(bandit --version | head -1)"
else
    echo "❌ bandit: 未安装"
fi

if check_tool "mypy"; then
    echo "✅ mypy: $(mypy --version)"
else
    echo "❌ mypy: 未安装"
fi

if check_tool "safety"; then
    echo "✅ safety: $(safety --version)"
else
    echo "❌ safety: 未安装"
fi

# 验证 Node.js 工具
if check_tool "eslint"; then
    echo "✅ eslint: $(eslint --version)"
else
    echo "⚠️  eslint: 未安装 (可选)"
fi

if check_tool "prettier"; then
    echo "✅ prettier: $(prettier --version)"
else
    echo "⚠️  prettier: 未安装 (可选)"
fi

log_success "代码质量工具安装完成！"
log_info "现在可以运行 ./tools/scripts/daily_quality_check.sh 进行质量检查"

echo -e "\n${BLUE}配置文件已创建：${NC}"
echo "- .golangci.yml (Go 代码检查配置)"
echo "- setup.cfg (Python 工具配置)"

echo -e "\n${BLUE}建议的下一步：${NC}"
echo "1. 运行质量检查: ./tools/scripts/daily_quality_check.sh"
echo "2. 设置 CI/CD 自动化质量检查"
echo "3. 配置 IDE 集成这些工具"
echo "4. 定期运行质量检查（建议每日）"
