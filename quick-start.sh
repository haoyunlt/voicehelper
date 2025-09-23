#!/bin/bash

# VoiceHelper AI - 快速启动脚本
# 一键启动完整的 VoiceHelper AI 系统

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示横幅
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                    VoiceHelper AI                            ║
║                   快速启动向导                                ║
║                                                              ║
║  🚀 一键部署完整的多模态AI助手平台                            ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        echo "安装指南: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装"
        exit 1
    fi
    
    # 检查 Docker 是否运行
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行，请启动 Docker"
        exit 1
    fi
    
    # 检查系统资源
    local available_memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}' 2>/dev/null || echo "0")
    if [[ "$available_memory" -lt 4096 ]]; then
        log_warning "可用内存不足 4GB，可能影响性能"
    fi
    
    log_success "系统要求检查通过"
}

# 准备环境配置
prepare_environment() {
    log_info "准备环境配置..."
    
    # 复制环境配置文件
    if [[ ! -f ".env" ]]; then
        if [[ -f "env.unified" ]]; then
            cp env.unified .env
            log_success "已复制环境配置文件"
        else
            log_error "未找到环境配置文件 env.unified"
            exit 1
        fi
    fi
    
    # 检查必要的配置
    if ! grep -q "ARK_API_KEY" .env || grep -q "your-.*-api-key-here" .env; then
        log_warning "检测到默认的 API 密钥配置"
        echo
        echo "为了获得最佳体验，请配置以下 API 密钥："
        echo "1. 豆包 API 密钥 (ARK_API_KEY) - 推荐"
        echo "2. GLM-4 API 密钥 (GLM_API_KEY) - 备用"
        echo
        echo "获取方式："
        echo "- 豆包: https://console.volcengine.com/"
        echo "- GLM-4: https://open.bigmodel.cn/"
        echo
        read -p "是否现在配置 API 密钥？[y/N]: " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            configure_api_keys
        else
            log_warning "跳过 API 密钥配置，将使用默认配置"
        fi
    fi
}

# 配置 API 密钥
configure_api_keys() {
    echo
    log_info "配置 API 密钥..."
    
    # 配置豆包 API
    read -p "请输入豆包 API 密钥 (ARK_API_KEY): " -r ark_key
    if [[ -n "$ark_key" ]]; then
        sed -i.bak "s/ARK_API_KEY=.*/ARK_API_KEY=$ark_key/" .env
        log_success "豆包 API 密钥已配置"
    fi
    
    # 配置 GLM-4 API
    read -p "请输入 GLM-4 API 密钥 (GLM_API_KEY): " -r glm_key
    if [[ -n "$glm_key" ]]; then
        sed -i.bak "s/GLM_API_KEY=.*/GLM_API_KEY=$glm_key/" .env
        log_success "GLM-4 API 密钥已配置"
    fi
    
    # 清理备份文件
    rm -f .env.bak
}

# 选择部署模式
select_deployment_mode() {
    echo
    log_info "选择部署模式："
    echo "1. 🚀 快速体验 (核心服务 + Web界面)"
    echo "2. 🔧 开发模式 (包含开发工具和热重载)"
    echo "3. 🏭 生产模式 (完整功能 + 监控)"
    echo "4. 📊 监控模式 (仅监控服务)"
    echo
    
    while true; do
        read -p "请选择部署模式 [1-4]: " -r choice
        case $choice in
            1)
                DEPLOYMENT_MODE="quick"
                break
                ;;
            2)
                DEPLOYMENT_MODE="dev"
                break
                ;;
            3)
                DEPLOYMENT_MODE="prod"
                break
                ;;
            4)
                DEPLOYMENT_MODE="monitoring"
                break
                ;;
            *)
                log_error "无效选择，请输入 1-4"
                ;;
        esac
    done
    
    log_info "已选择: $DEPLOYMENT_MODE 模式"
}

# 启动服务
start_services() {
    log_info "启动 VoiceHelper AI 服务..."
    
    case $DEPLOYMENT_MODE in
        quick)
            ./deploy.sh -e local -p core up -d
            ;;
        dev)
            ./deploy.sh -e dev -p all up -d
            ;;
        prod)
            ./deploy.sh -e prod -p all up -d
            ;;
        monitoring)
            ./deploy.sh -e dev -p monitoring up -d
            ;;
    esac
    
    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务启动..."
    
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "服务已就绪"
            return 0
        fi
        
        echo -n "."
        sleep 5
        ((attempt++))
    done
    
    log_warning "服务启动超时，请检查日志"
    return 1
}

# 显示访问信息
show_access_info() {
    echo
    log_success "🎉 VoiceHelper AI 部署成功！"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${CYAN}📱 访问地址${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    case $DEPLOYMENT_MODE in
        quick|dev)
            echo "🌐 Web 应用:      http://localhost:3000"
            echo "🔧 API 网关:      http://localhost:8080"
            echo "🤖 算法服务:      http://localhost:8000"
            echo "🎤 语音服务:      http://localhost:8001"
            echo "⚙️  管理后台:      http://localhost:5001"
            ;;
        prod)
            echo "🌐 Web 应用:      http://localhost:80"
            echo "🔧 API 网关:      http://localhost:8080"
            echo "🤖 算法服务:      http://localhost:8000"
            echo "🎤 语音服务:      http://localhost:8001"
            echo "⚙️  管理后台:      http://localhost:5001"
            ;;
    esac
    
    if [[ "$DEPLOYMENT_MODE" != "quick" ]]; then
        echo
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "${CYAN}🛠️  管理工具${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Grafana:       http://localhost:3004 (admin/admin123)"
        echo "📈 Prometheus:    http://localhost:9090"
        echo "🗄️  pgAdmin:       http://localhost:5050 (admin@voicehelper.ai/admin123)"
        echo "🔴 Redis 管理:     http://localhost:8081"
        echo "🔍 向量数据库:     http://localhost:3001"
        echo "📧 邮件测试:      http://localhost:8025"
        echo "📖 API 文档:      http://localhost:8082"
    fi
    
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${CYAN}🔧 管理命令${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "查看状态:         ./deploy.sh status"
    echo "查看日志:         ./deploy.sh logs"
    echo "重启服务:         ./deploy.sh restart"
    echo "停止服务:         ./deploy.sh down"
    echo "健康检查:         ./deploy.sh health"
    echo "备份数据:         ./deploy.sh backup"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}🎯 快速开始${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1. 打开浏览器访问 http://localhost:3000"
    echo "2. 开始与 AI 助手对话"
    echo "3. 尝试语音交互功能"
    echo "4. 查看管理后台了解系统状态"
    echo
}

# 主函数
main() {
    show_banner
    
    # 检查是否在项目根目录
    if [[ ! -f "docker-compose.yml" ]] || [[ ! -f "deploy.sh" ]]; then
        log_error "请在 VoiceHelper 项目根目录下运行此脚本"
        exit 1
    fi
    
    check_requirements
    prepare_environment
    select_deployment_mode
    start_services
    
    if wait_for_services; then
        show_access_info
    else
        echo
        log_error "服务启动可能存在问题，请检查："
        echo "1. 运行 './deploy.sh logs' 查看详细日志"
        echo "2. 运行 './deploy.sh status' 查看服务状态"
        echo "3. 确保端口 3000, 8080, 8000 等未被占用"
    fi
}

# 错误处理
trap 'log_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"
