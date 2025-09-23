#!/bin/bash

# VoiceHelper + Dify 快速启动脚本
# 一键启动VoiceHelper核心服务和Dify AI平台

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示横幅
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║              VoiceHelper + Dify AI 平台                     ║
║                    快速启动脚本                             ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 检查Docker环境
check_docker() {
    log_info "检查Docker环境..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker服务未运行，请启动Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose未安装"
        exit 1
    fi
    
    log_success "Docker环境检查通过"
}

# 准备环境配置
prepare_env() {
    log_info "准备环境配置..."
    
    if [[ ! -f ".env" ]]; then
        if [[ -f "env.unified" ]]; then
            log_info "复制统一配置文件到.env"
            cp env.unified .env
        else
            log_error "未找到环境配置文件"
            exit 1
        fi
    fi
    
    log_success "环境配置准备完成"
}

# 启动服务
start_services() {
    local mode="${1:-full}"
    
    case $mode in
        "core")
            log_info "启动VoiceHelper核心服务..."
            ./deploy.sh -p core up -d
            ;;
        "dify")
            log_info "启动Dify AI平台..."
            ./deploy.sh -p dify up -d
            ;;
        "full")
            log_info "启动VoiceHelper核心服务..."
            ./deploy.sh -p core up -d
            
            log_info "等待核心服务启动..."
            sleep 30
            
            log_info "启动Dify AI平台..."
            ./deploy.sh -p dify up -d
            ;;
        *)
            log_error "未知模式: $mode"
            exit 1
            ;;
    esac
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务启动完成..."
    
    local services=(
        "localhost:8080:/health:VoiceHelper Gateway"
        "localhost:8000:/health:VoiceHelper Algo"
        "localhost:5001:/health:Dify API"
        "localhost:3001:/:Dify Web Console"
        "localhost:8200:/health:Dify Adapter"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r host path name <<< "$service"
        
        log_info "等待 $name 启动..."
        local max_attempts=60
        local attempt=0
        
        while [[ $attempt -lt $max_attempts ]]; do
            if curl -s -f "http://$host$path" &> /dev/null; then
                log_success "$name 已就绪"
                break
            fi
            
            ((attempt++))
            sleep 5
            
            if [[ $attempt -eq $max_attempts ]]; then
                log_warning "$name 启动超时，请检查服务状态"
            fi
        done
    done
}

# 显示服务信息
show_services() {
    log_success "🎉 所有服务启动完成！"
    echo
    echo -e "${CYAN}📋 服务访问地址:${NC}"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ VoiceHelper 服务                                           │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ 🌐 Web应用:        http://localhost:3000                   │"
    echo "│ 🔗 API网关:        http://localhost:8080                   │"
    echo "│ 🤖 算法服务:       http://localhost:8000                   │"
    echo "│ 🎤 语音服务:       http://localhost:8001                   │"
    echo "│ 🛠️  管理后台:       http://localhost:5001                   │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ Dify AI 平台                                               │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ 🎯 Dify控制台:     http://localhost:3001                   │"
    echo "│ 📡 Dify API:       http://localhost:5001                   │"
    echo "│ 🔗 集成适配器:     http://localhost:8200                   │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ 数据库管理                                                  │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ 🗄️  PostgreSQL:    http://localhost:5050                   │"
    echo "│ 🗄️  Dify数据库:    http://localhost:5051                   │"
    echo "│ 💾 Redis:          http://localhost:8081                   │"
    echo "│ 💾 Dify Redis:     http://localhost:8083                   │"
    echo "│ 📊 向量数据库:     http://localhost:3001 (Attu)            │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo
    echo -e "${YELLOW}🔑 默认登录信息:${NC}"
    echo "  Dify控制台: admin@dify.ai / password123"
    echo "  pgAdmin: admin@voicehelper.ai / admin123"
    echo "  Dify pgAdmin: dify-admin@voicehelper.ai / dify123"
    echo
    echo -e "${GREEN}✨ 快速开始:${NC}"
    echo "  1. 访问 Dify控制台 创建应用"
    echo "  2. 获取应用API Key"
    echo "  3. 通过集成适配器调用: curl http://localhost:8200/api/v1/chat"
    echo "  4. 查看服务状态: ./deploy.sh status"
    echo
}

# 显示帮助
show_help() {
    cat << EOF
VoiceHelper + Dify 快速启动脚本

用法:
    $0 [选项] [模式]

模式:
    full    - 启动完整服务 (VoiceHelper + Dify) [默认]
    core    - 仅启动VoiceHelper核心服务
    dify    - 仅启动Dify AI平台

选项:
    -h, --help     显示帮助信息
    -s, --status   显示服务状态
    -d, --down     停止所有服务
    -r, --restart  重启所有服务
    -l, --logs     显示服务日志

示例:
    $0              # 启动完整服务
    $0 core         # 仅启动VoiceHelper
    $0 dify         # 仅启动Dify
    $0 -s           # 显示服务状态
    $0 -d           # 停止所有服务
EOF
}

# 显示服务状态
show_status() {
    log_info "显示服务状态..."
    ./deploy.sh status
}

# 停止服务
stop_services() {
    log_info "停止所有服务..."
    ./deploy.sh down
    log_success "所有服务已停止"
}

# 重启服务
restart_services() {
    log_info "重启所有服务..."
    stop_services
    sleep 5
    start_services "full"
    wait_for_services
    show_services
}

# 显示日志
show_logs() {
    log_info "显示服务日志..."
    ./deploy.sh logs
}

# 主函数
main() {
    show_banner
    
    # 解析参数
    case "${1:-}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--status)
            show_status
            exit 0
            ;;
        -d|--down)
            stop_services
            exit 0
            ;;
        -r|--restart)
            restart_services
            exit 0
            ;;
        -l|--logs)
            show_logs
            exit 0
            ;;
        core|dify|full|"")
            local mode="${1:-full}"
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
    
    # 检查环境
    check_docker
    prepare_env
    
    # 启动服务
    start_services "$mode"
    
    # 等待服务就绪
    wait_for_services
    
    # 显示服务信息
    show_services
}

# 错误处理
trap 'log_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"
