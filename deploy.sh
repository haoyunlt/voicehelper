#!/bin/bash

# VoiceHelper AI - Docker Compose 部署脚本
# 版本: 2.0.0
# 作者: VoiceHelper Team

set -euo pipefail

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="voicehelper"
DEFAULT_ENV="dev"
DEFAULT_PROFILE="all"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# 显示横幅
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                    VoiceHelper AI                            ║
║                Docker Compose 部署工具                       ║
║                     版本: 2.0.0                             ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
VoiceHelper AI Docker Compose 部署工具

用法:
    $0 [选项] <命令>

命令:
    up          启动所有服务
    down        停止所有服务
    restart     重启所有服务
    status      显示服务状态
    logs        显示服务日志
    build       构建所有镜像
    pull        拉取最新镜像
    clean       清理未使用的资源
    backup      备份数据
    restore     恢复数据
    health      健康检查
    scale       扩缩容服务

选项:
    -e, --env ENV           环境 (dev|prod|local) [默认: dev]
    -p, --profile PROFILE   服务配置 (all|core|monitoring|tools) [默认: all]
    -s, --service SERVICE   指定服务名称
    -f, --force            强制执行操作
    -v, --verbose          详细输出
    -d, --detach           后台运行
    -h, --help             显示帮助信息

环境说明:
    dev     - 开发环境，启用热重载和调试功能
    prod    - 生产环境，优化性能和安全配置
    local   - 本地环境，使用现有的 docker-compose.local.yml

配置说明:
    all         - 启动所有服务
    core        - 仅启动核心服务 (数据库、网关、算法服务)
    monitoring  - 仅启动监控服务
    tools       - 仅启动开发工具
    dify        - 启动Dify AI平台服务
    dify-tools  - 启动Dify管理工具

示例:
    # 启动开发环境
    $0 -e dev up

    # 启动生产环境核心服务
    $0 -e prod -p core up

    # 启动Dify AI平台
    $0 -p dify up

    # 查看特定服务日志
    $0 -s gateway logs

    # 扩容算法服务到3个实例
    $0 -e prod scale algo-service=3

    # 备份数据
    $0 backup

    # 健康检查
    $0 health
EOF
}

# 解析命令行参数
parse_args() {
    ENV="$DEFAULT_ENV"
    PROFILE="$DEFAULT_PROFILE"
    SERVICE=""
    FORCE=false
    VERBOSE=false
    DETACH=false
    COMMAND=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENV="$2"
                shift 2
                ;;
            -p|--profile)
                PROFILE="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                DEBUG=true
                shift
                ;;
            -d|--detach)
                DETACH=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            up|down|restart|status|logs|build|pull|clean|backup|restore|health|scale|check)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 处理剩余参数
    EXTRA_ARGS=("$@")
}

# 验证环境
validate_environment() {
    case $ENV in
        dev|prod|local)
            ;;
        *)
            log_error "无效的环境: $ENV"
            log_error "支持的环境: dev, prod, local"
            exit 1
            ;;
    esac

    case $PROFILE in
        all|core|monitoring|tools|dify|dify-tools)
            ;;
        *)
            log_error "无效的配置: $PROFILE"
            log_error "支持的配置: all, core, monitoring, tools, dify, dify-tools"
            exit 1
            ;;
    esac
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖工具..."
    
    local missing_tools=()
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_tools+=("docker-compose")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "缺少以下工具: ${missing_tools[*]}"
        log_error "请安装 Docker 和 Docker Compose"
        exit 1
    fi
    
    # 检查 Docker 是否运行
    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行，请启动 Docker"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 获取 Docker Compose 文件
get_compose_files() {
    local files=("-f" "docker-compose.yml")
    
    case $ENV in
        dev)
            files+=("-f" "docker-compose.dev.yml")
            ;;
        prod)
            files+=("-f" "docker-compose.prod.yml")
            ;;
        local)
            files=("-f" "docker-compose.local.yml")
            ;;
    esac
    
    # 如果配置包含Dify，添加Dify compose文件
    case $PROFILE in
        dify|dify-tools|all)
            files+=("-f" "docker-compose.dify.yml")
            ;;
    esac
    
    echo "${files[@]}"
}

# 获取 Docker Compose 命令
get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "docker compose"
    fi
}

# 获取服务配置
get_profile_services() {
    case $PROFILE in
        core)
            echo "postgres redis milvus-etcd milvus-minio milvus nats gateway algo-service voice-service"
            ;;
        monitoring)
            echo "prometheus grafana jaeger"
            ;;
        tools)
            echo "pgadmin redis-commander attu mailhog swagger-ui"
            ;;
        dify)
            echo "dify-postgres dify-redis dify-weaviate dify-sandbox dify-api dify-worker dify-web dify-adapter"
            ;;
        dify-tools)
            echo "dify-pgadmin dify-redis-commander"
            ;;
        all)
            echo ""  # 空字符串表示所有服务
            ;;
    esac
}

# 执行 Docker Compose 命令
run_compose() {
    local cmd="$1"
    shift
    
    local compose_cmd
    compose_cmd=$(get_compose_cmd)
    
    local compose_files
    compose_files=($(get_compose_files))
    
    local profile_services
    profile_services=$(get_profile_services)
    
    local full_cmd=("$compose_cmd" "${compose_files[@]}" "-p" "$PROJECT_NAME")
    
    # 添加服务过滤
    if [[ -n "$SERVICE" ]]; then
        full_cmd+=("$cmd" "$SERVICE" "$@")
    elif [[ -n "$profile_services" ]]; then
        full_cmd+=("$cmd" $profile_services "$@")
    else
        full_cmd+=("$cmd" "$@")
    fi
    
    log_debug "执行命令: ${full_cmd[*]}"
    
    if [[ "$VERBOSE" == "true" ]]; then
        "${full_cmd[@]}"
    else
        "${full_cmd[@]}" 2>/dev/null || {
            log_error "命令执行失败: ${full_cmd[*]}"
            return 1
        }
    fi
}

# 启动服务
cmd_up() {
    log_info "启动 VoiceHelper AI 服务..."
    log_info "环境: $ENV, 配置: $PROFILE"
    
    local args=()
    if [[ "$DETACH" == "true" ]]; then
        args+=("-d")
    fi
    
    # 确保环境文件存在
    if [[ ! -f ".env" ]]; then
        if [[ -f "env.unified" ]]; then
            log_info "复制环境配置文件..."
            cp env.unified .env
        else
            log_warning "未找到环境配置文件，使用默认配置"
        fi
    fi
    
    run_compose "up" "${args[@]}" "${EXTRA_ARGS[@]}"
    
    if [[ "$DETACH" == "true" ]]; then
        log_success "服务已在后台启动"
        sleep 5
        cmd_status
    else
        log_success "服务启动完成"
    fi
}

# 停止服务
cmd_down() {
    log_info "停止 VoiceHelper AI 服务..."
    
    local args=()
    if [[ "$FORCE" == "true" ]]; then
        args+=("--remove-orphans" "-v")
    fi
    
    run_compose "down" "${args[@]}" "${EXTRA_ARGS[@]}"
    log_success "服务已停止"
}

# 重启服务
cmd_restart() {
    log_info "重启 VoiceHelper AI 服务..."
    cmd_down
    sleep 2
    cmd_up
}

# 显示服务状态
cmd_status() {
    log_info "VoiceHelper AI 服务状态:"
    run_compose "ps" "${EXTRA_ARGS[@]}"
}

# 显示日志
cmd_logs() {
    log_info "显示服务日志..."
    
    local args=("-f")
    if [[ -n "$SERVICE" ]]; then
        args+=("--tail=100")
    fi
    
    run_compose "logs" "${args[@]}" "${EXTRA_ARGS[@]}"
}

# 构建镜像
cmd_build() {
    log_info "构建 VoiceHelper AI 镜像..."
    
    local args=("--no-cache")
    if [[ "$FORCE" == "true" ]]; then
        args+=("--pull")
    fi
    
    run_compose "build" "${args[@]}" "${EXTRA_ARGS[@]}"
    log_success "镜像构建完成"
}

# 拉取镜像
cmd_pull() {
    log_info "拉取最新镜像..."
    run_compose "pull" "${EXTRA_ARGS[@]}"
    log_success "镜像拉取完成"
}

# 清理资源
cmd_clean() {
    log_info "清理未使用的 Docker 资源..."
    
    if [[ "$FORCE" == "true" ]] || confirm "确定要清理未使用的资源吗？"; then
        docker system prune -f
        docker volume prune -f
        docker network prune -f
        log_success "资源清理完成"
    else
        log_info "操作已取消"
    fi
}

# 备份数据
cmd_backup() {
    log_info "备份 VoiceHelper AI 数据..."
    
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # 备份 PostgreSQL
    log_info "备份 PostgreSQL 数据..."
    run_compose "exec" "-T" "postgres" "pg_dumpall" "-U" "voicehelper" > "$backup_dir/postgres.sql"
    
    # 备份 Redis
    log_info "备份 Redis 数据..."
    run_compose "exec" "-T" "redis" "redis-cli" "--rdb" "/tmp/dump.rdb"
    docker cp "${PROJECT_NAME}_redis_1:/tmp/dump.rdb" "$backup_dir/redis.rdb"
    
    # 备份配置文件
    log_info "备份配置文件..."
    cp -r ./tools/deployment/config "$backup_dir/"
    cp .env "$backup_dir/" 2>/dev/null || true
    
    log_success "数据备份完成: $backup_dir"
}

# 恢复数据
cmd_restore() {
    local backup_dir="${EXTRA_ARGS[0]:-}"
    
    if [[ -z "$backup_dir" ]] || [[ ! -d "$backup_dir" ]]; then
        log_error "请指定有效的备份目录"
        exit 1
    fi
    
    log_info "从 $backup_dir 恢复数据..."
    
    if [[ "$FORCE" == "true" ]] || confirm "确定要恢复数据吗？这将覆盖现有数据。"; then
        # 恢复 PostgreSQL
        if [[ -f "$backup_dir/postgres.sql" ]]; then
            log_info "恢复 PostgreSQL 数据..."
            run_compose "exec" "-T" "postgres" "psql" "-U" "voicehelper" < "$backup_dir/postgres.sql"
        fi
        
        # 恢复 Redis
        if [[ -f "$backup_dir/redis.rdb" ]]; then
            log_info "恢复 Redis 数据..."
            docker cp "$backup_dir/redis.rdb" "${PROJECT_NAME}_redis_1:/tmp/dump.rdb"
            run_compose "exec" "redis" "redis-cli" "DEBUG" "RELOAD"
        fi
        
        log_success "数据恢复完成"
    else
        log_info "操作已取消"
    fi
}

# 健康检查
cmd_health() {
    log_info "执行健康检查..."
    
    local services=("postgres" "redis" "gateway" "algo-service")
    local failed_services=()
    
    for service in "${services[@]}"; do
        log_info "检查 $service..."
        if run_compose "exec" "-T" "$service" "sh" "-c" "exit 0" 2>/dev/null; then
            log_success "$service 运行正常"
        else
            log_error "$service 运行异常"
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "所有服务运行正常"
    else
        log_error "以下服务运行异常: ${failed_services[*]}"
        exit 1
    fi
}

# 扩缩容服务
cmd_scale() {
    local scale_args="${EXTRA_ARGS[0]:-}"
    
    if [[ -z "$scale_args" ]]; then
        log_error "请指定扩缩容参数，格式: service=replicas"
        log_error "示例: $0 scale algo-service=3"
        exit 1
    fi
    
    log_info "扩缩容服务: $scale_args"
    run_compose "up" "-d" "--scale" "$scale_args"
    log_success "扩缩容完成"
}

# 环境检查
cmd_check() {
    log_info "检查部署环境..."
    
    # 检查 Docker 版本
    log_info "Docker 版本:"
    docker --version || {
        log_error "Docker 未安装或无法访问"
        exit 1
    }
    
    # 检查 Docker Compose 版本
    log_info "Docker Compose 版本:"
    if command -v docker-compose &> /dev/null; then
        docker-compose --version
    elif docker compose version &> /dev/null; then
        docker compose version
    else
        log_error "Docker Compose 未安装"
        exit 1
    fi
    
    # 检查 Docker 服务状态
    if docker info &> /dev/null; then
        log_success "Docker 服务运行正常"
    else
        log_error "Docker 服务未运行"
        exit 1
    fi
    
    # 检查环境配置文件
    if [[ -f ".env" ]]; then
        log_success ".env 配置文件存在"
    elif [[ -f "env.unified" ]]; then
        log_warning ".env 文件不存在，但找到 env.unified"
        log_info "可以运行 'cp env.unified .env' 创建配置文件"
    else
        log_error "未找到环境配置文件"
        exit 1
    fi
    
    # 检查系统资源
    log_info "系统资源检查:"
    
    # 检查内存
    if command -v free &> /dev/null; then
        local available_memory
        available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}' 2>/dev/null || echo "0")
        log_info "可用内存: ${available_memory}MB"
        if [[ "$available_memory" -lt 2048 ]]; then
            log_warning "可用内存不足 2GB，可能影响性能"
        fi
    fi
    
    # 检查磁盘空间
    local available_disk
    available_disk=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    log_info "可用磁盘空间: ${available_disk}GB"
    if [[ "$available_disk" -lt 10 ]]; then
        log_warning "可用磁盘空间不足 10GB"
    fi
    
    # 检查端口占用
    log_info "检查端口占用:"
    local ports=(3000 8080 8000 8001 5001 5432 6379 19530 3001 5433 6380 8194 8200)
    local occupied_ports=()
    
    for port in "${ports[@]}"; do
        if lsof -i ":$port" &> /dev/null || netstat -tuln 2>/dev/null | grep ":$port " &> /dev/null; then
            occupied_ports+=("$port")
        fi
    done
    
    if [[ ${#occupied_ports[@]} -gt 0 ]]; then
        log_warning "以下端口已被占用: ${occupied_ports[*]}"
        log_warning "可能需要停止相关服务或修改端口配置"
    else
        log_success "所有必要端口都可用"
    fi
    
    # 检查网络连接
    log_info "检查网络连接:"
    if curl -s --connect-timeout 5 https://registry-1.docker.io/v2/ &> /dev/null; then
        log_success "Docker Hub 连接正常"
    else
        log_warning "无法连接到 Docker Hub，可能影响镜像拉取"
    fi
    
    log_success "环境检查完成"
}

# 确认对话框
confirm() {
    local message="$1"
    echo -n -e "${YELLOW}$message [y/N]: ${NC}"
    read -r response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# 主函数
main() {
    show_banner
    
    parse_args "$@"
    validate_environment
    check_dependencies
    
    case "$COMMAND" in
        up)
            cmd_up
            ;;
        down)
            cmd_down
            ;;
        restart)
            cmd_restart
            ;;
        status)
            cmd_status
            ;;
        logs)
            cmd_logs
            ;;
        build)
            cmd_build
            ;;
        pull)
            cmd_pull
            ;;
        clean)
            cmd_clean
            ;;
        backup)
            cmd_backup
            ;;
        restore)
            cmd_restore
            ;;
        health)
            cmd_health
            ;;
        scale)
            cmd_scale
            ;;
        check)
            cmd_check
            ;;
        "")
            log_error "请指定命令"
            show_help
            exit 1
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# 错误处理
trap 'log_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"
