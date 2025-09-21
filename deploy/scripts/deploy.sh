#!/bin/bash

# 智能聊天机器人系统 - 智能部署脚本
# 支持环境检测、增量部署、选择性部署

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 部署模式
DEPLOY_MODE=""
FORCE_DEPLOY=false
SKIP_BUILD=false
VERBOSE=false

# 服务定义
declare -A SERVICE_PORTS=(
    ["postgres"]=5432
    ["redis"]=6379
    ["milvus"]=19530
    ["neo4j"]=7474
    ["prometheus"]=9090
    ["grafana"]=3001
    ["elasticsearch"]=9200
    ["kibana"]=5601
    ["rabbitmq"]=15672
    ["minio"]=9000
)

declare -A SERVICE_CONTAINERS=(
    ["postgres"]="chatbot-postgres"
    ["redis"]="chatbot-redis"
    ["milvus"]="milvus-standalone"
    ["neo4j"]="chatbot-neo4j"
    ["prometheus"]="chatbot-prometheus"
    ["grafana"]="chatbot-grafana"
    ["elasticsearch"]="chatbot-elasticsearch"
    ["kibana"]="chatbot-kibana"
    ["rabbitmq"]="chatbot-rabbitmq"
    ["minio"]="milvus-minio"
    ["etcd"]="milvus-etcd"
)

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
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

log_section() {
    echo
    echo -e "${MAGENTA}========== $1 ==========${NC}"
    echo
}

# 显示帮助信息
show_help() {
    cat << EOF
智能聊天机器人系统 - 智能部署脚本

用法: $0 [选项]

部署模式:
  --full          完整部署所有服务（基础设施 + 应用）
  --chatbot       仅部署聊天机器人应用服务
  --infra         仅部署基础设施服务
  --service NAME  部署特定服务（如 --service redis）

选项:
  --force         强制重新部署（即使服务已存在）
  --skip-build    跳过镜像构建步骤
  --verbose       显示详细调试信息
  --clean         清理所有服务后退出
  --status        显示服务状态后退出
  --help          显示帮助信息

服务列表:
  基础设施: postgres, redis, milvus, neo4j, prometheus, grafana, 
           elasticsearch, kibana, rabbitmq
  应用服务: gateway, algo, frontend, admin

示例:
  $0 --full                    # 完整部署所有服务
  $0 --chatbot                 # 仅部署应用服务
  $0 --service redis           # 仅部署Redis
  $0 --chatbot --force         # 强制重新部署应用服务
  $0 --status                  # 查看服务状态

EOF
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                DEPLOY_MODE="full"
                shift
                ;;
            --chatbot)
                DEPLOY_MODE="chatbot"
                shift
                ;;
            --infra)
                DEPLOY_MODE="infra"
                shift
                ;;
            --service)
                DEPLOY_MODE="service"
                SERVICE_NAME="$2"
                shift 2
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --clean)
                cleanup
                exit 0
                ;;
            --status)
                show_status
                exit 0
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 如果没有指定模式，默认为智能模式
    if [ -z "$DEPLOY_MODE" ]; then
        DEPLOY_MODE="smart"
    fi
}

# 检查Docker依赖
check_docker() {
    log_debug "检查Docker环境..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        echo "安装指南: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker 守护进程未运行，请启动 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        echo "安装指南: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    log_debug "Docker环境检查通过"
}

# 检查端口是否被占用
check_port() {
    local port=$1
    local service=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# 检查容器是否存在
check_container_exists() {
    local container_name=$1
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

# 检查容器是否运行
check_container_running() {
    local container_name=$1
    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        return 0
    else
        return 1
    fi
}

# 检查服务健康状态
check_service_health() {
    local service=$1
    local container=${SERVICE_CONTAINERS[$service]}
    local port=${SERVICE_PORTS[$service]}
    
    log_debug "检查服务 $service (容器: $container, 端口: $port)"
    
    # 检查容器是否存在
    if ! check_container_exists "$container"; then
        log_debug "$service 容器不存在"
        return 1
    fi
    
    # 检查容器是否运行
    if ! check_container_running "$container"; then
        log_debug "$service 容器未运行"
        return 2
    fi
    
    # 检查端口是否可访问
    if ! nc -z localhost $port 2>/dev/null; then
        log_debug "$service 端口 $port 不可访问"
        return 3
    fi
    
    log_debug "$service 服务健康"
    return 0
}

# 智能检测需要部署的服务
detect_required_services() {
    local required_services=()
    
    log_section "环境检测"
    
    # 基础设施服务检测
    for service in postgres redis milvus neo4j; do
        if ! check_service_health "$service"; then
            log_warning "$service 未部署或未运行，将自动部署"
            required_services+=("$service")
        else
            log_success "$service 已就绪 ✓"
        fi
    done
    
    # 监控服务检测（可选）
    for service in prometheus grafana; do
        if ! check_service_health "$service"; then
            log_info "$service 未部署（可选服务）"
            read -p "是否部署 $service？(y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                required_services+=("$service")
            fi
        else
            log_success "$service 已就绪 ✓"
        fi
    done
    
    echo "${required_services[@]}"
}

# 创建必要的目录和配置文件
setup_environment() {
    log_debug "设置环境..."
    
    # 创建目录结构
    mkdir -p ../local/{config,logs,data}
    mkdir -p ../local/config/{redis,milvus,prometheus,grafana/{provisioning/{datasources,dashboards},dashboards}}
    mkdir -p ../local/config/logstash/{pipeline,config}
    mkdir -p ../local/init-scripts/postgres
    
    # 创建配置文件（如果不存在）
    if [ ! -f "../config/prometheus.yml" ]; then
        create_prometheus_config
    fi
    
    if [ ! -f "../../.env" ] && [ -f "../config/env.local" ]; then
        cp ../config/env.local ../../.env
        log_warning "已创建 .env 文件，请根据需要修改配置"
    fi
    
    # 创建Grafana配置
    create_grafana_config
    
    log_debug "环境设置完成"
}

# 创建Prometheus配置
create_prometheus_config() {
    cat > ../config/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'chatbot-gateway'
    static_configs:
      - targets: ['gateway:8080']
    metrics_path: '/metrics'

  - job_name: 'chatbot-algo'
    static_configs:
      - targets: ['algo:8000']
    metrics_path: '/metrics'
EOF
}

# 创建Grafana配置
create_grafana_config() {
    # 数据源配置
    cat > ../local/config/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # 仪表板配置
    cat > ../local/config/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
}

# 部署单个服务
deploy_service() {
    local service=$1
    local container=${SERVICE_CONTAINERS[$service]}
    
    log_info "部署 $service..."
    
    # 特殊处理Milvus（需要etcd和minio）
    if [ "$service" = "milvus" ]; then
        log_debug "Milvus 需要先启动 etcd 和 minio"
        docker-compose -f ../docker-compose.local.yml up -d etcd minio
        sleep 5
    fi
    
    # 启动服务
    docker-compose -f ../docker-compose.local.yml up -d $service
    
    # 等待服务就绪
    wait_for_service "$service"
    
    log_success "$service 部署完成 ✓"
}

# 等待服务就绪
wait_for_service() {
    local service=$1
    local container=${SERVICE_CONTAINERS[$service]}
    local port=${SERVICE_PORTS[$service]}
    local timeout=60
    
    log_debug "等待 $service 就绪..."
    
    while [ $timeout -gt 0 ]; do
        if check_service_health "$service"; then
            log_debug "$service 已就绪"
            return 0
        fi
        sleep 2
        timeout=$((timeout-2))
    done
    
    log_warning "$service 启动超时，但继续部署"
    return 1
}

# 部署基础设施
deploy_infrastructure() {
    log_section "部署基础设施"
    
    local services=(postgres redis milvus neo4j)
    
    if [ "$DEPLOY_MODE" = "smart" ]; then
        # 智能模式：只部署需要的服务
        for service in "${services[@]}"; do
            if ! check_service_health "$service"; then
                deploy_service "$service"
            else
                log_info "$service 已存在，跳过"
            fi
        done
    else
        # 完整模式：部署所有服务
        for service in "${services[@]}"; do
            if [ "$FORCE_DEPLOY" = true ] || ! check_service_health "$service"; then
                deploy_service "$service"
            else
                log_info "$service 已存在，跳过"
            fi
        done
        
        # 可选服务
        deploy_service "prometheus"
        deploy_service "grafana"
        deploy_service "elasticsearch"
        deploy_service "kibana"
        deploy_service "rabbitmq"
    fi
}

# 构建应用镜像
build_applications() {
    if [ "$SKIP_BUILD" = true ]; then
        log_info "跳过镜像构建"
        return
    fi
    
    log_section "构建应用镜像"
    
    # 检查并构建各个服务
    if [ -f "backend/Dockerfile" ]; then
        log_info "构建 Gateway 镜像..."
        docker build -t chatbot-gateway:latest ./backend
        log_success "Gateway 镜像构建完成"
    fi
    
    if [ -f "algo/Dockerfile" ]; then
        log_info "构建 Algorithm 镜像..."
        docker build -t chatbot-algo:latest ./algo
        log_success "Algorithm 镜像构建完成"
    fi
    
    if [ -f "frontend/Dockerfile" ]; then
        log_info "构建 Frontend 镜像..."
        docker build -t chatbot-frontend:latest ./frontend
        log_success "Frontend 镜像构建完成"
    fi
    
    if [ -f "admin/Dockerfile" ]; then
        log_info "构建 Admin 镜像..."
        docker build -t chatbot-admin:latest ./admin
        log_success "Admin 镜像构建完成"
    fi
}

# 部署应用服务
deploy_applications() {
    log_section "部署应用服务"
    
    # 确保基础设施就绪
    local required_services=(postgres redis milvus)
    for service in "${required_services[@]}"; do
        if ! check_service_health "$service"; then
            log_warning "$service 未就绪，先部署基础服务"
            deploy_service "$service"
        fi
    done
    
    # 构建镜像
    build_applications
    
    # 部署应用
    local app_services=(gateway algo frontend admin)
    for app in "${app_services[@]}"; do
        if docker images | grep -q "chatbot-$app"; then
            log_info "部署 $app..."
            docker-compose -f ../docker-compose.local.yml up -d $app
            log_success "$app 已部署"
        else
            log_warning "$app 镜像不存在，跳过"
        fi
    done
}

# 初始化数据库
init_database() {
    log_info "初始化数据库..."
    
    # 确保PostgreSQL运行
    if ! check_service_health "postgres"; then
        log_error "PostgreSQL 未运行，无法初始化数据库"
        return 1
    fi
    
    # 执行初始化脚本
    if [ -f "../database/schema.sql" ]; then
        log_debug "执行数据库初始化脚本..."
        docker exec -i chatbot-postgres psql -U chatbot -d chatbot < ../database/schema.sql 2>/dev/null || true
        log_success "数据库初始化完成"
    else
        log_debug "数据库初始化脚本不存在"
    fi
}

# 显示服务状态
show_status() {
    log_section "服务状态"
    
    echo "基础设施服务:"
    echo "----------------------------------------"
    printf "%-15s %-15s %-10s\n" "服务" "容器" "状态"
    echo "----------------------------------------"
    
    for service in postgres redis milvus neo4j prometheus grafana elasticsearch kibana rabbitmq; do
        local container=${SERVICE_CONTAINERS[$service]}
        local port=${SERVICE_PORTS[$service]}
        local status="❌ 未部署"
        
        if check_container_running "$container" 2>/dev/null; then
            if nc -z localhost $port 2>/dev/null; then
                status="✅ 运行中"
            else
                status="⚠️  启动中"
            fi
        elif check_container_exists "$container" 2>/dev/null; then
            status="⏸️  已停止"
        fi
        
        printf "%-15s %-15s %-10s\n" "$service" "$container" "$status"
    done
    
    echo
    echo "应用服务:"
    echo "----------------------------------------"
    printf "%-15s %-15s %-10s\n" "服务" "端口" "状态"
    echo "----------------------------------------"
    
    local app_services=(
        "gateway:8080"
        "algo:8000"
        "frontend:3000"
        "admin:5001"
    )
    
    for service_port in "${app_services[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        local status="❌ 未运行"
        
        if nc -z localhost $port 2>/dev/null; then
            status="✅ 运行中"
        fi
        
        printf "%-15s %-15s %-10s\n" "$service" "$port" "$status"
    done
}

# 显示访问信息
show_access_info() {
    log_section "访问信息"
    
    echo "📱 应用服务:"
    if nc -z localhost 3000 2>/dev/null; then
        echo "  前端界面:    ${GREEN}http://localhost:3000${NC}"
    fi
    if nc -z localhost 8080 2>/dev/null; then
        echo "  API网关:     ${GREEN}http://localhost:8080${NC}"
    fi
    if nc -z localhost 8000 2>/dev/null; then
        echo "  算法服务:    ${GREEN}http://localhost:8000/docs${NC}"
    fi
    if nc -z localhost 5001 2>/dev/null; then
        echo "  管理后台:    ${GREEN}http://localhost:5001${NC}"
    fi
    
    echo
    echo "📊 监控服务:"
    if nc -z localhost 9090 2>/dev/null; then
        echo "  Prometheus:  ${GREEN}http://localhost:9090${NC}"
    fi
    if nc -z localhost 3001 2>/dev/null; then
        echo "  Grafana:     ${GREEN}http://localhost:3001${NC} (admin/admin123)"
    fi
    
    echo
    echo "🗄️ 数据服务:"
    if nc -z localhost 5432 2>/dev/null; then
        echo "  PostgreSQL:  ${GREEN}localhost:5432${NC} (chatbot/chatbot123)"
    fi
    if nc -z localhost 6379 2>/dev/null; then
        echo "  Redis:       ${GREEN}localhost:6379${NC} (password: redis123)"
    fi
    if nc -z localhost 19530 2>/dev/null; then
        echo "  Milvus:      ${GREEN}localhost:19530${NC}"
    fi
    if nc -z localhost 7474 2>/dev/null; then
        echo "  Neo4j:       ${GREEN}http://localhost:7474${NC} (neo4j/neo4j123)"
    fi
    
    echo
    echo "💡 常用命令:"
    echo "  查看日志:    docker-compose -f deploy/docker-compose.local.yml logs -f [service]"
    echo "  停止服务:    deploy/scripts/deploy.sh --clean"
    echo "  查看状态:    deploy/scripts/deploy.sh --status"
}

# 清理服务
cleanup() {
    log_section "清理服务"
    
    read -p "确定要停止并删除所有服务吗？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "取消清理"
        return
    fi
    
    log_info "停止所有服务..."
    docker-compose -f ../docker-compose.local.yml down
    
    log_success "清理完成"
}

# 主函数
main() {
    echo "🤖 智能聊天机器人系统 - 智能部署脚本"
    echo "=================================================="
    
    # 检查Docker环境
    check_docker
    
    # 设置环境
    setup_environment
    
    # 根据模式执行部署
    case "$DEPLOY_MODE" in
        "full")
            log_info "执行完整部署..."
            deploy_infrastructure
            deploy_applications
            init_database
            ;;
        "chatbot")
            log_info "仅部署聊天机器人应用..."
            deploy_applications
            init_database
            ;;
        "infra")
            log_info "仅部署基础设施..."
            deploy_infrastructure
            ;;
        "service")
            log_info "部署特定服务: $SERVICE_NAME"
            deploy_service "$SERVICE_NAME"
            ;;
        "smart")
            log_info "智能部署模式..."
            # 检测并部署必要的服务
            deploy_infrastructure
            read -p "是否部署应用服务？(y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                deploy_applications
                init_database
            fi
            ;;
    esac
    
    # 显示状态和访问信息
    show_status
    show_access_info
    
    log_success "🎉 部署完成！"
}

# 解析参数并执行
parse_arguments "$@"
main