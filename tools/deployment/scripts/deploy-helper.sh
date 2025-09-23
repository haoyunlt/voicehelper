#!/bin/bash

# 智能聊天机器人系统 - 部署辅助脚本
# 提供更多高级功能和便捷操作

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${MAGENTA}========== $1 ==========${NC}\n"; }

# 快速启动特定组合
quick_start() {
    local mode=$1
    
    case "$mode" in
        "dev")
            log_section "开发环境快速启动"
            log_info "启动最小化开发环境..."
            ./deploy.sh --service postgres
            ./deploy.sh --service redis
            log_success "开发环境就绪！"
            echo "现在可以本地启动应用服务进行开发"
            ;;
        "test")
            log_section "测试环境快速启动"
            log_info "启动完整测试环境..."
            ./deploy.sh --infra
            ./deploy.sh --chatbot
            log_success "测试环境就绪！"
            ;;
        "demo")
            log_section "演示环境快速启动"
            log_info "启动演示环境（含监控）..."
            ./deploy.sh --full
            log_success "演示环境就绪！"
            ;;
        *)
            log_error "未知的快速启动模式: $mode"
            echo "可用模式: dev, test, demo"
            exit 1
            ;;
    esac
}

# 健康检查报告
health_report() {
    log_section "系统健康检查报告"
    
    # 检查Docker资源
    echo "📊 Docker资源使用:"
    docker system df
    echo
    
    # 检查容器状态
    echo "📦 容器状态:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Size}}"
    echo
    
    # 检查网络连通性
    echo "🌐 服务连通性测试:"
    services=(
        "PostgreSQL:5432"
        "Redis:6379"
        "Neo4j:7474"
        "Gateway:8080"
        "Algorithm:8000"
        "Frontend:3000"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if nc -z localhost $port 2>/dev/null; then
            echo "  ✅ $name ($port) - 正常"
        else
            echo "  ❌ $name ($port) - 无法连接"
        fi
    done
    echo
    
    # 检查磁盘空间
    echo "💾 磁盘空间:"
    df -h | grep -E "^/|Filesystem"
}

# 备份数据
backup_data() {
    log_section "数据备份"
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # 备份PostgreSQL
    if docker ps | grep -q chatbot-postgres; then
        log_info "备份 PostgreSQL..."
        docker exec chatbot-postgres pg_dump -U chatbot chatbot > "$backup_dir/postgres.sql"
        log_success "PostgreSQL 备份完成"
    fi
    
    # 备份Redis
    if docker ps | grep -q chatbot-redis; then
        log_info "备份 Redis..."
        docker exec chatbot-redis redis-cli --rdb "$backup_dir/redis.rdb" BGSAVE
        sleep 2
        docker cp chatbot-redis:/data/dump.rdb "$backup_dir/redis.rdb"
        log_success "Redis 备份完成"
    fi
    
    # 备份配置文件
    log_info "备份配置文件..."
    cp ../../.env "$backup_dir/.env" 2>/dev/null || true
    cp ../docker-compose.local.yml "$backup_dir/docker-compose.local.yml"
    
    # 压缩备份
    log_info "压缩备份文件..."
    tar -czf "$backup_dir.tar.gz" -C backups "$(basename $backup_dir)"
    rm -rf "$backup_dir"
    
    log_success "备份完成: $backup_dir.tar.gz"
}

# 恢复数据
restore_data() {
    local backup_file=$1
    
    if [ -z "$backup_file" ]; then
        log_error "请指定备份文件"
        echo "用法: $0 restore <backup_file.tar.gz>"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "备份文件不存在: $backup_file"
        exit 1
    fi
    
    log_section "数据恢复"
    
    # 解压备份
    local temp_dir="temp_restore_$(date +%s)"
    mkdir -p "$temp_dir"
    tar -xzf "$backup_file" -C "$temp_dir"
    
    local backup_dir=$(find "$temp_dir" -maxdepth 1 -type d | tail -1)
    
    # 恢复PostgreSQL
    if [ -f "$backup_dir/postgres.sql" ]; then
        log_info "恢复 PostgreSQL..."
        docker exec -i chatbot-postgres psql -U chatbot chatbot < "$backup_dir/postgres.sql"
        log_success "PostgreSQL 恢复完成"
    fi
    
    # 恢复Redis
    if [ -f "$backup_dir/redis.rdb" ]; then
        log_info "恢复 Redis..."
        docker cp "$backup_dir/redis.rdb" chatbot-redis:/data/dump.rdb
        docker restart chatbot-redis
        log_success "Redis 恢复完成"
    fi
    
    # 清理临时文件
    rm -rf "$temp_dir"
    
    log_success "数据恢复完成"
}

# 性能调优建议
performance_tune() {
    log_section "性能调优建议"
    
    echo "🔧 Docker配置优化:"
    echo "  1. 增加Docker内存限制:"
    echo "     Docker Desktop -> Preferences -> Resources"
    echo "     推荐: Memory: 8GB+, CPUs: 4+"
    echo
    
    echo "📊 服务配置优化:"
    echo "  1. PostgreSQL优化:"
    echo "     - shared_buffers = 256MB"
    echo "     - effective_cache_size = 1GB"
    echo "     - work_mem = 4MB"
    echo
    echo "  2. Redis优化:"
    echo "     - maxmemory 512mb"
    echo "     - maxmemory-policy allkeys-lru"
    echo
    echo "     - 增加内存分配"
    echo "     - 调整索引参数"
    echo
    
    echo "💡 应用优化建议:"
    echo "  1. 启用缓存层"
    echo "  2. 使用连接池"
    echo "  3. 异步处理长时间任务"
    echo "  4. 启用压缩传输"
}

# 故障诊断
diagnose() {
    log_section "故障诊断"
    
    # 检查常见问题
    echo "🔍 检查常见问题..."
    
    # 1. 端口冲突
    echo -e "\n1. 端口占用检查:"
    ports=(5432 6379 19530 7474 8080 8000 3000)
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
            process=$(ps -p $pid -o comm=)
            echo "  ⚠️  端口 $port 被占用 (PID: $pid, 进程: $process)"
        fi
    done
    
    # 2. Docker问题
    echo -e "\n2. Docker状态:"
    docker version > /dev/null 2>&1 && echo "  ✅ Docker运行正常" || echo "  ❌ Docker未运行"
    
    # 3. 磁盘空间
    echo -e "\n3. 磁盘空间:"
    available=$(df -h . | awk 'NR==2 {print $4}')
    echo "  可用空间: $available"
    
    # 4. 内存使用
    echo -e "\n4. 内存状态:"
    if command -v free >/dev/null 2>&1; then
        free -h | grep -E "^Mem|^Swap"
    else
        echo "  内存信息不可用（非Linux系统）"
    fi
    
    # 5. 容器日志检查
    echo -e "\n5. 错误日志摘要:"
    for container in $(docker ps -a --format '{{.Names}}' | grep chatbot); do
        errors=$(docker logs $container 2>&1 | grep -i error | tail -3)
        if [ -n "$errors" ]; then
            echo "  $container:"
            echo "$errors" | sed 's/^/    /'
        fi
    done
    
    echo -e "\n💡 建议:"
    echo "  - 如有端口冲突，请停止占用端口的服务或修改配置"
    echo "  - 如磁盘空间不足，请清理Docker缓存: docker system prune"
    echo "  - 查看完整日志: docker logs [容器名]"
}

# 监控仪表板
monitor_dashboard() {
    log_section "监控仪表板"
    
    echo "📊 实时监控面板:"
    echo
    echo "1. Grafana (推荐):"
    echo "   ${GREEN}http://localhost:3001${NC}"
    echo "   用户名: admin"
    echo "   密码: admin123"
    echo
    echo "2. Prometheus:"
    echo "   ${GREEN}http://localhost:9090${NC}"
    echo
    echo "3. 容器资源监控:"
    echo "   运行: docker stats"
    echo
    echo "4. 日志监控:"
    echo "   Kibana: ${GREEN}http://localhost:5601${NC}"
    echo
    
    read -p "是否打开Grafana？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v open >/dev/null 2>&1; then
            open http://localhost:3001
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open http://localhost:3001
        else
            echo "请手动打开浏览器访问: http://localhost:3001"
        fi
    fi
}

# 更新服务
update_services() {
    log_section "更新服务"
    
    echo "更新选项:"
    echo "  1. 更新基础镜像"
    echo "  2. 重建应用镜像"
    echo "  3. 更新并重启所有服务"
    echo
    read -p "请选择 (1-3): " choice
    
    case $choice in
        1)
            log_info "更新基础镜像..."
            docker-compose -f ../docker-compose.local.yml pull
            log_success "基础镜像更新完成"
            ;;
        2)
            log_info "重建应用镜像..."
            ./deploy.sh --chatbot --skip-build=false --force
            log_success "应用镜像重建完成"
            ;;
        3)
            log_info "更新并重启所有服务..."
            docker-compose -f ../docker-compose.local.yml pull
            docker-compose -f ../docker-compose.local.yml down
            ./deploy.sh --full
            log_success "所有服务已更新并重启"
            ;;
        *)
            log_error "无效选择"
            ;;
    esac
}

# 主菜单
show_menu() {
    echo "🤖 智能聊天机器人系统 - 部署助手"
    echo "=================================================="
    echo
    echo "1. 快速启动 (dev/test/demo)"
    echo "2. 健康检查报告"
    echo "3. 备份数据"
    echo "4. 恢复数据"
    echo "5. 性能调优建议"
    echo "6. 故障诊断"
    echo "7. 监控仪表板"
    echo "8. 更新服务"
    echo "9. 退出"
    echo
    read -p "请选择操作 (1-9): " choice
    
    case $choice in
        1)
            read -p "选择环境 (dev/test/demo): " env
            quick_start "$env"
            ;;
        2)
            health_report
            ;;
        3)
            backup_data
            ;;
        4)
            read -p "输入备份文件路径: " backup_file
            restore_data "$backup_file"
            ;;
        5)
            performance_tune
            ;;
        6)
            diagnose
            ;;
        7)
            monitor_dashboard
            ;;
        8)
            update_services
            ;;
        9)
            echo "再见！"
            exit 0
            ;;
        *)
            log_error "无效选择"
            ;;
    esac
    
    echo
    read -p "按回车键继续..."
    clear
    show_menu
}

# 命令行参数处理
case "${1:-menu}" in
    "quick")
        quick_start "${2:-dev}"
        ;;
    "health")
        health_report
        ;;
    "backup")
        backup_data
        ;;
    "restore")
        restore_data "$2"
        ;;
    "tune")
        performance_tune
        ;;
    "diagnose")
        diagnose
        ;;
    "monitor")
        monitor_dashboard
        ;;
    "update")
        update_services
        ;;
    "menu"|*)
        clear
        show_menu
        ;;
esac
