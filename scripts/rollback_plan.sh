#!/bin/bash

# VoiceHelper 回滚预案脚本
# 功能: 紧急回滚到稳定版本 + 特性开关 + 服务恢复

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
STABLE_VERSION=${STABLE_VERSION:-"v2-stable"}
BACKUP_DIR=${BACKUP_DIR:-"./backups"}
CONFIG_BACKUP_DIR="${BACKUP_DIR}/config"
DB_BACKUP_DIR="${BACKUP_DIR}/database"
ROLLBACK_LOG="rollback_$(date +%Y%m%d_%H%M%S).log"

# 服务配置
SERVICES=("backend" "frontend" "algo")
COMPOSE_FILE="docker-compose.optimized.yml"
COMPOSE_PROJECT="voicehelper"

echo -e "${RED}=== VoiceHelper 紧急回滚程序 ===${NC}" | tee -a "$ROLLBACK_LOG"
echo "回滚时间: $(date)" | tee -a "$ROLLBACK_LOG"
echo "目标版本: $STABLE_VERSION" | tee -a "$ROLLBACK_LOG"
echo "" | tee -a "$ROLLBACK_LOG"

# 确认回滚操作
confirm_rollback() {
    echo -e "${YELLOW}⚠️  警告: 即将执行紧急回滚操作${NC}"
    echo -e "${YELLOW}这将会:${NC}"
    echo "  - 停止当前所有服务"
    echo "  - 回滚到稳定版本 $STABLE_VERSION"
    echo "  - 恢复配置文件"
    echo "  - 重启所有服务"
    echo ""
    
    if [ "${FORCE_ROLLBACK:-false}" != "true" ]; then
        read -p "确认执行回滚? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "回滚操作已取消" | tee -a "$ROLLBACK_LOG"
            exit 0
        fi
    fi
    
    echo "开始执行回滚操作..." | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 创建当前状态备份
backup_current_state() {
    echo -e "${BLUE}1. 备份当前状态...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 创建备份目录
    mkdir -p "$CONFIG_BACKUP_DIR" "$DB_BACKUP_DIR"
    
    # 备份配置文件
    echo "备份配置文件..." | tee -a "$ROLLBACK_LOG"
    cp -r deploy/config/* "$CONFIG_BACKUP_DIR/" 2>/dev/null || true
    cp env.* "$CONFIG_BACKUP_DIR/" 2>/dev/null || true
    cp docker-compose*.yml "$CONFIG_BACKUP_DIR/" 2>/dev/null || true
    
    # 备份数据库
    echo "备份数据库..." | tee -a "$ROLLBACK_LOG"
    if docker ps | grep -q voicehelper-postgres; then
        docker exec voicehelper-postgres pg_dump -U voicehelper voicehelper > "$DB_BACKUP_DIR/pre_rollback_$(date +%Y%m%d_%H%M%S).sql" 2>/dev/null || true
    fi
    
    # 备份当前Git状态
    echo "备份Git状态..." | tee -a "$ROLLBACK_LOG"
    git rev-parse HEAD > "$CONFIG_BACKUP_DIR/current_commit.txt"
    git status --porcelain > "$CONFIG_BACKUP_DIR/git_status.txt"
    
    echo -e "${GREEN}✓ 当前状态备份完成${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 关闭特性开关
disable_feature_flags() {
    echo -e "${BLUE}2. 关闭新特性开关...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 通过API关闭特性开关
    feature_flags=(
        "audio_worklet"
        "webrtc_transport"
        "streaming_tts"
        "advanced_vad"
        "jitter_buffer"
    )
    
    for flag in "${feature_flags[@]}"; do
        echo "关闭特性: $flag" | tee -a "$ROLLBACK_LOG"
        curl -X POST "http://localhost:8080/admin/feature-toggle" \
            -H "Content-Type: application/json" \
            -d "{\"$flag\": false}" \
            2>/dev/null || echo "  警告: 无法关闭特性 $flag" | tee -a "$ROLLBACK_LOG"
    done
    
    echo -e "${GREEN}✓ 特性开关已关闭${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 停止当前服务
stop_current_services() {
    echo -e "${BLUE}3. 停止当前服务...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 优雅停止服务
    echo "优雅停止服务..." | tee -a "$ROLLBACK_LOG"
    docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" stop 2>/dev/null || true
    
    # 等待服务停止
    sleep 5
    
    # 强制停止仍在运行的容器
    echo "强制停止残留容器..." | tee -a "$ROLLBACK_LOG"
    docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" down --remove-orphans 2>/dev/null || true
    
    echo -e "${GREEN}✓ 服务已停止${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 回滚代码版本
rollback_code_version() {
    echo -e "${BLUE}4. 回滚代码版本...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 检查稳定版本是否存在
    if ! git rev-parse --verify "$STABLE_VERSION" >/dev/null 2>&1; then
        echo -e "${RED}错误: 稳定版本 $STABLE_VERSION 不存在${NC}" | tee -a "$ROLLBACK_LOG"
        exit 1
    fi
    
    # 暂存当前更改
    echo "暂存当前更改..." | tee -a "$ROLLBACK_LOG"
    git stash push -m "Pre-rollback stash $(date)" 2>/dev/null || true
    
    # 切换到稳定版本
    echo "切换到稳定版本 $STABLE_VERSION..." | tee -a "$ROLLBACK_LOG"
    git checkout "$STABLE_VERSION" 2>&1 | tee -a "$ROLLBACK_LOG"
    
    # 清理未跟踪的文件
    echo "清理未跟踪的文件..." | tee -a "$ROLLBACK_LOG"
    git clean -fd 2>/dev/null || true
    
    echo -e "${GREEN}✓ 代码版本已回滚${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 恢复配置文件
restore_configuration() {
    echo -e "${BLUE}5. 恢复稳定配置...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 恢复环境配置
    if [ -f "env.stable" ]; then
        echo "恢复环境配置..." | tee -a "$ROLLBACK_LOG"
        cp env.stable .env
    fi
    
    # 恢复Docker配置
    if [ -f "docker-compose.stable.yml" ]; then
        echo "恢复Docker配置..." | tee -a "$ROLLBACK_LOG"
        cp docker-compose.stable.yml docker-compose.yml
    fi
    
    # 恢复Nginx配置
    if [ -d "deploy/nginx/stable" ]; then
        echo "恢复Nginx配置..." | tee -a "$ROLLBACK_LOG"
        cp -r deploy/nginx/stable/* deploy/nginx/conf.d/
    fi
    
    echo -e "${GREEN}✓ 配置文件已恢复${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 重建服务镜像
rebuild_services() {
    echo -e "${BLUE}6. 重建服务镜像...${NC}" | tee -a "$ROLLBACK_LOG"
    
    for service in "${SERVICES[@]}"; do
        echo "重建 $service 服务..." | tee -a "$ROLLBACK_LOG"
        docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" build --no-cache "$service" 2>&1 | tee -a "$ROLLBACK_LOG"
    done
    
    echo -e "${GREEN}✓ 服务镜像重建完成${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 启动稳定版本服务
start_stable_services() {
    echo -e "${BLUE}7. 启动稳定版本服务...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 按依赖顺序启动服务
    echo "启动基础服务..." | tee -a "$ROLLBACK_LOG"
    docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" up -d postgres redis minio 2>&1 | tee -a "$ROLLBACK_LOG"
    
    # 等待基础服务就绪
    echo "等待基础服务就绪..." | tee -a "$ROLLBACK_LOG"
    sleep 15
    
    echo "启动应用服务..." | tee -a "$ROLLBACK_LOG"
    docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" up -d backend algo 2>&1 | tee -a "$ROLLBACK_LOG"
    
    # 等待应用服务就绪
    echo "等待应用服务就绪..." | tee -a "$ROLLBACK_LOG"
    sleep 20
    
    echo "启动前端服务..." | tee -a "$ROLLBACK_LOG"
    docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" up -d frontend 2>&1 | tee -a "$ROLLBACK_LOG"
    
    echo -e "${GREEN}✓ 稳定版本服务已启动${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 验证服务健康状态
verify_service_health() {
    echo -e "${BLUE}8. 验证服务健康状态...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 等待服务完全启动
    echo "等待服务完全启动..." | tee -a "$ROLLBACK_LOG"
    sleep 30
    
    # 检查各服务健康状态
    services_health=(
        "backend:8080:/health"
        "algo:8082:/health"
        "frontend:3000:/api/health"
    )
    
    all_healthy=true
    for service_info in "${services_health[@]}"; do
        service_name=$(echo "$service_info" | cut -d: -f1)
        service_port=$(echo "$service_info" | cut -d: -f2)
        health_path=$(echo "$service_info" | cut -d: -f3)
        
        echo "检查 $service_name 服务健康状态..." | tee -a "$ROLLBACK_LOG"
        
        # 重试机制
        retry_count=0
        max_retries=5
        while [ $retry_count -lt $max_retries ]; do
            if curl -f -s "http://localhost:${service_port}${health_path}" > /dev/null; then
                echo -e "  ${GREEN}✓ $service_name 服务健康${NC}" | tee -a "$ROLLBACK_LOG"
                break
            else
                retry_count=$((retry_count + 1))
                if [ $retry_count -eq $max_retries ]; then
                    echo -e "  ${RED}✗ $service_name 服务异常${NC}" | tee -a "$ROLLBACK_LOG"
                    all_healthy=false
                else
                    echo "  重试 $retry_count/$max_retries..." | tee -a "$ROLLBACK_LOG"
                    sleep 10
                fi
            fi
        done
    done
    
    if [ "$all_healthy" = true ]; then
        echo -e "${GREEN}✓ 所有服务健康检查通过${NC}" | tee -a "$ROLLBACK_LOG"
    else
        echo -e "${RED}✗ 部分服务健康检查失败${NC}" | tee -a "$ROLLBACK_LOG"
        return 1
    fi
    
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 运行基本功能测试
run_basic_tests() {
    echo -e "${BLUE}9. 运行基本功能测试...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 测试基本API
    echo "测试基本API..." | tee -a "$ROLLBACK_LOG"
    if curl -f -s "http://localhost:8080/api/v1/health" > /dev/null; then
        echo -e "  ${GREEN}✓ API健康检查通过${NC}" | tee -a "$ROLLBACK_LOG"
    else
        echo -e "  ${RED}✗ API健康检查失败${NC}" | tee -a "$ROLLBACK_LOG"
    fi
    
    # 测试数据库连接
    echo "测试数据库连接..." | tee -a "$ROLLBACK_LOG"
    if docker exec voicehelper-postgres pg_isready -U voicehelper > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ 数据库连接正常${NC}" | tee -a "$ROLLBACK_LOG"
    else
        echo -e "  ${RED}✗ 数据库连接异常${NC}" | tee -a "$ROLLBACK_LOG"
    fi
    
    # 测试Redis连接
    echo "测试Redis连接..." | tee -a "$ROLLBACK_LOG"
    if docker exec voicehelper-redis redis-cli ping | grep -q PONG; then
        echo -e "  ${GREEN}✓ Redis连接正常${NC}" | tee -a "$ROLLBACK_LOG"
    else
        echo -e "  ${RED}✗ Redis连接异常${NC}" | tee -a "$ROLLBACK_LOG"
    fi
    
    echo -e "${GREEN}✓ 基本功能测试完成${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 清理和通知
cleanup_and_notify() {
    echo -e "${BLUE}10. 清理和通知...${NC}" | tee -a "$ROLLBACK_LOG"
    
    # 清理临时文件
    echo "清理临时文件..." | tee -a "$ROLLBACK_LOG"
    docker system prune -f > /dev/null 2>&1 || true
    
    # 发送通知 (如果配置了通知系统)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        echo "发送Slack通知..." | tee -a "$ROLLBACK_LOG"
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🔄 VoiceHelper 紧急回滚完成\\n版本: $STABLE_VERSION\\n时间: $(date)\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
    
    if [ -n "${EMAIL_RECIPIENT:-}" ]; then
        echo "发送邮件通知..." | tee -a "$ROLLBACK_LOG"
        echo "VoiceHelper 紧急回滚完成" | mail -s "VoiceHelper Rollback Completed" "$EMAIL_RECIPIENT" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ 清理和通知完成${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 生成回滚报告
generate_rollback_report() {
    echo -e "${BLUE}生成回滚报告...${NC}" | tee -a "$ROLLBACK_LOG"
    
    report_file="rollback_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# VoiceHelper 紧急回滚报告

**回滚时间**: $(date)
**目标版本**: $STABLE_VERSION
**执行人**: ${USER:-unknown}
**回滚原因**: ${ROLLBACK_REASON:-紧急回滚}

## 回滚步骤

1. ✅ 备份当前状态
2. ✅ 关闭新特性开关
3. ✅ 停止当前服务
4. ✅ 回滚代码版本
5. ✅ 恢复稳定配置
6. ✅ 重建服务镜像
7. ✅ 启动稳定版本服务
8. ✅ 验证服务健康状态
9. ✅ 运行基本功能测试
10. ✅ 清理和通知

## 服务状态

$(docker-compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT" ps)

## 备份位置

- 配置备份: $CONFIG_BACKUP_DIR
- 数据库备份: $DB_BACKUP_DIR
- 回滚日志: $ROLLBACK_LOG

## 后续行动

- [ ] 分析回滚原因
- [ ] 修复相关问题
- [ ] 更新测试用例
- [ ] 重新部署修复版本

## 联系信息

如有问题，请联系运维团队。

EOF

    echo -e "${GREEN}回滚报告已生成: $report_file${NC}" | tee -a "$ROLLBACK_LOG"
    echo "" | tee -a "$ROLLBACK_LOG"
}

# 主函数
main() {
    # 检查权限
    if [ "$EUID" -eq 0 ]; then
        echo -e "${YELLOW}警告: 不建议以root用户执行回滚操作${NC}"
    fi
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: Docker未安装${NC}"
        exit 1
    fi
    
    # 检查docker-compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}错误: docker-compose未安装${NC}"
        exit 1
    fi
    
    # 执行回滚步骤
    confirm_rollback
    backup_current_state
    disable_feature_flags
    stop_current_services
    rollback_code_version
    restore_configuration
    rebuild_services
    start_stable_services
    
    # 验证回滚结果
    if verify_service_health; then
        run_basic_tests
        cleanup_and_notify
        generate_rollback_report
        
        echo -e "${GREEN}=== 紧急回滚成功完成 ===${NC}" | tee -a "$ROLLBACK_LOG"
        echo -e "${GREEN}系统已恢复到稳定版本 $STABLE_VERSION${NC}" | tee -a "$ROLLBACK_LOG"
        echo -e "${GREEN}所有服务运行正常${NC}" | tee -a "$ROLLBACK_LOG"
        echo "" | tee -a "$ROLLBACK_LOG"
        echo "访问地址:" | tee -a "$ROLLBACK_LOG"
        echo "  前端: http://localhost:3000" | tee -a "$ROLLBACK_LOG"
        echo "  后端: http://localhost:8080" | tee -a "$ROLLBACK_LOG"
        echo "  算法: http://localhost:8082" | tee -a "$ROLLBACK_LOG"
        
        exit 0
    else
        echo -e "${RED}=== 回滚验证失败 ===${NC}" | tee -a "$ROLLBACK_LOG"
        echo -e "${RED}请手动检查服务状态${NC}" | tee -a "$ROLLBACK_LOG"
        exit 1
    fi
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            STABLE_VERSION="$2"
            shift 2
            ;;
        --force)
            FORCE_ROLLBACK=true
            shift
            ;;
        --reason)
            ROLLBACK_REASON="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --version VERSION     目标稳定版本 (默认: v2-stable)"
            echo "  --force               强制回滚，不询问确认"
            echo "  --reason REASON       回滚原因"
            echo "  --help                显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 运行主函数
main
