#!/bin/bash

# VoiceHelper Kubernetes 部署脚本
# 适用于 Docker Desktop Kubernetes

set -euo pipefail

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/tools/deployment/k8s"
HELM_DIR="$SCRIPT_DIR/tools/deployment/helm"
DEFAULT_NAMESPACE="voicehelper"
DEFAULT_ACTION="deploy"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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
║                VoiceHelper Kubernetes 部署工具              ║
║              适用于 Docker Desktop Kubernetes               ║
║                       版本: 2.0.0                          ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
VoiceHelper Kubernetes 部署工具

用法:
    $0 [选项] <命令>

命令:
    deploy          部署所有服务
    undeploy        删除所有服务
    status          显示部署状态
    logs            显示服务日志
    restart         重启服务
    scale           扩缩容服务
    upgrade         升级服务
    backup          备份数据
    restore         恢复数据

选项:
    -n, --namespace NAMESPACE   指定命名空间 [默认: voicehelper]
    -c, --component COMPONENT   指定组件 (core|dify|monitoring|all) [默认: all]
    -m, --method METHOD         部署方法 (kubectl|helm) [默认: kubectl]
    -f, --force                 强制执行操作
    -v, --verbose               详细输出
    -h, --help                  显示帮助信息

组件说明:
    core        - VoiceHelper 核心服务 (数据库、网关、算法服务)
    dify        - Dify AI 平台服务
    monitoring  - 监控和管理工具
    all         - 所有服务

示例:
    # 部署所有服务
    $0 deploy

    # 仅部署核心服务
    $0 -c core deploy

    # 使用 Helm 部署
    $0 -m helm deploy

    # 查看服务状态
    $0 status

    # 扩容算法服务
    $0 scale algo-service=3

    # 查看日志
    $0 logs gateway

    # 备份数据
    $0 backup
EOF
}

# 解析命令行参数
parse_args() {
    NAMESPACE="$DEFAULT_NAMESPACE"
    COMPONENT="all"
    METHOD="kubectl"
    FORCE=false
    VERBOSE=false
    ACTION=""
    TARGET=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--component)
                COMPONENT="$2"
                shift 2
                ;;
            -m|--method)
                METHOD="$2"
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
            -h|--help)
                show_help
                exit 0
                ;;
            deploy|undeploy|status|logs|restart|scale|upgrade|backup|restore)
                ACTION="$1"
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
    TARGET="${1:-}"
}

# 验证环境
validate_environment() {
    log_info "验证 Kubernetes 环境..."
    
    # 检查 kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装或不在 PATH 中"
        exit 1
    fi
    
    # 检查 Kubernetes 连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到 Kubernetes 集群"
        log_error "请确保 Docker Desktop Kubernetes 已启用"
        exit 1
    fi
    
    # 检查 Helm (如果使用 Helm 部署)
    if [[ "$METHOD" == "helm" ]] && ! command -v helm &> /dev/null; then
        log_error "Helm 未安装或不在 PATH 中"
        exit 1
    fi
    
    # 验证组件
    case $COMPONENT in
        core|dify|monitoring|all)
            ;;
        *)
            log_error "无效的组件: $COMPONENT"
            log_error "支持的组件: core, dify, monitoring, all"
            exit 1
            ;;
    esac
    
    # 验证方法
    case $METHOD in
        kubectl|helm)
            ;;
        *)
            log_error "无效的部署方法: $METHOD"
            log_error "支持的方法: kubectl, helm"
            exit 1
            ;;
    esac
    
    log_success "环境验证通过"
}

# 检查 Docker Desktop Kubernetes 配置
check_docker_desktop() {
    log_info "检查 Docker Desktop Kubernetes 配置..."
    
    # 检查当前上下文
    local current_context
    current_context=$(kubectl config current-context)
    
    if [[ "$current_context" != "docker-desktop" ]]; then
        log_warning "当前 Kubernetes 上下文不是 docker-desktop: $current_context"
        log_info "切换到 docker-desktop 上下文..."
        kubectl config use-context docker-desktop
    fi
    
    # 检查节点状态
    local node_status
    node_status=$(kubectl get nodes --no-headers | awk '{print $2}' | head -1)
    
    if [[ "$node_status" != "Ready" ]]; then
        log_error "Kubernetes 节点未就绪: $node_status"
        exit 1
    fi
    
    # 检查存储类
    if ! kubectl get storageclass &> /dev/null; then
        log_warning "未找到存储类，将创建默认存储类"
    fi
    
    log_success "Docker Desktop Kubernetes 配置正常"
}

# 创建命名空间
create_namespaces() {
    log_info "创建命名空间..."
    
    case $COMPONENT in
        core|all)
            kubectl apply -f "$K8S_DIR/namespace.yaml" || true
            ;;
        dify)
            kubectl create namespace voicehelper-dify --dry-run=client -o yaml | kubectl apply -f - || true
            ;;
        monitoring)
            kubectl create namespace voicehelper-monitoring --dry-run=client -o yaml | kubectl apply -f - || true
            ;;
    esac
    
    log_success "命名空间创建完成"
}

# 应用存储配置
apply_storage() {
    log_info "应用存储配置..."
    
    if kubectl apply -f "$K8S_DIR/storage.yaml"; then
        log_success "存储配置应用成功"
    else
        log_error "存储配置应用失败"
        return 1
    fi
}

# 应用配置和密钥
apply_configs() {
    log_info "应用配置和密钥..."
    
    # 检查是否需要用户配置密钥
    if [[ ! -f ".env" ]]; then
        if [[ -f "env.unified" ]]; then
            log_info "复制环境配置文件..."
            cp env.unified .env
        else
            log_warning "未找到环境配置文件，使用默认配置"
        fi
    fi
    
    # 应用 ConfigMap
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    
    # 应用 Secrets (需要用户手动编辑真实密钥)
    kubectl apply -f "$K8S_DIR/secrets.yaml"
    
    log_success "配置和密钥应用成功"
}

# 部署数据库服务
deploy_databases() {
    log_info "部署数据库服务..."
    
    kubectl apply -f "$K8S_DIR/databases.yaml"
    kubectl apply -f "$K8S_DIR/milvus.yaml"
    
    # 等待数据库就绪
    log_info "等待数据库服务启动..."
    kubectl wait --for=condition=ready pod -l component=database -n "$NAMESPACE" --timeout=300s || true
    
    log_success "数据库服务部署完成"
}

# 部署核心应用
deploy_applications() {
    log_info "部署核心应用服务..."
    
    kubectl apply -f "$K8S_DIR/applications.yaml"
    
    # 等待应用就绪
    log_info "等待应用服务启动..."
    kubectl wait --for=condition=ready pod -l component=api-gateway -n "$NAMESPACE" --timeout=300s || true
    
    log_success "核心应用服务部署完成"
}

# 部署 Dify 服务
deploy_dify() {
    log_info "部署 Dify AI 平台..."
    
    kubectl apply -f "$K8S_DIR/dify.yaml"
    
    # 等待 Dify 服务就绪
    log_info "等待 Dify 服务启动..."
    kubectl wait --for=condition=ready pod -l component=api-service -n voicehelper-dify --timeout=300s || true
    
    log_success "Dify AI 平台部署完成"
}

# 部署监控服务
deploy_monitoring() {
    log_info "部署监控和管理工具..."
    
    kubectl apply -f "$K8S_DIR/monitoring.yaml"
    
    # 等待监控服务就绪
    log_info "等待监控服务启动..."
    kubectl wait --for=condition=ready pod -l component=monitoring -n voicehelper-monitoring --timeout=300s || true
    
    log_success "监控和管理工具部署完成"
}

# 部署 Ingress
deploy_ingress() {
    log_info "部署 Ingress 和负载均衡..."
    
    kubectl apply -f "$K8S_DIR/ingress.yaml"
    
    log_success "Ingress 和负载均衡部署完成"
}

# 使用 kubectl 部署
deploy_with_kubectl() {
    log_info "使用 kubectl 部署 VoiceHelper..."
    
    # 创建命名空间
    create_namespaces
    
    # 应用存储配置
    apply_storage
    
    # 应用配置和密钥
    apply_configs
    
    case $COMPONENT in
        core)
            deploy_databases
            deploy_applications
            ;;
        dify)
            deploy_dify
            ;;
        monitoring)
            deploy_monitoring
            ;;
        all)
            deploy_databases
            deploy_applications
            deploy_dify
            deploy_monitoring
            deploy_ingress
            ;;
    esac
    
    log_success "kubectl 部署完成"
}

# 使用 Helm 部署
deploy_with_helm() {
    log_info "使用 Helm 部署 VoiceHelper..."
    
    # 添加必要的 Helm 仓库
    log_info "添加 Helm 仓库..."
    helm repo add bitnami https://charts.bitnami.com/bitnami || true
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
    helm repo add grafana https://grafana.github.io/helm-charts || true
    helm repo update
    
    # 创建命名空间
    create_namespaces
    
    # 部署 VoiceHelper
    local helm_args=()
    if [[ "$VERBOSE" == "true" ]]; then
        helm_args+=("--debug")
    fi
    
    case $COMPONENT in
        core)
            helm_args+=("--set" "dify.enabled=false" "--set" "monitoring.enabled=false")
            ;;
        dify)
            helm_args+=("--set" "services.gateway.enabled=false" "--set" "monitoring.enabled=false")
            ;;
        monitoring)
            helm_args+=("--set" "services.gateway.enabled=false" "--set" "dify.enabled=false")
            ;;
    esac
    
    helm upgrade --install voicehelper "$HELM_DIR/voicehelper" \
        --namespace "$NAMESPACE" \
        --create-namespace \
        "${helm_args[@]}" \
        --wait --timeout=20m
    
    log_success "Helm 部署完成"
}

# 主部署函数
cmd_deploy() {
    log_info "开始部署 VoiceHelper 到 Kubernetes..."
    log_info "组件: $COMPONENT, 方法: $METHOD, 命名空间: $NAMESPACE"
    
    case $METHOD in
        kubectl)
            deploy_with_kubectl
            ;;
        helm)
            deploy_with_helm
            ;;
    esac
    
    # 显示部署结果
    show_deployment_info
}

# 删除部署
cmd_undeploy() {
    log_info "删除 VoiceHelper 部署..."
    
    if [[ "$FORCE" == "true" ]] || confirm "确定要删除所有部署吗？"; then
        case $METHOD in
            kubectl)
                case $COMPONENT in
                    core)
                        kubectl delete -f "$K8S_DIR/applications.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/databases.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/milvus.yaml" --ignore-not-found=true
                        ;;
                    dify)
                        kubectl delete -f "$K8S_DIR/dify.yaml" --ignore-not-found=true
                        ;;
                    monitoring)
                        kubectl delete -f "$K8S_DIR/monitoring.yaml" --ignore-not-found=true
                        ;;
                    all)
                        kubectl delete -f "$K8S_DIR/ingress.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/monitoring.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/dify.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/applications.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/databases.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/milvus.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/configmap.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/secrets.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/storage.yaml" --ignore-not-found=true
                        kubectl delete -f "$K8S_DIR/namespace.yaml" --ignore-not-found=true
                        ;;
                esac
                ;;
            helm)
                helm uninstall voicehelper -n "$NAMESPACE" || true
                ;;
        esac
        
        log_success "部署删除完成"
    else
        log_info "操作已取消"
    fi
}

# 显示部署状态
cmd_status() {
    log_info "VoiceHelper Kubernetes 部署状态:"
    
    echo
    echo -e "${CYAN}📋 命名空间状态:${NC}"
    kubectl get namespaces | grep voicehelper || echo "未找到 VoiceHelper 命名空间"
    
    echo
    echo -e "${CYAN}🏗️ 核心服务状态:${NC}"
    kubectl get pods,svc -n "$NAMESPACE" -o wide || echo "未找到核心服务"
    
    echo
    echo -e "${CYAN}🤖 Dify 服务状态:${NC}"
    kubectl get pods,svc -n voicehelper-dify -o wide || echo "未找到 Dify 服务"
    
    echo
    echo -e "${CYAN}📊 监控服务状态:${NC}"
    kubectl get pods,svc -n voicehelper-monitoring -o wide || echo "未找到监控服务"
    
    echo
    echo -e "${CYAN}🌐 Ingress 状态:${NC}"
    kubectl get ingress -A || echo "未找到 Ingress"
    
    echo
    echo -e "${CYAN}💾 存储状态:${NC}"
    kubectl get pv,pvc -A | grep voicehelper || echo "未找到存储卷"
}

# 显示服务日志
cmd_logs() {
    local service="${TARGET:-gateway}"
    
    log_info "显示 $service 服务日志..."
    
    if kubectl get pods -n "$NAMESPACE" -l app="$service" &> /dev/null; then
        kubectl logs -f -n "$NAMESPACE" -l app="$service" --tail=100
    else
        log_error "未找到服务: $service"
        log_info "可用服务:"
        kubectl get pods -n "$NAMESPACE" -o custom-columns=NAME:.metadata.labels.app --no-headers | sort | uniq
    fi
}

# 重启服务
cmd_restart() {
    local service="${TARGET:-all}"
    
    log_info "重启服务: $service"
    
    if [[ "$service" == "all" ]]; then
        kubectl rollout restart deployment -n "$NAMESPACE"
        kubectl rollout restart deployment -n voicehelper-dify
        kubectl rollout restart deployment -n voicehelper-monitoring
    else
        kubectl rollout restart deployment "$service" -n "$NAMESPACE"
    fi
    
    log_success "服务重启完成"
}

# 扩缩容服务
cmd_scale() {
    if [[ -z "$TARGET" ]]; then
        log_error "请指定扩缩容参数，格式: service=replicas"
        log_error "示例: $0 scale gateway=5"
        exit 1
    fi
    
    local service_replicas="$TARGET"
    local service="${service_replicas%=*}"
    local replicas="${service_replicas#*=}"
    
    log_info "扩缩容服务 $service 到 $replicas 个副本..."
    
    kubectl scale deployment "$service" --replicas="$replicas" -n "$NAMESPACE"
    
    log_success "扩缩容完成"
}

# 升级服务
cmd_upgrade() {
    log_info "升级 VoiceHelper 服务..."
    
    case $METHOD in
        kubectl)
            # 重新应用配置
            kubectl apply -f "$K8S_DIR/"
            ;;
        helm)
            helm upgrade voicehelper "$HELM_DIR/voicehelper" -n "$NAMESPACE"
            ;;
    esac
    
    log_success "服务升级完成"
}

# 备份数据
cmd_backup() {
    log_info "备份 VoiceHelper 数据..."
    
    local backup_dir="./backups/k8s/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # 备份配置
    log_info "备份 Kubernetes 配置..."
    kubectl get all,configmap,secret,pvc -n "$NAMESPACE" -o yaml > "$backup_dir/voicehelper-config.yaml"
    kubectl get all,configmap,secret,pvc -n voicehelper-dify -o yaml > "$backup_dir/dify-config.yaml" || true
    kubectl get all,configmap,secret,pvc -n voicehelper-monitoring -o yaml > "$backup_dir/monitoring-config.yaml" || true
    
    # 备份数据库
    log_info "备份数据库数据..."
    kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_dumpall -U voicehelper > "$backup_dir/postgres.sql" || true
    kubectl exec -n voicehelper-dify deployment/dify-postgres -- pg_dumpall -U dify > "$backup_dir/dify-postgres.sql" || true
    
    log_success "数据备份完成: $backup_dir"
}

# 恢复数据
cmd_restore() {
    local backup_dir="${TARGET:-}"
    
    if [[ -z "$backup_dir" ]] || [[ ! -d "$backup_dir" ]]; then
        log_error "请指定有效的备份目录"
        exit 1
    fi
    
    log_info "从 $backup_dir 恢复数据..."
    
    if [[ "$FORCE" == "true" ]] || confirm "确定要恢复数据吗？这将覆盖现有数据。"; then
        # 恢复配置
        if [[ -f "$backup_dir/voicehelper-config.yaml" ]]; then
            kubectl apply -f "$backup_dir/voicehelper-config.yaml"
        fi
        
        # 恢复数据库
        if [[ -f "$backup_dir/postgres.sql" ]]; then
            kubectl exec -i -n "$NAMESPACE" deployment/postgres -- psql -U voicehelper < "$backup_dir/postgres.sql"
        fi
        
        log_success "数据恢复完成"
    else
        log_info "操作已取消"
    fi
}

# 显示部署信息
show_deployment_info() {
    log_success "🎉 VoiceHelper Kubernetes 部署完成！"
    echo
    echo -e "${CYAN}📋 服务访问地址:${NC}"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ 主要服务 (需要配置 hosts 文件)                             │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ 🌐 VoiceHelper Web:  http://voicehelper.local              │"
    echo "│ 🤖 Dify 控制台:      http://voicehelper.local/dify         │"
    echo "│ 🛠️  管理工具:         http://admin.voicehelper.local       │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ 直接访问 (NodePort)                                        │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ 🌐 Nginx 入口:       http://localhost:30080               │"
    echo "│ 🔒 HTTPS 入口:       https://localhost:30443              │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo
    echo -e "${YELLOW}🔧 配置 hosts 文件:${NC}"
    echo "echo '127.0.0.1 voicehelper.local admin.voicehelper.local' | sudo tee -a /etc/hosts"
    echo
    echo -e "${GREEN}✨ 快速开始:${NC}"
    echo "  1. 配置 hosts 文件"
    echo "  2. 访问 http://voicehelper.local"
    echo "  3. 访问 Dify 控制台: http://voicehelper.local/dify"
    echo "  4. 查看服务状态: $0 status"
    echo "  5. 查看日志: $0 logs gateway"
    echo
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
    check_docker_desktop
    
    case "$ACTION" in
        deploy)
            cmd_deploy
            ;;
        undeploy)
            cmd_undeploy
            ;;
        status)
            cmd_status
            ;;
        logs)
            cmd_logs
            ;;
        restart)
            cmd_restart
            ;;
        scale)
            cmd_scale
            ;;
        upgrade)
            cmd_upgrade
            ;;
        backup)
            cmd_backup
            ;;
        restore)
            cmd_restore
            ;;
        "")
            log_error "请指定命令"
            show_help
            exit 1
            ;;
        *)
            log_error "未知命令: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# 错误处理
trap 'log_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"
