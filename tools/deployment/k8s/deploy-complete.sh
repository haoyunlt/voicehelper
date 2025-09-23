#!/bin/bash

# VoiceHelper Kubernetes 完整部署脚本
# 版本: 2.0.0
# 作者: VoiceHelper Team

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 配置变量
NAMESPACE="voicehelper"
STORAGE_NAMESPACE="voicehelper-storage"
MONITORING_NAMESPACE="voicehelper-monitoring"
DEPLOYMENT_MODE="${1:-full}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"
TIMEOUT="${TIMEOUT:-600}"

# 日志函数
log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# 检查依赖
check_dependencies() {
    log "检查部署依赖..."
    
    local deps=("kubectl" "docker" "helm" "jq" "curl")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "缺少依赖: $dep"
        fi
    done
    
    # 检查Kubernetes连接
    if ! kubectl cluster-info &> /dev/null; then
        error "无法连接到Kubernetes集群"
    fi
    
    # 检查Docker守护进程
    if ! docker info &> /dev/null; then
        error "Docker守护进程未运行"
    fi
    
    success "依赖检查完成"
}

# 构建镜像
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        warning "跳过镜像构建"
        return
    fi
    
    log "构建应用镜像..."
    
    # 构建BGE服务镜像
    info "构建BGE服务镜像..."
    docker build -f "$PROJECT_ROOT/algo/Dockerfile.bge" -t voicehelper/bge-service:latest "$PROJECT_ROOT/algo"
    
    # 构建FAISS服务镜像
    info "构建FAISS服务镜像..."
    docker build -f "$PROJECT_ROOT/algo/Dockerfile.faiss" -t voicehelper/faiss-service:latest "$PROJECT_ROOT/algo"
    
    # 构建Gateway镜像
    info "构建Gateway镜像..."
    docker build -f "$PROJECT_ROOT/backend/Dockerfile" -t voicehelper/gateway:latest "$PROJECT_ROOT/backend"
    
    # 构建Algo服务镜像
    info "构建Algo服务镜像..."
    docker build -f "$PROJECT_ROOT/algo/Dockerfile" -t voicehelper/algo-service:latest "$PROJECT_ROOT/algo"
    
    # 构建前端镜像
    info "构建前端镜像..."
    docker build -f "$PROJECT_ROOT/platforms/web/Dockerfile" -t voicehelper/frontend:latest "$PROJECT_ROOT/platforms/web"
    
    # 构建Voice服务镜像
    info "构建Voice服务镜像..."
    docker build -f "$PROJECT_ROOT/platforms/web/Dockerfile.voice" -t voicehelper/voice-service:latest "$PROJECT_ROOT/platforms/web"
    
    success "镜像构建完成"
}

# 应用YAML文件
apply_yaml() {
    local file="$1"
    local description="$2"
    
    if [[ ! -f "$file" ]]; then
        warning "文件不存在: $file"
        return
    fi
    
    info "部署: $description"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl apply -f "$file" --dry-run=client
    else
        kubectl apply -f "$file"
    fi
}

# 等待资源就绪
wait_for_resource() {
    local resource_type="$1"
    local resource_name="$2"
    local namespace="$3"
    local timeout="${4:-300}"
    
    info "等待 $resource_type/$resource_name 在命名空间 $namespace 中就绪..."
    
    if ! kubectl wait --for=condition=ready "$resource_type/$resource_name" -n "$namespace" --timeout="${timeout}s"; then
        error "$resource_type/$resource_name 未能在 ${timeout}s 内就绪"
    fi
}

# 部署前置条件
deploy_prerequisites() {
    log "部署前置条件..."
    
    # 命名空间和RBAC
    apply_yaml "$SCRIPT_DIR/00-prerequisites/namespace.yaml" "命名空间配置"
    apply_yaml "$SCRIPT_DIR/00-prerequisites/rbac.yaml" "RBAC配置"
    apply_yaml "$SCRIPT_DIR/00-prerequisites/storage-classes.yaml" "存储类配置"
    
    success "前置条件部署完成"
}

# 部署基础设施
deploy_infrastructure() {
    log "部署基础设施..."
    
    # 配置和密钥
    apply_yaml "$SCRIPT_DIR/01-infrastructure/configmaps.yaml" "配置映射"
    apply_yaml "$SCRIPT_DIR/01-infrastructure/secrets.yaml" "密钥配置"
    
    success "基础设施部署完成"
}

# 部署存储服务
deploy_storage_services() {
    log "部署存储服务..."
    
    # PostgreSQL集群
    apply_yaml "$SCRIPT_DIR/02-storage-services/postgresql-cluster.yaml" "PostgreSQL集群"
    
    # Redis集群
    apply_yaml "$SCRIPT_DIR/02-storage-services/redis-cluster.yaml" "Redis集群"
    
    # MinIO对象存储
    apply_yaml "$SCRIPT_DIR/02-storage-services/minio-cluster.yaml" "MinIO对象存储"
    
    # 等待存储服务就绪
    if [[ "$DRY_RUN" != "true" ]]; then
        wait_for_resource "statefulset" "postgres-master" "$STORAGE_NAMESPACE" 300
        wait_for_resource "statefulset" "redis-cluster" "$STORAGE_NAMESPACE" 300
        wait_for_resource "statefulset" "minio" "$STORAGE_NAMESPACE" 300
    fi
    
    success "存储服务部署完成"
}

# 部署消息队列服务
deploy_messaging_services() {
    log "部署消息队列服务..."
    
    # NATS JetStream
    if [[ -f "$SCRIPT_DIR/03-messaging-services/nats-jetstream.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/03-messaging-services/nats-jetstream.yaml" "NATS JetStream"
    fi
    
    # Kafka集群
    if [[ -f "$SCRIPT_DIR/03-messaging-services/kafka-cluster.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/03-messaging-services/kafka-cluster.yaml" "Kafka集群"
    fi
    
    # RabbitMQ集群
    if [[ -f "$SCRIPT_DIR/03-messaging-services/rabbitmq-cluster.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/03-messaging-services/rabbitmq-cluster.yaml" "RabbitMQ集群"
    fi
    
    success "消息队列服务部署完成"
}

# 部署AI/ML服务
deploy_ai_ml_services() {
    log "部署AI/ML服务..."
    
    # BGE向量化服务
    apply_yaml "$SCRIPT_DIR/04-ai-ml-services/bge-service.yaml" "BGE向量化服务"
    
    # FAISS向量搜索服务
    apply_yaml "$SCRIPT_DIR/04-ai-ml-services/faiss-service.yaml" "FAISS向量搜索服务"
    
    # 等待AI/ML服务就绪
    if [[ "$DRY_RUN" != "true" ]]; then
        wait_for_resource "deployment" "bge-service" "$NAMESPACE" 300
        wait_for_resource "statefulset" "faiss-service" "$NAMESPACE" 300
    fi
    
    success "AI/ML服务部署完成"
}

# 部署应用服务
deploy_application_services() {
    log "部署应用服务..."
    
    # Gateway API网关
    apply_yaml "$SCRIPT_DIR/05-application-services/gateway.yaml" "Gateway API网关"
    
    # Algo服务
    if [[ -f "$SCRIPT_DIR/05-application-services/algo-service.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/05-application-services/algo-service.yaml" "Algo服务"
    fi
    
    # Voice服务
    if [[ -f "$SCRIPT_DIR/05-application-services/voice-service.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/05-application-services/voice-service.yaml" "Voice服务"
    fi
    
    # 前端服务
    if [[ -f "$SCRIPT_DIR/05-application-services/frontend.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/05-application-services/frontend.yaml" "前端服务"
    fi
    
    # 等待应用服务就绪
    if [[ "$DRY_RUN" != "true" ]]; then
        wait_for_resource "deployment" "gateway" "$NAMESPACE" 300
    fi
    
    success "应用服务部署完成"
}

# 部署监控系统
deploy_monitoring_stack() {
    log "部署监控系统..."
    
    # Prometheus
    if [[ -f "$SCRIPT_DIR/06-monitoring-stack/prometheus.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/06-monitoring-stack/prometheus.yaml" "Prometheus监控"
    fi
    
    # Grafana
    if [[ -f "$SCRIPT_DIR/06-monitoring-stack/grafana.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/06-monitoring-stack/grafana.yaml" "Grafana仪表盘"
    fi
    
    # Jaeger链路追踪
    if [[ -f "$SCRIPT_DIR/06-monitoring-stack/jaeger.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/06-monitoring-stack/jaeger.yaml" "Jaeger链路追踪"
    fi
    
    # Fluentd日志收集
    if [[ -f "$SCRIPT_DIR/06-monitoring-stack/fluentd.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/06-monitoring-stack/fluentd.yaml" "Fluentd日志收集"
    fi
    
    success "监控系统部署完成"
}

# 部署网络入口
deploy_ingress_networking() {
    log "部署网络入口..."
    
    # Nginx Ingress
    if [[ -f "$SCRIPT_DIR/08-ingress-networking/nginx-ingress.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/08-ingress-networking/nginx-ingress.yaml" "Nginx Ingress"
    fi
    
    # Ingress规则
    if [[ -f "$SCRIPT_DIR/08-ingress-networking/ingress-rules.yaml" ]]; then
        apply_yaml "$SCRIPT_DIR/08-ingress-networking/ingress-rules.yaml" "Ingress规则"
    fi
    
    success "网络入口部署完成"
}

# 验证部署
verify_deployment() {
    log "验证部署状态..."
    
    # 检查所有Pod状态
    info "检查Pod状态..."
    kubectl get pods -n "$NAMESPACE" -o wide
    kubectl get pods -n "$STORAGE_NAMESPACE" -o wide
    kubectl get pods -n "$MONITORING_NAMESPACE" -o wide
    
    # 检查服务状态
    info "检查服务状态..."
    kubectl get services -n "$NAMESPACE"
    kubectl get services -n "$STORAGE_NAMESPACE"
    kubectl get services -n "$MONITORING_NAMESPACE"
    
    # 健康检查
    info "执行健康检查..."
    
    # 检查Gateway健康状态
    if kubectl get service gateway -n "$NAMESPACE" &> /dev/null; then
        local gateway_ip=$(kubectl get service gateway -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if curl -f "http://$gateway_ip:8080/health" &> /dev/null; then
            success "Gateway健康检查通过"
        else
            warning "Gateway健康检查失败"
        fi
    fi
    
    # 检查BGE服务
    if kubectl get service bge-service -n "$NAMESPACE" &> /dev/null; then
        local bge_ip=$(kubectl get service bge-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if curl -f "http://$bge_ip:8080/health" &> /dev/null; then
            success "BGE服务健康检查通过"
        else
            warning "BGE服务健康检查失败"
        fi
    fi
    
    # 检查FAISS服务
    if kubectl get service faiss-client -n "$NAMESPACE" &> /dev/null; then
        local faiss_ip=$(kubectl get service faiss-client -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if curl -f "http://$faiss_ip:8081/health" &> /dev/null; then
            success "FAISS服务健康检查通过"
        else
            warning "FAISS服务健康检查失败"
        fi
    fi
    
    success "部署验证完成"
}

# 显示访问信息
show_access_info() {
    log "显示访问信息..."
    
    echo -e "\n${PURPLE}=== VoiceHelper 访问信息 ===${NC}"
    
    # 获取Ingress信息
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        echo -e "\n${CYAN}应用访问地址:${NC}"
        kubectl get ingress -n "$NAMESPACE" -o custom-columns=NAME:.metadata.name,HOSTS:.spec.rules[*].host,ADDRESS:.status.loadBalancer.ingress[*].ip
    fi
    
    # 显示端口转发命令
    echo -e "\n${CYAN}本地端口转发命令:${NC}"
    echo "kubectl port-forward -n $NAMESPACE service/gateway 8080:8080"
    echo "kubectl port-forward -n $NAMESPACE service/bge-service 8080:8080"
    echo "kubectl port-forward -n $NAMESPACE service/faiss-client 8081:8081"
    
    if kubectl get service grafana -n "$MONITORING_NAMESPACE" &> /dev/null; then
        echo "kubectl port-forward -n $MONITORING_NAMESPACE service/grafana 3000:3000"
    fi
    
    # 显示监控访问信息
    echo -e "\n${CYAN}监控系统访问:${NC}"
    echo "Grafana: http://localhost:3000 (admin/VoiceHelper2025!)"
    echo "Prometheus: http://localhost:9090"
    echo "Jaeger: http://localhost:16686"
    
    # 显示hosts配置
    echo -e "\n${CYAN}Hosts文件配置 (/etc/hosts):${NC}"
    echo "127.0.0.1 voicehelper.local"
    echo "127.0.0.1 api.voicehelper.local"
    echo "127.0.0.1 bge.voicehelper.local"
    echo "127.0.0.1 faiss.voicehelper.local"
    echo "127.0.0.1 grafana.voicehelper.local"
    echo "127.0.0.1 prometheus.voicehelper.local"
    echo "127.0.0.1 jaeger.voicehelper.local"
    
    success "访问信息显示完成"
}

# 清理部署
cleanup_deployment() {
    log "清理部署..."
    
    warning "这将删除所有VoiceHelper相关资源，包括数据！"
    read -p "确认继续? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "取消清理操作"
        return
    fi
    
    # 删除命名空间（会级联删除所有资源）
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    kubectl delete namespace "$STORAGE_NAMESPACE" --ignore-not-found=true
    kubectl delete namespace "$MONITORING_NAMESPACE" --ignore-not-found=true
    kubectl delete namespace "istio-system" --ignore-not-found=true
    
    # 删除集群级别资源
    kubectl delete clusterrole voicehelper-admin voicehelper-app voicehelper-monitoring --ignore-not-found=true
    kubectl delete clusterrolebinding voicehelper-admin voicehelper-app voicehelper-monitoring --ignore-not-found=true
    kubectl delete podsecuritypolicy voicehelper-psp --ignore-not-found=true
    
    # 删除存储类
    kubectl delete storageclass voicehelper-ssd voicehelper-standard voicehelper-fast voicehelper-nfs voicehelper-local --ignore-not-found=true
    
    # 删除持久化卷
    kubectl delete pv postgres-master-pv postgres-replica-1-pv postgres-replica-2-pv --ignore-not-found=true
    kubectl delete pv redis-node-1-pv minio-data-1-pv faiss-data-pv bge-models-pv --ignore-not-found=true
    kubectl delete pv prometheus-data-pv elasticsearch-data-pv --ignore-not-found=true
    
    success "清理完成"
}

# 显示帮助信息
show_help() {
    echo -e "${PURPLE}VoiceHelper Kubernetes 部署脚本${NC}"
    echo
    echo -e "${CYAN}用法:${NC}"
    echo "  $0 [模式] [选项]"
    echo
    echo -e "${CYAN}部署模式:${NC}"
    echo "  full              完整部署（默认）"
    echo "  minimal           最小部署（仅核心服务）"
    echo "  storage-only      仅部署存储服务"
    echo "  ai-only           仅部署AI/ML服务"
    echo "  monitoring-only   仅部署监控系统"
    echo "  cleanup           清理所有部署"
    echo
    echo -e "${CYAN}环境变量:${NC}"
    echo "  DRY_RUN=true      仅验证配置，不实际部署"
    echo "  SKIP_BUILD=true   跳过镜像构建"
    echo "  TIMEOUT=600       部署超时时间（秒）"
    echo
    echo -e "${CYAN}示例:${NC}"
    echo "  $0 full                    # 完整部署"
    echo "  DRY_RUN=true $0 full       # 验证配置"
    echo "  SKIP_BUILD=true $0 minimal # 最小部署，跳过构建"
    echo "  $0 cleanup                 # 清理部署"
}

# 主函数
main() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                VoiceHelper Kubernetes 部署                   ║"
    echo "║                        版本: 2.0.0                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    case "$DEPLOYMENT_MODE" in
        "full")
            check_dependencies
            build_images
            deploy_prerequisites
            deploy_infrastructure
            deploy_storage_services
            deploy_messaging_services
            deploy_ai_ml_services
            deploy_application_services
            deploy_monitoring_stack
            deploy_ingress_networking
            verify_deployment
            show_access_info
            ;;
        "minimal")
            check_dependencies
            build_images
            deploy_prerequisites
            deploy_infrastructure
            deploy_storage_services
            deploy_ai_ml_services
            deploy_application_services
            verify_deployment
            show_access_info
            ;;
        "storage-only")
            check_dependencies
            deploy_prerequisites
            deploy_infrastructure
            deploy_storage_services
            verify_deployment
            ;;
        "ai-only")
            check_dependencies
            build_images
            deploy_prerequisites
            deploy_infrastructure
            deploy_ai_ml_services
            verify_deployment
            ;;
        "monitoring-only")
            check_dependencies
            deploy_prerequisites
            deploy_infrastructure
            deploy_monitoring_stack
            verify_deployment
            ;;
        "cleanup")
            cleanup_deployment
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            error "未知的部署模式: $DEPLOYMENT_MODE"
            show_help
            ;;
    esac
    
    success "部署脚本执行完成！"
}

# 执行主函数
main "$@"
