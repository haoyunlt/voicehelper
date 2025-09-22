#!/bin/bash

# VoiceHelper 部署脚本
# 支持本地、测试、生产环境部署

set -euo pipefail

# 默认配置
ENVIRONMENT="${ENVIRONMENT:-local}"
NAMESPACE="${NAMESPACE:-voicehelper}"
HELM_RELEASE="${HELM_RELEASE:-voicehelper}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN="${DRY_RUN:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600s}"

# 颜色输出
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

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
VoiceHelper 部署脚本

用法:
    $0 [选项]

选项:
    -e, --environment ENV    部署环境 (local|staging|production) [默认: local]
    -n, --namespace NS       Kubernetes命名空间 [默认: voicehelper]
    -r, --release NAME       Helm发布名称 [默认: voicehelper]
    -t, --tag TAG           镜像标签 [默认: latest]
    -d, --dry-run           仅显示将要执行的操作，不实际部署
    -w, --wait-timeout TIME  等待部署完成的超时时间 [默认: 600s]
    -h, --help              显示此帮助信息

环境变量:
    KUBECONFIG              Kubernetes配置文件路径
    DOCKER_REGISTRY         Docker镜像仓库地址
    HELM_VALUES_FILE        自定义Helm values文件路径

示例:
    # 本地部署
    $0 -e local

    # 部署到测试环境
    $0 -e staging -t v1.2.0

    # 生产环境部署（干运行）
    $0 -e production -t v1.2.0 -d

    # 使用自定义配置
    HELM_VALUES_FILE=./custom-values.yaml $0 -e production
EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--release)
                HELM_RELEASE="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -w|--wait-timeout)
                WAIT_TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 验证环境
validate_environment() {
    case $ENVIRONMENT in
        local|staging|production)
            ;;
        *)
            log_error "无效的环境: $ENVIRONMENT"
            log_error "支持的环境: local, staging, production"
            exit 1
            ;;
    esac
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖工具..."
    
    local missing_tools=()
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v helm &> /dev/null; then
        missing_tools+=("helm")
    fi
    
    if [[ $ENVIRONMENT == "local" ]] && ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "缺少以下工具: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "依赖检查通过"
}

# 检查Kubernetes连接
check_kubernetes() {
    log_info "检查Kubernetes连接..."
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到Kubernetes集群"
        log_error "请检查KUBECONFIG或kubectl配置"
        exit 1
    fi
    
    local context=$(kubectl config current-context)
    log_success "已连接到Kubernetes集群: $context"
}

# 准备命名空间
prepare_namespace() {
    log_info "准备命名空间: $NAMESPACE"
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "创建命名空间: $NAMESPACE"
        if [[ $DRY_RUN == "true" ]]; then
            log_info "[DRY RUN] kubectl create namespace $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
        fi
    else
        log_info "命名空间已存在: $NAMESPACE"
    fi
}

# 构建本地镜像
build_local_images() {
    if [[ $ENVIRONMENT != "local" ]]; then
        return
    fi
    
    log_info "构建本地Docker镜像..."
    
    local components=("gateway" "algo" "voice" "frontend")
    
    for component in "${components[@]}"; do
        log_info "构建 $component 镜像..."
        
        if [[ $DRY_RUN == "true" ]]; then
            log_info "[DRY RUN] docker build -t voicehelper/$component:$IMAGE_TAG ./$component"
        else
            docker build -t "voicehelper/$component:$IMAGE_TAG" "./$component" || {
                log_error "构建 $component 镜像失败"
                exit 1
            }
        fi
    done
    
    log_success "本地镜像构建完成"
}

# 获取Helm values文件
get_values_file() {
    local values_file="./deploy/helm/voicehelper/values.yaml"
    
    # 检查自定义values文件
    if [[ -n "${HELM_VALUES_FILE:-}" ]]; then
        if [[ -f "$HELM_VALUES_FILE" ]]; then
            values_file="$HELM_VALUES_FILE"
            log_info "使用自定义values文件: $values_file"
        else
            log_error "自定义values文件不存在: $HELM_VALUES_FILE"
            exit 1
        fi
    else
        # 检查环境特定的values文件
        local env_values_file="./deploy/helm/voicehelper/values-$ENVIRONMENT.yaml"
        if [[ -f "$env_values_file" ]]; then
            values_file="$env_values_file"
            log_info "使用环境特定values文件: $values_file"
        fi
    fi
    
    echo "$values_file"
}

# 部署应用
deploy_application() {
    log_info "部署VoiceHelper到 $ENVIRONMENT 环境..."
    
    local values_file
    values_file=$(get_values_file)
    
    local helm_args=(
        "upgrade" "--install" "$HELM_RELEASE"
        "./deploy/helm/voicehelper"
        "--namespace" "$NAMESPACE"
        "--values" "$values_file"
        "--set" "image.tag=$IMAGE_TAG"
        "--timeout" "$WAIT_TIMEOUT"
    )
    
    # 环境特定配置
    case $ENVIRONMENT in
        local)
            helm_args+=(
                "--set" "postgresql.enabled=true"
                "--set" "redis.enabled=true"
                "--set" "ingress.enabled=false"
            )
            ;;
        staging)
            helm_args+=(
                "--set" "ingress.hosts[0].host=staging.voicehelper.ai"
                "--set" "ingress.hosts[1].host=staging-api.voicehelper.ai"
            )
            ;;
        production)
            helm_args+=(
                "--set" "ingress.hosts[0].host=voicehelper.ai"
                "--set" "ingress.hosts[1].host=api.voicehelper.ai"
                "--set" "services.gateway.replicaCount=5"
                "--set" "services.algo.replicaCount=3"
            )
            ;;
    esac
    
    if [[ $DRY_RUN == "true" ]]; then
        helm_args+=("--dry-run")
        log_info "[DRY RUN] helm ${helm_args[*]}"
    else
        helm_args+=("--wait")
    fi
    
    log_info "执行Helm部署..."
    helm "${helm_args[@]}" || {
        log_error "Helm部署失败"
        exit 1
    }
    
    if [[ $DRY_RUN == "false" ]]; then
        log_success "应用部署完成"
    else
        log_success "干运行完成，未实际部署"
    fi
}

# 验证部署
verify_deployment() {
    if [[ $DRY_RUN == "true" ]]; then
        return
    fi
    
    log_info "验证部署状态..."
    
    # 检查Pod状态
    log_info "检查Pod状态..."
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$HELM_RELEASE"
    
    # 等待Pod就绪
    log_info "等待Pod就绪..."
    kubectl wait --for=condition=ready pod \
        -l "app.kubernetes.io/instance=$HELM_RELEASE" \
        -n "$NAMESPACE" \
        --timeout="$WAIT_TIMEOUT" || {
        log_error "Pod未能在指定时间内就绪"
        kubectl describe pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=$HELM_RELEASE"
        exit 1
    }
    
    # 检查服务状态
    log_info "检查服务状态..."
    kubectl get services -n "$NAMESPACE" -l "app.kubernetes.io/instance=$HELM_RELEASE"
    
    log_success "部署验证通过"
}

# 运行健康检查
health_check() {
    if [[ $DRY_RUN == "true" ]]; then
        return
    fi
    
    log_info "运行健康检查..."
    
    # 获取网关服务端点
    local gateway_service="$HELM_RELEASE-gateway"
    local gateway_port
    
    if [[ $ENVIRONMENT == "local" ]]; then
        # 本地环境使用端口转发
        log_info "启动端口转发..."
        kubectl port-forward -n "$NAMESPACE" "service/$gateway_service" 8080:8080 &
        local port_forward_pid=$!
        sleep 5
        
        # 健康检查
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "健康检查通过"
        else
            log_error "健康检查失败"
            kill $port_forward_pid 2>/dev/null || true
            exit 1
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    else
        # 远程环境直接访问
        local health_url
        case $ENVIRONMENT in
            staging)
                health_url="https://staging-api.voicehelper.ai/health"
                ;;
            production)
                health_url="https://api.voicehelper.ai/health"
                ;;
        esac
        
        if curl -f "$health_url" &> /dev/null; then
            log_success "健康检查通过: $health_url"
        else
            log_error "健康检查失败: $health_url"
            exit 1
        fi
    fi
}

# 显示部署信息
show_deployment_info() {
    if [[ $DRY_RUN == "true" ]]; then
        return
    fi
    
    log_info "部署信息:"
    echo "  环境: $ENVIRONMENT"
    echo "  命名空间: $NAMESPACE"
    echo "  Helm发布: $HELM_RELEASE"
    echo "  镜像标签: $IMAGE_TAG"
    
    case $ENVIRONMENT in
        local)
            echo "  访问地址:"
            echo "    前端: http://localhost:3000 (需要端口转发)"
            echo "    API: http://localhost:8080 (需要端口转发)"
            echo ""
            echo "  端口转发命令:"
            echo "    kubectl port-forward -n $NAMESPACE service/$HELM_RELEASE-frontend 3000:3000"
            echo "    kubectl port-forward -n $NAMESPACE service/$HELM_RELEASE-gateway 8080:8080"
            ;;
        staging)
            echo "  访问地址:"
            echo "    前端: https://staging.voicehelper.ai"
            echo "    API: https://staging-api.voicehelper.ai"
            ;;
        production)
            echo "  访问地址:"
            echo "    前端: https://voicehelper.ai"
            echo "    API: https://api.voicehelper.ai"
            ;;
    esac
    
    echo ""
    echo "  监控地址:"
    echo "    Grafana: https://grafana.voicehelper.ai"
    echo "    Prometheus: https://prometheus.voicehelper.ai"
}

# 主函数
main() {
    log_info "VoiceHelper 部署脚本启动"
    
    parse_args "$@"
    validate_environment
    check_dependencies
    check_kubernetes
    prepare_namespace
    build_local_images
    deploy_application
    verify_deployment
    health_check
    show_deployment_info
    
    log_success "部署完成！"
}

# 错误处理
trap 'log_error "部署过程中发生错误，退出码: $?"' ERR

# 执行主函数
main "$@"