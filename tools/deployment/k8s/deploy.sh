#!/bin/bash

# VoiceHelper Kubernetes 部署脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 配置
NAMESPACE="voicehelper"
MONITORING_NAMESPACE="voicehelper-monitoring"
STORAGE_NAMESPACE="voicehelper-storage"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"

# 部署模式
DEPLOY_MODE=${1:-"full"}  # full, minimal, monitoring-only

echo -e "${BLUE}🚀 VoiceHelper Kubernetes 部署开始${NC}"
echo -e "${BLUE}部署模式: ${DEPLOY_MODE}${NC}"
echo -e "${BLUE}脚本目录: ${SCRIPT_DIR}${NC}"
echo -e "${BLUE}项目根目录: ${PROJECT_ROOT}${NC}"

# 函数：检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 命令未找到，请先安装${NC}"
        exit 1
    fi
}

# 函数：等待部署就绪
wait_for_deployment() {
    local namespace=$1
    local deployment=$2
    local timeout=${3:-300}
    
    echo -e "${YELLOW}⏳ 等待 ${deployment} 在 ${namespace} 命名空间中就绪...${NC}"
    
    if kubectl wait --for=condition=available --timeout=${timeout}s deployment/${deployment} -n ${namespace}; then
        echo -e "${GREEN}✅ ${deployment} 部署就绪${NC}"
        return 0
    else
        echo -e "${RED}❌ ${deployment} 部署超时${NC}"
        return 1
    fi
}

# 函数：等待StatefulSet就绪
wait_for_statefulset() {
    local namespace=$1
    local statefulset=$2
    local timeout=${3:-300}
    
    echo -e "${YELLOW}⏳ 等待 ${statefulset} StatefulSet 在 ${namespace} 命名空间中就绪...${NC}"
    
    if kubectl wait --for=jsonpath='{.status.readyReplicas}'=1 --timeout=${timeout}s statefulset/${statefulset} -n ${namespace}; then
        echo -e "${GREEN}✅ ${statefulset} StatefulSet 就绪${NC}"
        return 0
    else
        echo -e "${RED}❌ ${statefulset} StatefulSet 超时${NC}"
        return 1
    fi
}

# 函数：检查Pod状态
check_pod_status() {
    local namespace=$1
    local label_selector=$2
    
    echo -e "${BLUE}📊 检查 ${namespace} 命名空间中的 Pod 状态 (${label_selector})...${NC}"
    kubectl get pods -n ${namespace} -l ${label_selector} -o wide
}

# 函数：应用YAML文件
apply_yaml() {
    local yaml_file=$1
    local description=$2
    
    if [ ! -f "${yaml_file}" ]; then
        echo -e "${RED}❌ YAML 文件不存在: ${yaml_file}${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}📄 应用 ${description}: ${yaml_file}${NC}"
    
    if kubectl apply -f "${yaml_file}"; then
        echo -e "${GREEN}✅ ${description} 应用成功${NC}"
        return 0
    else
        echo -e "${RED}❌ ${description} 应用失败${NC}"
        return 1
    fi
}

# 函数：创建持久化卷
create_persistent_volumes() {
    echo -e "\n${PURPLE}💾 创建持久化卷...${NC}"
    
    # 创建本地存储目录
    local storage_base="/opt/voicehelper-storage"
    
    cat << EOF | kubectl apply -f -
# PostgreSQL 持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: postgresql-pv
  labels:
    app.kubernetes.io/name: postgresql
    app.kubernetes.io/component: database
spec:
  capacity:
    storage: 20Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/postgresql
    type: DirectoryOrCreate
---
# Redis 持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: redis-pv
  labels:
    app.kubernetes.io/name: redis
    app.kubernetes.io/component: cache
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/redis
    type: DirectoryOrCreate
---
# MinIO 持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: minio-pv
  labels:
    app.kubernetes.io/name: minio
    app.kubernetes.io/component: storage
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/minio
    type: DirectoryOrCreate
---
# FAISS 数据持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: faiss-data-pv
  labels:
    app.kubernetes.io/name: faiss-service
    app.kubernetes.io/component: vector-search
spec:
  capacity:
    storage: 20Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/faiss
    type: DirectoryOrCreate
---
# NATS 持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nats-pv
  labels:
    app.kubernetes.io/name: nats
    app.kubernetes.io/component: message-queue
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/nats
    type: DirectoryOrCreate
---
# 模型缓存持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-cache-pv
  labels:
    app.kubernetes.io/name: model-cache
    app.kubernetes.io/component: storage
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/model-cache
    type: DirectoryOrCreate
---
# Prometheus 持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: prometheus-pv
  labels:
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/component: monitoring
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/prometheus
    type: DirectoryOrCreate
---
# Grafana 持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: grafana-pv
  labels:
    app.kubernetes.io/name: grafana
    app.kubernetes.io/component: visualization
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/grafana
    type: DirectoryOrCreate
---
# AlertManager 持久化卷
apiVersion: v1
kind: PersistentVolume
metadata:
  name: alertmanager-pv
  labels:
    app.kubernetes.io/name: alertmanager
    app.kubernetes.io/component: alerting
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: voicehelper-storage
  hostPath:
    path: ${storage_base}/alertmanager
    type: DirectoryOrCreate
EOF

    echo -e "${GREEN}✅ 持久化卷创建完成${NC}"
}

# 函数：部署基础设施
deploy_infrastructure() {
    echo -e "\n${PURPLE}🏗️  部署基础设施...${NC}"
    
    # 1. 命名空间和基础配置
    apply_yaml "${SCRIPT_DIR}/00-namespace.yaml" "命名空间和基础配置"
    
    # 2. 配置和密钥
    apply_yaml "${SCRIPT_DIR}/01-configmap-secrets.yaml" "配置和密钥"
    
    # 3. 创建持久化卷
    create_persistent_volumes
    
    # 4. 第三方服务
    apply_yaml "${SCRIPT_DIR}/02-third-party-services.yaml" "第三方服务"
    
    # 5. BGE+FAISS向量服务
    apply_yaml "${SCRIPT_DIR}/03-vector-services-bge-faiss.yaml" "BGE+FAISS向量服务"
    
    echo -e "${GREEN}✅ 基础设施部署完成${NC}"
}

# 函数：部署应用服务
deploy_applications() {
    echo -e "\n${PURPLE}🚀 部署应用服务...${NC}"
    
    # 应用服务
    apply_yaml "${SCRIPT_DIR}/04-application-services.yaml" "应用服务"
    
    echo -e "${GREEN}✅ 应用服务部署完成${NC}"
}

# 函数：部署监控系统
deploy_monitoring() {
    echo -e "\n${PURPLE}📊 部署监控系统...${NC}"
    
    # 监控服务
    apply_yaml "${SCRIPT_DIR}/05-monitoring-services.yaml" "监控服务"
    
    echo -e "${GREEN}✅ 监控系统部署完成${NC}"
}

# 函数：部署Ingress
deploy_ingress() {
    echo -e "\n${PURPLE}🌐 部署 Ingress 和负载均衡...${NC}"
    
    # Ingress 和负载均衡
    apply_yaml "${SCRIPT_DIR}/06-ingress-loadbalancer.yaml" "Ingress 和负载均衡"
    
    echo -e "${GREEN}✅ Ingress 部署完成${NC}"
}

# 函数：等待所有服务就绪
wait_for_services() {
    echo -e "\n${PURPLE}⏳ 等待服务就绪...${NC}"
    
    # 等待基础设施服务
    echo -e "${BLUE}等待基础设施服务...${NC}"
    wait_for_statefulset ${NAMESPACE} postgresql 600
    wait_for_statefulset ${NAMESPACE} redis 300
    wait_for_statefulset ${NAMESPACE} minio 300
    wait_for_deployment ${NAMESPACE} bge-service 600
    wait_for_statefulset ${NAMESPACE} faiss-service 300
    wait_for_statefulset ${NAMESPACE} nats 300
    
    # 等待应用服务
    if [ "${DEPLOY_MODE}" != "monitoring-only" ]; then
        echo -e "${BLUE}等待应用服务...${NC}"
        wait_for_deployment ${NAMESPACE} gateway 300
        wait_for_deployment ${NAMESPACE} algo-service 600
        wait_for_deployment ${NAMESPACE} frontend 300
        wait_for_deployment ${NAMESPACE} voice-service 300
    fi
    
    # 等待监控服务
    if [ "${DEPLOY_MODE}" == "full" ] || [ "${DEPLOY_MODE}" == "monitoring-only" ]; then
        echo -e "${BLUE}等待监控服务...${NC}"
        wait_for_statefulset ${MONITORING_NAMESPACE} prometheus 300
        wait_for_statefulset ${MONITORING_NAMESPACE} grafana 300
        wait_for_deployment ${MONITORING_NAMESPACE} jaeger 300
        wait_for_statefulset ${MONITORING_NAMESPACE} alertmanager 300
    fi
    
    echo -e "${GREEN}✅ 所有服务已就绪${NC}"
}

# 函数：显示部署状态
show_deployment_status() {
    echo -e "\n${PURPLE}📋 部署状态概览${NC}"
    
    # 显示命名空间
    echo -e "\n${BLUE}命名空间:${NC}"
    kubectl get namespaces | grep voicehelper
    
    # 显示服务状态
    echo -e "\n${BLUE}服务状态:${NC}"
    kubectl get services -n ${NAMESPACE} -o wide
    
    # 显示Pod状态
    echo -e "\n${BLUE}Pod 状态:${NC}"
    kubectl get pods -n ${NAMESPACE} -o wide
    
    # 显示持久化卷
    echo -e "\n${BLUE}持久化卷:${NC}"
    kubectl get pv | grep voicehelper-storage
    
    # 显示持久化卷声明
    echo -e "\n${BLUE}持久化卷声明:${NC}"
    kubectl get pvc -n ${NAMESPACE}
    
    # 显示Ingress
    echo -e "\n${BLUE}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE}
    
    if [ "${DEPLOY_MODE}" == "full" ] || [ "${DEPLOY_MODE}" == "monitoring-only" ]; then
        echo -e "\n${BLUE}监控服务:${NC}"
        kubectl get pods -n ${MONITORING_NAMESPACE} -o wide
        kubectl get ingress -n ${MONITORING_NAMESPACE}
    fi
}

# 函数：显示访问信息
show_access_info() {
    echo -e "\n${PURPLE}🌐 访问信息${NC}"
    
    # 获取Ingress IP
    local ingress_ip=$(kubectl get service ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    
    if [ "${ingress_ip}" == "" ] || [ "${ingress_ip}" == "localhost" ]; then
        ingress_ip="localhost"
        echo -e "${YELLOW}⚠️  使用本地访问，请确保配置了端口转发${NC}"
    fi
    
    echo -e "\n${GREEN}🔗 应用访问地址:${NC}"
    echo -e "  主应用:           http://voicehelper.local"
    echo -e "  API服务:          http://api.voicehelper.local"
    echo -e "  WebSocket:        ws://ws.voicehelper.local"
    echo -e "  语音服务:         http://voice.voicehelper.local"
    
    if [ "${DEPLOY_MODE}" == "full" ] || [ "${DEPLOY_MODE}" == "monitoring-only" ]; then
        echo -e "\n${GREEN}📊 监控访问地址:${NC}"
        echo -e "  Grafana:          http://grafana.voicehelper.local (admin/VoiceHelper2025!)"
        echo -e "  Prometheus:       http://prometheus.voicehelper.local"
        echo -e "  Jaeger:           http://jaeger.voicehelper.local"
        echo -e "  AlertManager:     http://alertmanager.voicehelper.local"
    fi
    
    echo -e "\n${GREEN}🔧 管理工具:${NC}"
    echo -e "  MinIO Console:    http://minio.voicehelper.local (voicehelper/VoiceHelper2025Storage)"
    echo -e "  BGE服务:          http://bge.voicehelper.local"
    echo -e "  FAISS服务:        http://faiss.voicehelper.local"
    
    echo -e "\n${YELLOW}📝 注意事项:${NC}"
    echo -e "  1. 请将以上域名添加到 /etc/hosts 文件中，指向 ${ingress_ip}"
    echo -e "  2. 监控和管理工具需要基本认证 (admin/VoiceHelper2025!)"
    echo -e "  3. 首次启动可能需要几分钟时间下载和初始化"
    
    # 生成hosts文件条目
    echo -e "\n${BLUE}📋 /etc/hosts 条目:${NC}"
    cat << EOF
${ingress_ip} voicehelper.local
${ingress_ip} api.voicehelper.local
${ingress_ip} ws.voicehelper.local
${ingress_ip} voice.voicehelper.local
${ingress_ip} algo.voicehelper.local
${ingress_ip} grafana.voicehelper.local
${ingress_ip} prometheus.voicehelper.local
${ingress_ip} jaeger.voicehelper.local
${ingress_ip} alertmanager.voicehelper.local
${ingress_ip} minio.voicehelper.local
${ingress_ip} bge.voicehelper.local
${ingress_ip} faiss.voicehelper.local
EOF
}

# 函数：清理部署
cleanup_deployment() {
    echo -e "\n${RED}🧹 清理部署...${NC}"
    
    read -p "确定要删除所有 VoiceHelper 资源吗？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}删除命名空间和所有资源...${NC}"
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        kubectl delete namespace ${MONITORING_NAMESPACE} --ignore-not-found=true
        kubectl delete namespace ingress-nginx --ignore-not-found=true
        
        echo -e "${YELLOW}删除持久化卷...${NC}"
        kubectl delete pv -l app.kubernetes.io/name=postgresql --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=redis --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=minio --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=faiss-service --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=nats --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=model-cache --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=prometheus --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=grafana --ignore-not-found=true
        kubectl delete pv -l app.kubernetes.io/name=alertmanager --ignore-not-found=true
        
        echo -e "${GREEN}✅ 清理完成${NC}"
    else
        echo -e "${BLUE}取消清理操作${NC}"
    fi
}

# 主函数
main() {
    # 检查必要命令
    check_command kubectl
    check_command docker
    
    # 检查Kubernetes连接
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ 无法连接到 Kubernetes 集群${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Kubernetes 集群连接正常${NC}"
    
    case "${DEPLOY_MODE}" in
        "full")
            deploy_infrastructure
            deploy_applications
            deploy_monitoring
            deploy_ingress
            wait_for_services
            ;;
        "minimal")
            deploy_infrastructure
            deploy_applications
            deploy_ingress
            wait_for_services
            ;;
        "monitoring-only")
            deploy_infrastructure
            deploy_monitoring
            deploy_ingress
            wait_for_services
            ;;
        "cleanup")
            cleanup_deployment
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 未知的部署模式: ${DEPLOY_MODE}${NC}"
            echo -e "${BLUE}支持的模式: full, minimal, monitoring-only, cleanup${NC}"
            exit 1
            ;;
    esac
    
    # 显示部署状态和访问信息
    show_deployment_status
    show_access_info
    
    echo -e "\n${GREEN}🎉 VoiceHelper 部署完成！${NC}"
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}VoiceHelper Kubernetes 部署脚本${NC}"
    echo -e ""
    echo -e "${YELLOW}用法:${NC}"
    echo -e "  $0 [模式]"
    echo -e ""
    echo -e "${YELLOW}模式:${NC}"
    echo -e "  full           - 完整部署（默认）"
    echo -e "  minimal        - 最小部署（不包含监控）"
    echo -e "  monitoring-only - 仅部署监控系统"
    echo -e "  cleanup        - 清理所有资源"
    echo -e ""
    echo -e "${YELLOW}示例:${NC}"
    echo -e "  $0 full        # 完整部署"
    echo -e "  $0 minimal     # 最小部署"
    echo -e "  $0 cleanup     # 清理资源"
}

# 处理命令行参数
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_help
    exit 0
fi

# 运行主函数
main "$@"
