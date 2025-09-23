#!/bin/bash

# VoiceHelper 部署状态检查脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

NAMESPACE="voicehelper"
MONITORING_NAMESPACE="voicehelper-monitoring"

echo -e "${BLUE}🔍 VoiceHelper 部署状态检查${NC}"
echo -e "${BLUE}时间: $(date)${NC}"

# 检查命名空间
echo -e "\n${PURPLE}📦 命名空间状态${NC}"
kubectl get namespaces | grep voicehelper || echo -e "${RED}❌ 命名空间未创建${NC}"

# 检查持久化卷
echo -e "\n${PURPLE}💾 持久化卷状态${NC}"
kubectl get pv | grep voicehelper-storage || echo -e "${RED}❌ 持久化卷未创建${NC}"

# 检查持久化卷声明
echo -e "\n${PURPLE}📋 持久化卷声明状态${NC}"
kubectl get pvc -n ${NAMESPACE} || echo -e "${RED}❌ PVC 未创建${NC}"

# 检查配置和密钥
echo -e "\n${PURPLE}⚙️  配置和密钥状态${NC}"
kubectl get configmaps -n ${NAMESPACE}
kubectl get secrets -n ${NAMESPACE}

# 检查服务状态
echo -e "\n${PURPLE}🌐 服务状态${NC}"
kubectl get services -n ${NAMESPACE} -o wide

# 检查Pod状态
echo -e "\n${PURPLE}🚀 Pod 状态${NC}"
kubectl get pods -n ${NAMESPACE} -o wide

# 检查Pod详细状态
echo -e "\n${PURPLE}📊 Pod 详细状态${NC}"
for pod in $(kubectl get pods -n ${NAMESPACE} -o jsonpath='{.items[*].metadata.name}'); do
    status=$(kubectl get pod ${pod} -n ${NAMESPACE} -o jsonpath='{.status.phase}')
    ready=$(kubectl get pod ${pod} -n ${NAMESPACE} -o jsonpath='{.status.containerStatuses[*].ready}' | grep -o true | wc -l)
    total=$(kubectl get pod ${pod} -n ${NAMESPACE} -o jsonpath='{.spec.containers[*].name}' | wc -w)
    
    if [ "${status}" == "Running" ] && [ "${ready}" == "${total}" ]; then
        echo -e "  ${GREEN}✅ ${pod}: ${status} (${ready}/${total})${NC}"
    elif [ "${status}" == "Running" ]; then
        echo -e "  ${YELLOW}⚠️  ${pod}: ${status} (${ready}/${total})${NC}"
    else
        echo -e "  ${RED}❌ ${pod}: ${status} (${ready}/${total})${NC}"
    fi
done

# 检查失败的Pod
echo -e "\n${PURPLE}🔍 失败Pod详情${NC}"
failed_pods=$(kubectl get pods -n ${NAMESPACE} --field-selector=status.phase!=Running -o jsonpath='{.items[*].metadata.name}')
if [ -n "${failed_pods}" ]; then
    for pod in ${failed_pods}; do
        echo -e "\n${RED}❌ Pod: ${pod}${NC}"
        kubectl describe pod ${pod} -n ${NAMESPACE} | tail -10
    done
else
    echo -e "${GREEN}✅ 所有Pod运行正常${NC}"
fi

# 检查Ingress
echo -e "\n${PURPLE}🌍 Ingress 状态${NC}"
kubectl get ingress -n ${NAMESPACE} || echo -e "${YELLOW}⚠️  Ingress 未部署${NC}"

# 检查HPA
echo -e "\n${PURPLE}📈 自动扩缩容状态${NC}"
kubectl get hpa -n ${NAMESPACE} || echo -e "${YELLOW}⚠️  HPA 未部署${NC}"

# 检查监控服务（如果存在）
if kubectl get namespace ${MONITORING_NAMESPACE} &> /dev/null; then
    echo -e "\n${PURPLE}📊 监控服务状态${NC}"
    kubectl get pods -n ${MONITORING_NAMESPACE} -o wide
fi

# 资源使用情况
echo -e "\n${PURPLE}💻 资源使用情况${NC}"
kubectl top nodes || echo -e "${YELLOW}⚠️  Metrics Server 未安装${NC}"
kubectl top pods -n ${NAMESPACE} || echo -e "${YELLOW}⚠️  Pod 指标不可用${NC}"

# 事件检查
echo -e "\n${PURPLE}📝 最近事件${NC}"
kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | tail -10

# 连接测试
echo -e "\n${PURPLE}🔗 连接测试${NC}"

# 测试数据库连接
if kubectl get pod postgresql-0 -n ${NAMESPACE} &> /dev/null; then
    echo -e "${BLUE}测试 PostgreSQL 连接...${NC}"
    if kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- pg_isready -U voicehelper -d voicehelper; then
        echo -e "${GREEN}✅ PostgreSQL 连接正常${NC}"
    else
        echo -e "${RED}❌ PostgreSQL 连接失败${NC}"
    fi
fi

# 测试Redis连接
if kubectl get pod redis-0 -n ${NAMESPACE} &> /dev/null; then
    echo -e "${BLUE}测试 Redis 连接...${NC}"
    if kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli ping; then
        echo -e "${GREEN}✅ Redis 连接正常${NC}"
    else
        echo -e "${RED}❌ Redis 连接失败${NC}"
    fi
fi

# 测试MinIO连接
if kubectl get pod minio-0 -n ${NAMESPACE} &> /dev/null; then
    echo -e "${BLUE}测试 MinIO 连接...${NC}"
    if kubectl exec -n ${NAMESPACE} minio-0 -- curl -f http://localhost:9000/minio/health/live; then
        echo -e "${GREEN}✅ MinIO 连接正常${NC}"
    else
        echo -e "${RED}❌ MinIO 连接失败${NC}"
    fi
fi

# 总结
echo -e "\n${BLUE}📋 部署状态总结${NC}"
total_pods=$(kubectl get pods -n ${NAMESPACE} --no-headers | wc -l)
running_pods=$(kubectl get pods -n ${NAMESPACE} --field-selector=status.phase=Running --no-headers | wc -l)
ready_pods=$(kubectl get pods -n ${NAMESPACE} -o jsonpath='{.items[*].status.containerStatuses[*].ready}' | grep -o true | wc -l)

echo -e "  总Pod数: ${total_pods}"
echo -e "  运行中: ${running_pods}"
echo -e "  就绪数: ${ready_pods}"

if [ "${running_pods}" == "${total_pods}" ]; then
    echo -e "\n${GREEN}🎉 所有服务运行正常！${NC}"
else
    echo -e "\n${YELLOW}⚠️  部分服务未就绪，请检查上述详情${NC}"
fi

echo -e "\n${BLUE}💡 有用的命令:${NC}"
echo -e "  查看Pod日志: kubectl logs <pod-name> -n ${NAMESPACE}"
echo -e "  进入Pod调试: kubectl exec -it <pod-name> -n ${NAMESPACE} -- /bin/sh"
echo -e "  查看服务详情: kubectl describe svc <service-name> -n ${NAMESPACE}"
echo -e "  重启Pod: kubectl delete pod <pod-name> -n ${NAMESPACE}"
