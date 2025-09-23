#!/bin/bash

# VoiceHelper 测试部署脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NAMESPACE="voicehelper-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🧪 VoiceHelper 测试部署开始${NC}"

# 创建测试命名空间
echo -e "${YELLOW}📦 创建测试命名空间...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# 部署一个简单的测试应用
echo -e "${YELLOW}🚀 部署测试应用...${NC}"
cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
  namespace: ${NAMESPACE}
  labels:
    app: test-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test-app
  template:
    metadata:
      labels:
        app: test-app
    spec:
      containers:
      - name: test-app
        image: nginx:alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "128Mi"
            cpu: "100m"
---
apiVersion: v1
kind: Service
metadata:
  name: test-service
  namespace: ${NAMESPACE}
spec:
  selector:
    app: test-app
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
EOF

# 等待部署就绪
echo -e "${YELLOW}⏳ 等待测试应用就绪...${NC}"
kubectl wait --for=condition=available --timeout=120s deployment/test-app -n ${NAMESPACE}

# 检查状态
echo -e "${BLUE}📊 检查测试应用状态...${NC}"
kubectl get pods -n ${NAMESPACE}
kubectl get svc -n ${NAMESPACE}

# 测试服务连接
echo -e "${YELLOW}🔍 测试服务连接...${NC}"
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=test-app -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${POD_NAME} -- curl -s http://test-service/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 测试部署成功！Kubernetes 集群工作正常${NC}"
else
    echo -e "${RED}❌ 测试部署失败${NC}"
    exit 1
fi

# 清理测试资源
echo -e "${YELLOW}🧹 清理测试资源...${NC}"
kubectl delete namespace ${NAMESPACE}

echo -e "${GREEN}🎉 测试完成！可以开始正式部署${NC}"
