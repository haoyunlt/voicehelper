#!/bin/bash

# BGE+FAISS 服务测试脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 BGE+FAISS 服务测试开始${NC}"

# 配置
NAMESPACE="voicehelper"
BGE_SERVICE="bge-service"
FAISS_SERVICE="faiss-service"

# 函数：检查服务状态
check_service_status() {
    local service=$1
    local port=$2
    
    echo -e "${YELLOW}检查 ${service} 服务状态...${NC}"
    
    # 检查Pod状态
    if kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/name=${service} | grep -q "Running"; then
        echo -e "${GREEN}✅ ${service} Pod 正在运行${NC}"
    else
        echo -e "${RED}❌ ${service} Pod 未运行${NC}"
        kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/name=${service}
        return 1
    fi
    
    # 检查服务端点
    if kubectl get endpoints -n ${NAMESPACE} ${service} | grep -q "${port}"; then
        echo -e "${GREEN}✅ ${service} 服务端点可用${NC}"
    else
        echo -e "${RED}❌ ${service} 服务端点不可用${NC}"
        kubectl get endpoints -n ${NAMESPACE} ${service}
        return 1
    fi
    
    return 0
}

# 函数：测试BGE服务
test_bge_service() {
    echo -e "\n${BLUE}🔤 测试BGE向量化服务...${NC}"
    
    # 端口转发
    echo -e "${YELLOW}启动端口转发...${NC}"
    kubectl port-forward -n ${NAMESPACE} service/${BGE_SERVICE} 8080:8080 &
    PORT_FORWARD_PID=$!
    sleep 5
    
    # 健康检查
    echo -e "${YELLOW}健康检查...${NC}"
    if curl -s http://localhost:8080/health | grep -q "healthy"; then
        echo -e "${GREEN}✅ BGE服务健康检查通过${NC}"
    else
        echo -e "${RED}❌ BGE服务健康检查失败${NC}"
        kill $PORT_FORWARD_PID
        return 1
    fi
    
    # 测试向量化
    echo -e "${YELLOW}测试文本向量化...${NC}"
    RESPONSE=$(curl -s -X POST http://localhost:8080/embed \
        -H "Content-Type: application/json" \
        -d '{
            "texts": ["你好世界", "Hello World"],
            "normalize": true
        }')
    
    if echo "$RESPONSE" | grep -q "embeddings"; then
        echo -e "${GREEN}✅ 文本向量化测试通过${NC}"
        DIMENSION=$(echo "$RESPONSE" | jq -r '.dimension')
        echo -e "${BLUE}向量维度: ${DIMENSION}${NC}"
    else
        echo -e "${RED}❌ 文本向量化测试失败${NC}"
        echo "$RESPONSE"
        kill $PORT_FORWARD_PID
        return 1
    fi
    
    # 清理
    kill $PORT_FORWARD_PID
    sleep 2
    
    return 0
}

# 函数：测试FAISS服务
test_faiss_service() {
    echo -e "\n${BLUE}🔍 测试FAISS搜索服务...${NC}"
    
    # 端口转发
    echo -e "${YELLOW}启动端口转发...${NC}"
    kubectl port-forward -n ${NAMESPACE} service/${FAISS_SERVICE} 8081:8081 &
    PORT_FORWARD_PID=$!
    sleep 5
    
    # 健康检查
    echo -e "${YELLOW}健康检查...${NC}"
    if curl -s http://localhost:8081/health | grep -q "healthy"; then
        echo -e "${GREEN}✅ FAISS服务健康检查通过${NC}"
    else
        echo -e "${RED}❌ FAISS服务健康检查失败${NC}"
        kill $PORT_FORWARD_PID
        return 1
    fi
    
    # 测试添加向量
    echo -e "${YELLOW}测试添加向量...${NC}"
    # 生成测试向量（1024维）
    VECTORS='[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0'
    for i in {11..1024}; do
        VECTORS="${VECTORS}, $(echo "scale=3; $i/1000" | bc)"
    done
    VECTORS="${VECTORS}], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1"
    for i in {11..1024}; do
        VECTORS="${VECTORS}, $(echo "scale=3; ($i+100)/1000" | bc)"
    done
    VECTORS="${VECTORS}]]"
    
    ADD_RESPONSE=$(curl -s -X POST http://localhost:8081/add \
        -H "Content-Type: application/json" \
        -d "{
            \"vectors\": ${VECTORS},
            \"ids\": [\"test1\", \"test2\"],
            \"metadata\": [{\"text\": \"测试向量1\"}, {\"text\": \"测试向量2\"}]
        }")
    
    if echo "$ADD_RESPONSE" | grep -q "Added 2 vectors"; then
        echo -e "${GREEN}✅ 向量添加测试通过${NC}"
    else
        echo -e "${RED}❌ 向量添加测试失败${NC}"
        echo "$ADD_RESPONSE"
        kill $PORT_FORWARD_PID
        return 1
    fi
    
    # 测试向量搜索
    echo -e "${YELLOW}测试向量搜索...${NC}"
    QUERY_VECTOR='[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.95'
    for i in {11..1024}; do
        QUERY_VECTOR="${QUERY_VECTOR}, $(echo "scale=3; ($i+50)/1000" | bc)"
    done
    QUERY_VECTOR="${QUERY_VECTOR}]"
    
    SEARCH_RESPONSE=$(curl -s -X POST http://localhost:8081/search \
        -H "Content-Type: application/json" \
        -d "{
            \"query_vector\": ${QUERY_VECTOR},
            \"k\": 2
        }")
    
    if echo "$SEARCH_RESPONSE" | grep -q "results"; then
        echo -e "${GREEN}✅ 向量搜索测试通过${NC}"
        RESULT_COUNT=$(echo "$SEARCH_RESPONSE" | jq -r '.results | length')
        echo -e "${BLUE}搜索结果数量: ${RESULT_COUNT}${NC}"
    else
        echo -e "${RED}❌ 向量搜索测试失败${NC}"
        echo "$SEARCH_RESPONSE"
        kill $PORT_FORWARD_PID
        return 1
    fi
    
    # 获取统计信息
    echo -e "${YELLOW}获取索引统计信息...${NC}"
    STATS_RESPONSE=$(curl -s http://localhost:8081/stats)
    if echo "$STATS_RESPONSE" | grep -q "total_vectors"; then
        echo -e "${GREEN}✅ 统计信息获取成功${NC}"
        TOTAL_VECTORS=$(echo "$STATS_RESPONSE" | jq -r '.total_vectors')
        DIMENSION=$(echo "$STATS_RESPONSE" | jq -r '.dimension')
        echo -e "${BLUE}索引向量数量: ${TOTAL_VECTORS}${NC}"
        echo -e "${BLUE}向量维度: ${DIMENSION}${NC}"
    else
        echo -e "${YELLOW}⚠️  统计信息获取失败${NC}"
    fi
    
    # 清理
    kill $PORT_FORWARD_PID
    sleep 2
    
    return 0
}

# 函数：集成测试
integration_test() {
    echo -e "\n${BLUE}🔗 BGE+FAISS 集成测试...${NC}"
    
    # 同时启动两个服务的端口转发
    echo -e "${YELLOW}启动端口转发...${NC}"
    kubectl port-forward -n ${NAMESPACE} service/${BGE_SERVICE} 8080:8080 &
    BGE_PID=$!
    kubectl port-forward -n ${NAMESPACE} service/${FAISS_SERVICE} 8081:8081 &
    FAISS_PID=$!
    sleep 5
    
    # 1. 使用BGE生成向量
    echo -e "${YELLOW}步骤1: 使用BGE生成向量...${NC}"
    BGE_RESPONSE=$(curl -s -X POST http://localhost:8080/embed \
        -H "Content-Type: application/json" \
        -d '{
            "texts": ["人工智能技术", "机器学习算法", "深度学习模型"],
            "normalize": true
        }')
    
    if ! echo "$BGE_RESPONSE" | grep -q "embeddings"; then
        echo -e "${RED}❌ BGE向量生成失败${NC}"
        kill $BGE_PID $FAISS_PID
        return 1
    fi
    
    # 提取向量
    EMBEDDINGS=$(echo "$BGE_RESPONSE" | jq -r '.embeddings')
    echo -e "${GREEN}✅ BGE向量生成成功${NC}"
    
    # 2. 将向量添加到FAISS
    echo -e "${YELLOW}步骤2: 将向量添加到FAISS...${NC}"
    ADD_RESPONSE=$(curl -s -X POST http://localhost:8081/add \
        -H "Content-Type: application/json" \
        -d "{
            \"vectors\": ${EMBEDDINGS},
            \"ids\": [\"ai_tech\", \"ml_algo\", \"dl_model\"],
            \"metadata\": [
                {\"text\": \"人工智能技术\", \"category\": \"AI\"},
                {\"text\": \"机器学习算法\", \"category\": \"ML\"},
                {\"text\": \"深度学习模型\", \"category\": \"DL\"}
            ]
        }")
    
    if ! echo "$ADD_RESPONSE" | grep -q "Added 3 vectors"; then
        echo -e "${RED}❌ FAISS向量添加失败${NC}"
        kill $BGE_PID $FAISS_PID
        return 1
    fi
    echo -e "${GREEN}✅ FAISS向量添加成功${NC}"
    
    # 3. 生成查询向量并搜索
    echo -e "${YELLOW}步骤3: 生成查询向量并搜索...${NC}"
    QUERY_RESPONSE=$(curl -s -X POST http://localhost:8080/embed \
        -H "Content-Type: application/json" \
        -d '{
            "texts": ["AI技术应用"],
            "normalize": true
        }')
    
    if ! echo "$QUERY_RESPONSE" | grep -q "embeddings"; then
        echo -e "${RED}❌ 查询向量生成失败${NC}"
        kill $BGE_PID $FAISS_PID
        return 1
    fi
    
    QUERY_VECTOR=$(echo "$QUERY_RESPONSE" | jq -r '.embeddings[0]')
    
    # 4. 执行相似度搜索
    echo -e "${YELLOW}步骤4: 执行相似度搜索...${NC}"
    SEARCH_RESPONSE=$(curl -s -X POST http://localhost:8081/search \
        -H "Content-Type: application/json" \
        -d "{
            \"query_vector\": ${QUERY_VECTOR},
            \"k\": 3
        }")
    
    if echo "$SEARCH_RESPONSE" | grep -q "results"; then
        echo -e "${GREEN}✅ 集成测试成功${NC}"
        
        # 显示搜索结果
        echo -e "${BLUE}搜索结果:${NC}"
        echo "$SEARCH_RESPONSE" | jq -r '.results[] | "ID: \(.id), Score: \(.score), Text: \(.metadata.text)"'
    else
        echo -e "${RED}❌ 相似度搜索失败${NC}"
        echo "$SEARCH_RESPONSE"
        kill $BGE_PID $FAISS_PID
        return 1
    fi
    
    # 清理
    kill $BGE_PID $FAISS_PID
    sleep 2
    
    return 0
}

# 主测试流程
main() {
    # 检查依赖
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl 未安装${NC}"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}❌ curl 未安装${NC}"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}❌ jq 未安装${NC}"
        exit 1
    fi
    
    if ! command -v bc &> /dev/null; then
        echo -e "${RED}❌ bc 未安装${NC}"
        exit 1
    fi
    
    # 检查Kubernetes连接
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ 无法连接到 Kubernetes 集群${NC}"
        exit 1
    fi
    
    # 检查命名空间
    if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
        echo -e "${RED}❌ 命名空间 ${NAMESPACE} 不存在${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 环境检查通过${NC}"
    
    # 执行测试
    local test_failed=0
    
    # 检查服务状态
    if ! check_service_status ${BGE_SERVICE} 8080; then
        test_failed=1
    fi
    
    if ! check_service_status ${FAISS_SERVICE} 8081; then
        test_failed=1
    fi
    
    if [ $test_failed -eq 1 ]; then
        echo -e "${RED}❌ 服务状态检查失败${NC}"
        exit 1
    fi
    
    # 单独测试BGE服务
    if ! test_bge_service; then
        test_failed=1
    fi
    
    # 单独测试FAISS服务
    if ! test_faiss_service; then
        test_failed=1
    fi
    
    # 集成测试
    if ! integration_test; then
        test_failed=1
    fi
    
    # 测试结果
    if [ $test_failed -eq 0 ]; then
        echo -e "\n${GREEN}🎉 所有测试通过！BGE+FAISS 服务运行正常${NC}"
    else
        echo -e "\n${RED}❌ 部分测试失败${NC}"
        exit 1
    fi
}

# 运行主函数
main "$@"
