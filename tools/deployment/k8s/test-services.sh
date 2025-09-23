#!/bin/bash

# VoiceHelper 服务功能测试套件
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

NAMESPACE="voicehelper"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🧪 VoiceHelper 服务功能测试开始${NC}"
echo -e "${BLUE}时间: $(date)${NC}"

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 函数：运行测试
run_test() {
    local test_name=$1
    local test_command=$2
    local expected_result=${3:-0}
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "\n${YELLOW}🔍 测试: ${test_name}${NC}"
    
    if eval "$test_command"; then
        if [ $? -eq $expected_result ]; then
            echo -e "${GREEN}✅ ${test_name} - 通过${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            echo -e "${RED}❌ ${test_name} - 失败 (返回码不匹配)${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        echo -e "${RED}❌ ${test_name} - 失败${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# 函数：等待Pod就绪
wait_for_pod() {
    local pod_name=$1
    local timeout=${2:-60}
    
    echo -e "${YELLOW}⏳ 等待 ${pod_name} 就绪...${NC}"
    
    if kubectl wait --for=condition=ready pod/${pod_name} -n ${NAMESPACE} --timeout=${timeout}s; then
        echo -e "${GREEN}✅ ${pod_name} 已就绪${NC}"
        return 0
    else
        echo -e "${RED}❌ ${pod_name} 等待超时${NC}"
        return 1
    fi
}

echo -e "\n${PURPLE}📋 1. 基础设施测试${NC}"

# 测试1: 检查命名空间
run_test "命名空间存在性检查" "kubectl get namespace ${NAMESPACE}"

# 测试2: 检查服务发现
run_test "服务发现测试" "kubectl get services -n ${NAMESPACE}"

# 测试3: 检查持久化卷
run_test "持久化卷状态检查" "kubectl get pv | grep voicehelper-storage"

echo -e "\n${PURPLE}📋 2. PostgreSQL 数据库测试${NC}"

# 测试4: PostgreSQL 连接测试
run_test "PostgreSQL 连接测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- pg_isready -U voicehelper -d voicehelper"

# 测试5: PostgreSQL 数据库创建测试
run_test "PostgreSQL 数据库操作测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- psql -U voicehelper -d voicehelper -c 'SELECT version();'"

# 测试6: PostgreSQL 表创建测试
run_test "PostgreSQL 表操作测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- psql -U voicehelper -d voicehelper -c \"CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name VARCHAR(50)); INSERT INTO test_table (name) VALUES ('test'); SELECT * FROM test_table; DROP TABLE test_table;\""

# 测试7: PostgreSQL 监控指标测试
run_test "PostgreSQL 监控指标测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgres-exporter -- wget -q -O - http://localhost:9187/metrics | head -5"

echo -e "\n${PURPLE}📋 3. Redis 缓存测试${NC}"

# 测试8: Redis 连接测试
run_test "Redis 连接测试" "kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli -a Redis2025!@# ping"

# 测试9: Redis 数据操作测试
run_test "Redis 数据操作测试" "kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli -a Redis2025!@# eval 'redis.call(\"SET\", \"test_key\", \"test_value\"); local val = redis.call(\"GET\", \"test_key\"); redis.call(\"DEL\", \"test_key\"); return val' 0"

# 测试10: Redis 监控指标测试
run_test "Redis 监控指标测试" "kubectl exec -n ${NAMESPACE} redis-0 -c redis-exporter -- wget -q -O - http://localhost:9121/metrics | head -5"

# 测试11: Redis 持久化测试
run_test "Redis 持久化配置测试" "kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli -a Redis2025!@# config get save"

echo -e "\n${PURPLE}📋 4. MinIO 对象存储测试${NC}"

# 测试12: MinIO 健康检查
run_test "MinIO 健康检查" "kubectl exec -n ${NAMESPACE} minio-0 -- mc --version"

# 测试13: MinIO 存储桶操作测试
run_test "MinIO 存储桶操作测试" "kubectl exec -n ${NAMESPACE} minio-0 -- sh -c 'mc alias set local http://localhost:9000 voicehelper VoiceHelper2025Storage; mc mb local/test-bucket || true; mc ls local/; mc rb local/test-bucket || true'"

# 测试14: MinIO 文件上传下载测试
run_test "MinIO 文件操作测试" "kubectl exec -n ${NAMESPACE} minio-0 -- sh -c 'mc alias set local http://localhost:9000 voicehelper VoiceHelper2025Storage; echo \"test content\" > /tmp/test.txt; mc mb local/test-bucket || true; mc cp /tmp/test.txt local/test-bucket/; mc cat local/test-bucket/test.txt; mc rm local/test-bucket/test.txt; mc rb local/test-bucket; rm /tmp/test.txt'"

echo -e "\n${PURPLE}📋 5. etcd 元数据存储测试${NC}"

# 测试15: etcd 健康检查
run_test "etcd 健康检查" "kubectl exec -n ${NAMESPACE} etcd-0 -- etcdctl endpoint health"

# 测试16: etcd 数据操作测试
run_test "etcd 数据操作测试" "kubectl exec -n ${NAMESPACE} etcd-0 -- sh -c 'etcdctl put test_key test_value; etcdctl get test_key; etcdctl del test_key'"

# 测试17: etcd 集群状态测试
run_test "etcd 集群状态测试" "kubectl exec -n ${NAMESPACE} etcd-0 -- etcdctl endpoint status --write-out=table"

echo -e "\n${PURPLE}📋 6. 服务间连接性测试${NC}"

# 测试18: 内部DNS解析测试
run_test "内部DNS解析测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- nslookup redis-service"

# 测试19: 服务端口连通性测试
run_test "Redis服务连通性测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- nc -z redis-service 6379"

# 测试20: MinIO服务连通性测试
run_test "MinIO服务连通性测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- nc -z minio-service 9000"

# 测试21: etcd服务连通性测试
run_test "etcd服务连通性测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- nc -z etcd-service 2379"

echo -e "\n${PURPLE}📋 7. 配置和密钥测试${NC}"

# 测试22: ConfigMap 配置测试
run_test "ConfigMap配置测试" "kubectl get configmap voicehelper-config -n ${NAMESPACE} -o jsonpath='{.data.POSTGRES_HOST}'"

# 测试23: Secret 密钥测试
run_test "Secret密钥测试" "kubectl get secret voicehelper-secrets -n ${NAMESPACE} -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d"

# 测试24: 环境变量注入测试
run_test "环境变量注入测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- env | grep POSTGRES_DB"

echo -e "\n${PURPLE}📋 8. 资源和性能测试${NC}"

# 测试25: Pod 资源使用测试
run_test "Pod资源限制测试" "kubectl describe pod postgresql-0 -n ${NAMESPACE} | grep -E 'Limits|Requests'"

# 测试26: 存储使用测试
run_test "存储使用测试" "kubectl exec -n ${NAMESPACE} postgresql-0 -c postgresql -- df -h /var/lib/postgresql/data"

# 测试27: 网络策略测试
run_test "网络策略测试" "kubectl get networkpolicy -n ${NAMESPACE}"

echo -e "\n${PURPLE}📋 9. 高可用性测试${NC}"

# 测试28: Pod 重启恢复测试
echo -e "${YELLOW}🔍 测试: Pod重启恢复测试${NC}"
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# 记录重启前的数据
kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli -a Redis2025!@# set restart_test "before_restart"
original_value=$(kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli -a Redis2025!@# get restart_test)

# 重启Redis Pod
kubectl delete pod redis-0 -n ${NAMESPACE} --wait=false
sleep 10

# 等待Pod重新启动
if wait_for_pod redis-0 120; then
    # 检查数据是否恢复
    recovered_value=$(kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli -a Redis2025!@# get restart_test)
    if [ "$recovered_value" = "$original_value" ]; then
        echo -e "${GREEN}✅ Pod重启恢复测试 - 通过${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ Pod重启恢复测试 - 失败 (数据未恢复)${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
else
    echo -e "${RED}❌ Pod重启恢复测试 - 失败 (Pod未能重启)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# 清理测试数据
kubectl exec -n ${NAMESPACE} redis-0 -c redis -- redis-cli -a Redis2025!@# del restart_test

echo -e "\n${PURPLE}📋 10. 监控和日志测试${NC}"

# 测试29: 日志输出测试
run_test "PostgreSQL日志测试" "kubectl logs postgresql-0 -n ${NAMESPACE} -c postgresql --tail=5"

# 测试30: 监控指标可用性测试
run_test "监控指标可用性测试" "kubectl get pods -n ${NAMESPACE} -l 'prometheus.io/scrape=true'"

echo -e "\n${PURPLE}📋 11. 问题服务诊断${NC}"

# 检查Milvus状态
echo -e "${YELLOW}🔍 诊断: Milvus服务状态${NC}"
kubectl describe pod milvus-0 -n ${NAMESPACE} | tail -10

# 检查NATS状态
echo -e "${YELLOW}🔍 诊断: NATS服务状态${NC}"
kubectl describe pod nats-0 -n ${NAMESPACE} | tail -10

# 生成测试报告
echo -e "\n${BLUE}📊 测试报告${NC}"
echo -e "=================================="
echo -e "总测试数: ${TOTAL_TESTS}"
echo -e "通过测试: ${GREEN}${PASSED_TESTS}${NC}"
echo -e "失败测试: ${RED}${FAILED_TESTS}${NC}"
echo -e "成功率: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 所有测试通过！${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  部分测试失败，请检查上述详情${NC}"
    exit 1
fi
