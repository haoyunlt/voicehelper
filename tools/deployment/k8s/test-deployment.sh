#!/bin/bash

# VoiceHelper Kubernetes 部署测试脚本
# 版本: 2.0.0

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置变量
NAMESPACE="voicehelper"
STORAGE_NAMESPACE="voicehelper-storage"
MONITORING_NAMESPACE="voicehelper-monitoring"
TEST_TIMEOUT=300
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
}

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 记录测试结果
record_test() {
    local test_name="$1"
    local result="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [[ "$result" == "PASS" ]]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        success "✓ $test_name"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        error "✗ $test_name"
    fi
}

# 等待Pod就绪
wait_for_pod() {
    local pod_selector="$1"
    local namespace="$2"
    local timeout="${3:-300}"
    
    info "等待Pod就绪: $pod_selector in $namespace"
    
    if kubectl wait --for=condition=ready pod -l "$pod_selector" -n "$namespace" --timeout="${timeout}s" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 测试HTTP端点
test_http_endpoint() {
    local service_name="$1"
    local namespace="$2"
    local port="$3"
    local path="$4"
    local expected_status="${5:-200}"
    
    local service_ip
    service_ip=$(kubectl get service "$service_name" -n "$namespace" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    
    if [[ -z "$service_ip" ]]; then
        return 1
    fi
    
    local url="http://$service_ip:$port$path"
    local status_code
    status_code=$(kubectl run test-curl-$(date +%s) --rm -i --restart=Never --image=curlimages/curl:latest -- curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [[ "$status_code" == "$expected_status" ]]; then
        return 0
    else
        return 1
    fi
}

# 测试数据库连接
test_database_connection() {
    local db_service="$1"
    local namespace="$2"
    local db_name="$3"
    local username="$4"
    
    info "测试数据库连接: $db_service"
    
    local test_pod="test-db-$(date +%s)"
    
    if kubectl run "$test_pod" --rm -i --restart=Never --image=postgres:15-alpine -n "$namespace" -- psql -h "$db_service" -U "$username" -d "$db_name" -c "SELECT 1;" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 测试Redis连接
test_redis_connection() {
    local redis_service="$1"
    local namespace="$2"
    
    info "测试Redis连接: $redis_service"
    
    local test_pod="test-redis-$(date +%s)"
    
    if kubectl run "$test_pod" --rm -i --restart=Never --image=redis:7-alpine -n "$namespace" -- redis-cli -h "$redis_service" ping &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 测试BGE服务
test_bge_service() {
    log "测试BGE向量化服务..."
    
    # 检查Pod状态
    if wait_for_pod "app.kubernetes.io/name=bge-service" "$NAMESPACE" 60; then
        record_test "BGE Service Pod Ready" "PASS"
    else
        record_test "BGE Service Pod Ready" "FAIL"
        return
    fi
    
    # 测试健康检查端点
    if test_http_endpoint "bge-service" "$NAMESPACE" "8080" "/health"; then
        record_test "BGE Service Health Check" "PASS"
    else
        record_test "BGE Service Health Check" "FAIL"
    fi
    
    # 测试就绪检查端点
    if test_http_endpoint "bge-service" "$NAMESPACE" "8080" "/ready"; then
        record_test "BGE Service Ready Check" "PASS"
    else
        record_test "BGE Service Ready Check" "FAIL"
    fi
    
    # 测试向量化API
    local test_pod="test-bge-$(date +%s)"
    local bge_ip
    bge_ip=$(kubectl get service bge-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    if kubectl run "$test_pod" --rm -i --restart=Never --image=curlimages/curl:latest -- \
        curl -s -X POST "http://$bge_ip:8080/embed" \
        -H "Content-Type: application/json" \
        -d '{"texts": ["测试文本"]}' | grep -q "embeddings"; then
        record_test "BGE Service Embedding API" "PASS"
    else
        record_test "BGE Service Embedding API" "FAIL"
    fi
}

# 测试FAISS服务
test_faiss_service() {
    log "测试FAISS向量搜索服务..."
    
    # 检查Pod状态
    if wait_for_pod "app.kubernetes.io/name=faiss-service" "$NAMESPACE" 60; then
        record_test "FAISS Service Pod Ready" "PASS"
    else
        record_test "FAISS Service Pod Ready" "FAIL"
        return
    fi
    
    # 测试健康检查端点
    if test_http_endpoint "faiss-client" "$NAMESPACE" "8081" "/health"; then
        record_test "FAISS Service Health Check" "PASS"
    else
        record_test "FAISS Service Health Check" "FAIL"
    fi
    
    # 测试就绪检查端点
    if test_http_endpoint "faiss-client" "$NAMESPACE" "8081" "/ready"; then
        record_test "FAISS Service Ready Check" "PASS"
    else
        record_test "FAISS Service Ready Check" "FAIL"
    fi
    
    # 测试统计信息API
    if test_http_endpoint "faiss-client" "$NAMESPACE" "8081" "/stats"; then
        record_test "FAISS Service Stats API" "PASS"
    else
        record_test "FAISS Service Stats API" "FAIL"
    fi
}

# 测试Gateway服务
test_gateway_service() {
    log "测试Gateway API网关服务..."
    
    # 检查Pod状态
    if wait_for_pod "app.kubernetes.io/name=gateway" "$NAMESPACE" 60; then
        record_test "Gateway Service Pod Ready" "PASS"
    else
        record_test "Gateway Service Pod Ready" "FAIL"
        return
    fi
    
    # 测试健康检查端点
    if test_http_endpoint "gateway" "$NAMESPACE" "8080" "/health"; then
        record_test "Gateway Service Health Check" "PASS"
    else
        record_test "Gateway Service Health Check" "FAIL"
    fi
    
    # 测试就绪检查端点
    if test_http_endpoint "gateway" "$NAMESPACE" "8080" "/ready"; then
        record_test "Gateway Service Ready Check" "PASS"
    else
        record_test "Gateway Service Ready Check" "FAIL"
    fi
    
    # 测试API路由
    if test_http_endpoint "gateway" "$NAMESPACE" "8080" "/api/v1/health"; then
        record_test "Gateway API Routing" "PASS"
    else
        record_test "Gateway API Routing" "FAIL"
    fi
}

# 测试存储服务
test_storage_services() {
    log "测试存储服务..."
    
    # 测试PostgreSQL
    if wait_for_pod "app.kubernetes.io/name=postgres,app.kubernetes.io/component=master" "$STORAGE_NAMESPACE" 60; then
        record_test "PostgreSQL Master Pod Ready" "PASS"
        
        if test_database_connection "postgres-master" "$STORAGE_NAMESPACE" "voicehelper" "voicehelper"; then
            record_test "PostgreSQL Connection Test" "PASS"
        else
            record_test "PostgreSQL Connection Test" "FAIL"
        fi
    else
        record_test "PostgreSQL Master Pod Ready" "FAIL"
    fi
    
    # 测试Redis
    if wait_for_pod "app.kubernetes.io/name=redis" "$STORAGE_NAMESPACE" 60; then
        record_test "Redis Cluster Pod Ready" "PASS"
        
        if test_redis_connection "redis" "$STORAGE_NAMESPACE"; then
            record_test "Redis Connection Test" "PASS"
        else
            record_test "Redis Connection Test" "FAIL"
        fi
    else
        record_test "Redis Cluster Pod Ready" "FAIL"
    fi
    
    # 测试MinIO
    if wait_for_pod "app.kubernetes.io/name=minio" "$STORAGE_NAMESPACE" 60; then
        record_test "MinIO Pod Ready" "PASS"
        
        if test_http_endpoint "minio-client" "$STORAGE_NAMESPACE" "9000" "/minio/health/live"; then
            record_test "MinIO Health Check" "PASS"
        else
            record_test "MinIO Health Check" "FAIL"
        fi
    else
        record_test "MinIO Pod Ready" "FAIL"
    fi
}

# 测试监控服务
test_monitoring_services() {
    log "测试监控服务..."
    
    # 测试Prometheus
    if kubectl get deployment prometheus -n "$MONITORING_NAMESPACE" &>/dev/null; then
        if wait_for_pod "app.kubernetes.io/name=prometheus" "$MONITORING_NAMESPACE" 60; then
            record_test "Prometheus Pod Ready" "PASS"
            
            if test_http_endpoint "prometheus" "$MONITORING_NAMESPACE" "9090" "/-/healthy"; then
                record_test "Prometheus Health Check" "PASS"
            else
                record_test "Prometheus Health Check" "FAIL"
            fi
        else
            record_test "Prometheus Pod Ready" "FAIL"
        fi
    else
        warning "Prometheus未部署，跳过测试"
    fi
    
    # 测试Grafana
    if kubectl get deployment grafana -n "$MONITORING_NAMESPACE" &>/dev/null; then
        if wait_for_pod "app.kubernetes.io/name=grafana" "$MONITORING_NAMESPACE" 60; then
            record_test "Grafana Pod Ready" "PASS"
            
            if test_http_endpoint "grafana" "$MONITORING_NAMESPACE" "3000" "/api/health"; then
                record_test "Grafana Health Check" "PASS"
            else
                record_test "Grafana Health Check" "FAIL"
            fi
        else
            record_test "Grafana Pod Ready" "FAIL"
        fi
    else
        warning "Grafana未部署，跳过测试"
    fi
}

# 测试网络连通性
test_network_connectivity() {
    log "测试网络连通性..."
    
    # 测试服务间通信
    local test_pod="test-network-$(date +%s)"
    
    # 创建测试Pod
    kubectl run "$test_pod" --image=curlimages/curl:latest -n "$NAMESPACE" --restart=Never -- sleep 3600
    
    # 等待Pod就绪
    kubectl wait --for=condition=ready pod "$test_pod" -n "$NAMESPACE" --timeout=60s
    
    # 测试Gateway到BGE服务的连通性
    if kubectl exec "$test_pod" -n "$NAMESPACE" -- curl -s -f "http://bge-service.voicehelper.svc.cluster.local:8080/health" &>/dev/null; then
        record_test "Gateway to BGE Service Connectivity" "PASS"
    else
        record_test "Gateway to BGE Service Connectivity" "FAIL"
    fi
    
    # 测试Gateway到FAISS服务的连通性
    if kubectl exec "$test_pod" -n "$NAMESPACE" -- curl -s -f "http://faiss-client.voicehelper.svc.cluster.local:8081/health" &>/dev/null; then
        record_test "Gateway to FAISS Service Connectivity" "PASS"
    else
        record_test "Gateway to FAISS Service Connectivity" "FAIL"
    fi
    
    # 清理测试Pod
    kubectl delete pod "$test_pod" -n "$NAMESPACE" --ignore-not-found=true
}

# 测试资源使用情况
test_resource_usage() {
    log "测试资源使用情况..."
    
    # 检查CPU和内存使用情况
    info "当前资源使用情况:"
    kubectl top nodes 2>/dev/null || warning "无法获取节点资源使用情况"
    kubectl top pods -n "$NAMESPACE" 2>/dev/null || warning "无法获取Pod资源使用情况"
    
    # 检查是否有资源不足的Pod
    local pending_pods
    pending_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Pending --no-headers 2>/dev/null | wc -l)
    
    if [[ "$pending_pods" -eq 0 ]]; then
        record_test "No Pending Pods" "PASS"
    else
        record_test "No Pending Pods" "FAIL"
        warning "发现 $pending_pods 个待调度的Pod"
    fi
    
    # 检查是否有失败的Pod
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed --no-headers 2>/dev/null | wc -l)
    
    if [[ "$failed_pods" -eq 0 ]]; then
        record_test "No Failed Pods" "PASS"
    else
        record_test "No Failed Pods" "FAIL"
        warning "发现 $failed_pods 个失败的Pod"
    fi
}

# 测试持久化存储
test_persistent_storage() {
    log "测试持久化存储..."
    
    # 检查PVC状态
    local bound_pvcs
    bound_pvcs=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[?(@.status.phase=="Bound")].metadata.name}' | wc -w)
    
    local total_pvcs
    total_pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers | wc -l)
    
    if [[ "$bound_pvcs" -eq "$total_pvcs" ]] && [[ "$total_pvcs" -gt 0 ]]; then
        record_test "All PVCs Bound" "PASS"
    else
        record_test "All PVCs Bound" "FAIL"
        warning "PVC绑定状态: $bound_pvcs/$total_pvcs"
    fi
    
    # 检查存储类
    if kubectl get storageclass voicehelper-ssd &>/dev/null; then
        record_test "SSD Storage Class Available" "PASS"
    else
        record_test "SSD Storage Class Available" "FAIL"
    fi
}

# 性能测试
performance_test() {
    log "执行性能测试..."
    
    # BGE服务性能测试
    local bge_ip
    bge_ip=$(kubectl get service bge-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "$bge_ip" ]]; then
        local test_pod="perf-test-$(date +%s)"
        
        # 创建性能测试Pod
        kubectl run "$test_pod" --image=curlimages/curl:latest -n "$NAMESPACE" --restart=Never -- sleep 3600
        kubectl wait --for=condition=ready pod "$test_pod" -n "$NAMESPACE" --timeout=60s
        
        # 执行并发请求测试
        local start_time=$(date +%s)
        for i in {1..10}; do
            kubectl exec "$test_pod" -n "$NAMESPACE" -- curl -s -X POST "http://$bge_ip:8080/embed" \
                -H "Content-Type: application/json" \
                -d '{"texts": ["性能测试文本'$i'"]}' &
        done
        wait
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [[ "$duration" -lt 30 ]]; then
            record_test "BGE Service Performance (10 concurrent requests < 30s)" "PASS"
        else
            record_test "BGE Service Performance (10 concurrent requests < 30s)" "FAIL"
        fi
        
        # 清理测试Pod
        kubectl delete pod "$test_pod" -n "$NAMESPACE" --ignore-not-found=true
    else
        record_test "BGE Service Performance Test" "FAIL"
    fi
}

# 生成测试报告
generate_test_report() {
    log "生成测试报告..."
    
    local report_file="$SCRIPT_DIR/test-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
VoiceHelper Kubernetes 部署测试报告
=====================================

测试时间: $(date)
测试环境: Kubernetes $(kubectl version --short --client | grep Client | awk '{print $3}')

测试结果统计:
- 总测试数: $TOTAL_TESTS
- 通过测试: $PASSED_TESTS
- 失败测试: $FAILED_TESTS
- 成功率: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

集群信息:
$(kubectl cluster-info)

节点信息:
$(kubectl get nodes -o wide)

命名空间资源:
$(kubectl get all -n $NAMESPACE)

存储命名空间资源:
$(kubectl get all -n $STORAGE_NAMESPACE)

监控命名空间资源:
$(kubectl get all -n $MONITORING_NAMESPACE)

PVC状态:
$(kubectl get pvc -A)

存储类:
$(kubectl get storageclass)

EOF
    
    success "测试报告已生成: $report_file"
}

# 主测试函数
main() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              VoiceHelper Kubernetes 部署测试                 ║"
    echo "║                        版本: 2.0.0                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # 检查Kubernetes连接
    if ! kubectl cluster-info &>/dev/null; then
        error "无法连接到Kubernetes集群"
        exit 1
    fi
    
    # 执行测试
    test_storage_services
    test_bge_service
    test_faiss_service
    test_gateway_service
    test_monitoring_services
    test_network_connectivity
    test_resource_usage
    test_persistent_storage
    performance_test
    
    # 生成报告
    generate_test_report
    
    # 显示测试结果
    echo -e "\n${PURPLE}=== 测试结果汇总 ===${NC}"
    echo -e "总测试数: ${CYAN}$TOTAL_TESTS${NC}"
    echo -e "通过测试: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "失败测试: ${RED}$FAILED_TESTS${NC}"
    echo -e "成功率: ${YELLOW}$(( PASSED_TESTS * 100 / TOTAL_TESTS ))%${NC}"
    
    if [[ "$FAILED_TESTS" -eq 0 ]]; then
        success "🎉 所有测试通过！部署成功！"
        exit 0
    else
        error "❌ 有 $FAILED_TESTS 个测试失败，请检查部署状态"
        exit 1
    fi
}

# 执行主函数
main "$@"
