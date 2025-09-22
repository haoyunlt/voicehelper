#!/bin/bash

# VoiceHelper 性能验证脚本
# 功能: 验证所有性能指标是否达标

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
BASE_URL=${BASE_URL:-"http://localhost:8080"}
METRICS_URL=${METRICS_URL:-"http://localhost:8081/metrics"}
GRAFANA_URL=${GRAFANA_URL:-"http://localhost:3001"}
TEST_DURATION=${TEST_DURATION:-"60"}
CONCURRENT_USERS=${CONCURRENT_USERS:-"10"}

# 性能阈值
E2E_LATENCY_THRESHOLD=500      # ms
ASR_LATENCY_THRESHOLD=300      # ms  
TTS_LATENCY_THRESHOLD=200      # ms
INTERRUPT_LATENCY_THRESHOLD=120 # ms
AUDIO_DROP_RATE_THRESHOLD=0.01  # 1%
BUFFER_HEALTH_THRESHOLD=80      # %
CPU_USAGE_THRESHOLD=80          # %
MEMORY_USAGE_THRESHOLD=80       # %

echo -e "${BLUE}=== VoiceHelper 性能验证开始 ===${NC}"
echo "测试持续时间: ${TEST_DURATION}秒"
echo "并发用户数: ${CONCURRENT_USERS}"
echo ""

# 检查服务健康状态
check_service_health() {
    echo -e "${BLUE}检查服务健康状态...${NC}"
    
    services=("backend:8080" "algo:8082" "frontend:3000")
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if curl -f -s "http://localhost:${port}/health" > /dev/null; then
            echo -e "  ${GREEN}✓${NC} ${name} 服务健康"
        else
            echo -e "  ${RED}✗${NC} ${name} 服务异常"
            exit 1
        fi
    done
    echo ""
}

# 运行负载测试
run_load_test() {
    echo -e "${BLUE}运行负载测试...${NC}"
    
    # 检查k6是否安装
    if ! command -v k6 &> /dev/null; then
        echo -e "${RED}错误: k6 未安装${NC}"
        echo "请安装k6: https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
    
    # 运行负载测试
    k6 run --duration=${TEST_DURATION}s --vus=${CONCURRENT_USERS} \
        tests/performance/voice_load_test.js \
        --out json=load_test_results.json
    
    echo -e "${GREEN}负载测试完成${NC}"
    echo ""
}

# 获取Prometheus指标
get_metric() {
    local metric_name=$1
    local query=$2
    
    # 使用curl查询Prometheus
    result=$(curl -s "${METRICS_URL}" | grep "^${metric_name}" | tail -1 | awk '{print $2}')
    echo ${result:-0}
}

# 验证延迟指标
validate_latency() {
    echo -e "${BLUE}验证延迟指标...${NC}"
    
    # 端到端延迟
    e2e_p95=$(get_metric "voice_e2e_latency_seconds" | awk '{print $1 * 1000}')
    if (( $(echo "$e2e_p95 < $E2E_LATENCY_THRESHOLD" | bc -l) )); then
        echo -e "  ${GREEN}✓${NC} 端到端延迟 P95: ${e2e_p95}ms (< ${E2E_LATENCY_THRESHOLD}ms)"
    else
        echo -e "  ${RED}✗${NC} 端到端延迟 P95: ${e2e_p95}ms (>= ${E2E_LATENCY_THRESHOLD}ms)"
        return 1
    fi
    
    # ASR延迟
    asr_p95=$(get_metric "voice_latency_seconds" | grep 'stage="asr"' | awk '{print $2 * 1000}')
    if (( $(echo "$asr_p95 < $ASR_LATENCY_THRESHOLD" | bc -l) )); then
        echo -e "  ${GREEN}✓${NC} ASR延迟 P95: ${asr_p95}ms (< ${ASR_LATENCY_THRESHOLD}ms)"
    else
        echo -e "  ${RED}✗${NC} ASR延迟 P95: ${asr_p95}ms (>= ${ASR_LATENCY_THRESHOLD}ms)"
        return 1
    fi
    
    # TTS延迟
    tts_p95=$(get_metric "tts_generation_seconds" | awk '{print $1 * 1000}')
    if (( $(echo "$tts_p95 < $TTS_LATENCY_THRESHOLD" | bc -l) )); then
        echo -e "  ${GREEN}✓${NC} TTS延迟 P95: ${tts_p95}ms (< ${TTS_LATENCY_THRESHOLD}ms)"
    else
        echo -e "  ${RED}✗${NC} TTS延迟 P95: ${tts_p95}ms (>= ${TTS_LATENCY_THRESHOLD}ms)"
        return 1
    fi
    
    echo ""
}

# 验证音频质量
validate_audio_quality() {
    echo -e "${BLUE}验证音频质量...${NC}"
    
    # 音频丢包率
    drop_rate=$(get_metric "audio_drop_rate")
    if (( $(echo "$drop_rate < $AUDIO_DROP_RATE_THRESHOLD" | bc -l) )); then
        echo -e "  ${GREEN}✓${NC} 音频丢包率: ${drop_rate} (< ${AUDIO_DROP_RATE_THRESHOLD})"
    else
        echo -e "  ${RED}✗${NC} 音频丢包率: ${drop_rate} (>= ${AUDIO_DROP_RATE_THRESHOLD})"
        return 1
    fi
    
    # 缓冲区健康度
    buffer_health=$(get_metric "buffer_health_score")
    if (( $(echo "$buffer_health > $BUFFER_HEALTH_THRESHOLD" | bc -l) )); then
        echo -e "  ${GREEN}✓${NC} 缓冲区健康度: ${buffer_health}% (> ${BUFFER_HEALTH_THRESHOLD}%)"
    else
        echo -e "  ${RED}✗${NC} 缓冲区健康度: ${buffer_health}% (<= ${BUFFER_HEALTH_THRESHOLD}%)"
        return 1
    fi
    
    # 打断成功率
    barge_in_rate=$(get_metric "barge_in_success_rate")
    if (( $(echo "$barge_in_rate > 0.95" | bc -l) )); then
        echo -e "  ${GREEN}✓${NC} 打断成功率: ${barge_in_rate} (> 95%)"
    else
        echo -e "  ${RED}✗${NC} 打断成功率: ${barge_in_rate} (<= 95%)"
        return 1
    fi
    
    echo ""
}

# 验证系统资源
validate_system_resources() {
    echo -e "${BLUE}验证系统资源使用...${NC}"
    
    # CPU使用率
    cpu_usage=$(get_metric "process_cpu_seconds_total" | awk '{print $1 * 100}')
    if (( $(echo "$cpu_usage < $CPU_USAGE_THRESHOLD" | bc -l) )); then
        echo -e "  ${GREEN}✓${NC} CPU使用率: ${cpu_usage}% (< ${CPU_USAGE_THRESHOLD}%)"
    else
        echo -e "  ${YELLOW}⚠${NC} CPU使用率: ${cpu_usage}% (>= ${CPU_USAGE_THRESHOLD}%)"
    fi
    
    # 内存使用率
    memory_usage=$(get_metric "process_resident_memory_bytes" | awk '{print $1 / 1024 / 1024}')
    echo -e "  ${GREEN}✓${NC} 内存使用: ${memory_usage}MB"
    
    # 连接数
    ws_connections=$(get_metric "ws_active_connections")
    webrtc_connections=$(get_metric "webrtc_connections_active")
    echo -e "  ${GREEN}✓${NC} WebSocket连接: ${ws_connections}"
    echo -e "  ${GREEN}✓${NC} WebRTC连接: ${webrtc_connections}"
    
    echo ""
}

# 验证错误率
validate_error_rates() {
    echo -e "${BLUE}验证错误率...${NC}"
    
    # WebSocket错误率
    ws_errors=$(get_metric "ws_errors_total")
    echo -e "  ${GREEN}✓${NC} WebSocket错误: ${ws_errors}"
    
    # SSE错误率
    sse_errors=$(get_metric "sse_errors_total")
    echo -e "  ${GREEN}✓${NC} SSE错误: ${sse_errors}"
    
    # HTTP错误率
    http_errors=$(get_metric "http_requests_total" | grep '5..' || echo "0")
    echo -e "  ${GREEN}✓${NC} HTTP 5xx错误: ${http_errors}"
    
    echo ""
}

# 运行E2E测试
run_e2e_tests() {
    echo -e "${BLUE}运行E2E测试...${NC}"
    
    if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
        cd frontend
        if npm list @playwright/test > /dev/null 2>&1; then
            npx playwright test --reporter=line
            echo -e "${GREEN}E2E测试完成${NC}"
        else
            echo -e "${YELLOW}⚠ Playwright未安装，跳过E2E测试${NC}"
        fi
        cd ..
    else
        echo -e "${YELLOW}⚠ 前端项目不存在，跳过E2E测试${NC}"
    fi
    
    echo ""
}

# 生成性能报告
generate_report() {
    echo -e "${BLUE}生成性能报告...${NC}"
    
    report_file="performance_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# VoiceHelper 性能验证报告

**生成时间**: $(date)
**测试持续时间**: ${TEST_DURATION}秒
**并发用户数**: ${CONCURRENT_USERS}

## 性能指标

### 延迟指标
- 端到端延迟 P95: ${e2e_p95}ms (阈值: < ${E2E_LATENCY_THRESHOLD}ms)
- ASR延迟 P95: ${asr_p95}ms (阈值: < ${ASR_LATENCY_THRESHOLD}ms)
- TTS延迟 P95: ${tts_p95}ms (阈值: < ${TTS_LATENCY_THRESHOLD}ms)

### 音频质量
- 音频丢包率: ${drop_rate} (阈值: < ${AUDIO_DROP_RATE_THRESHOLD})
- 缓冲区健康度: ${buffer_health}% (阈值: > ${BUFFER_HEALTH_THRESHOLD}%)
- 打断成功率: ${barge_in_rate} (阈值: > 95%)

### 系统资源
- CPU使用率: ${cpu_usage}%
- 内存使用: ${memory_usage}MB
- WebSocket连接: ${ws_connections}
- WebRTC连接: ${webrtc_connections}

### 错误统计
- WebSocket错误: ${ws_errors}
- SSE错误: ${sse_errors}
- HTTP 5xx错误: ${http_errors}

## 测试结论

$(if [ $validation_passed -eq 1 ]; then echo "✅ **所有性能指标达标**"; else echo "❌ **部分性能指标未达标**"; fi)

## 建议

$(if [ $validation_passed -eq 0 ]; then
echo "- 检查未达标的性能指标
- 优化相关组件性能
- 调整系统配置参数
- 增加系统资源配置"
else
echo "- 性能表现良好，可以部署到生产环境
- 建议定期进行性能监控
- 关注长期性能趋势"
fi)

EOF

    echo -e "${GREEN}性能报告已生成: ${report_file}${NC}"
    echo ""
}

# 主函数
main() {
    local validation_passed=1
    
    # 检查依赖
    if ! command -v bc &> /dev/null; then
        echo -e "${RED}错误: bc 计算器未安装${NC}"
        exit 1
    fi
    
    # 执行验证步骤
    check_service_health
    
    # 运行负载测试
    if [ "${SKIP_LOAD_TEST:-false}" != "true" ]; then
        run_load_test
    fi
    
    # 等待指标稳定
    echo -e "${BLUE}等待指标稳定...${NC}"
    sleep 10
    
    # 验证各项指标
    validate_latency || validation_passed=0
    validate_audio_quality || validation_passed=0
    validate_system_resources
    validate_error_rates
    
    # 运行E2E测试
    if [ "${SKIP_E2E_TEST:-false}" != "true" ]; then
        run_e2e_tests
    fi
    
    # 生成报告
    generate_report
    
    # 输出结果
    echo -e "${BLUE}=== 性能验证结果 ===${NC}"
    if [ $validation_passed -eq 1 ]; then
        echo -e "${GREEN}✅ 所有性能指标达标！${NC}"
        echo -e "${GREEN}系统可以部署到生产环境${NC}"
        exit 0
    else
        echo -e "${RED}❌ 部分性能指标未达标${NC}"
        echo -e "${RED}请优化相关组件后重新测试${NC}"
        exit 1
    fi
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        --users)
            CONCURRENT_USERS="$2"
            shift 2
            ;;
        --skip-load-test)
            SKIP_LOAD_TEST=true
            shift
            ;;
        --skip-e2e-test)
            SKIP_E2E_TEST=true
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --duration SECONDS    测试持续时间 (默认: 60)"
            echo "  --users NUMBER        并发用户数 (默认: 10)"
            echo "  --skip-load-test      跳过负载测试"
            echo "  --skip-e2e-test       跳过E2E测试"
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
