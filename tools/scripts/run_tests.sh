#!/bin/bash

# VoiceHelper 测试运行脚本
# 用于执行各种类型的测试：单元测试、集成测试、端到端测试、性能测试

set -e

# 颜色定义
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

# 检查依赖
check_dependencies() {
    log_info "检查测试依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        log_warning "pytest 未安装，正在安装..."
        pip3 install pytest pytest-asyncio pytest-cov
    fi
    
    # 检查其他依赖
    local deps=("requests" "aiohttp" "websockets" "locust" "selenium" "psutil")
    for dep in "${deps[@]}"; do
        if ! python3 -c "import $dep" &> /dev/null; then
            log_warning "$dep 未安装，正在安装..."
            pip3 install $dep
        fi
    done
    
    log_success "依赖检查完成"
}

# 检查服务状态
check_services() {
    log_info "检查服务状态..."
    
    # 检查后端服务
    if curl -s http://localhost:8080/health > /dev/null; then
        log_success "后端服务运行正常"
    else
        log_warning "后端服务未运行 (localhost:8080)"
    fi
    
    # 检查算法服务
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "算法服务运行正常"
    else
        log_warning "算法服务未运行 (localhost:8000)"
    fi
    
    # 检查前端服务
    if curl -s http://localhost:3000 > /dev/null; then
        log_success "前端服务运行正常"
    else
        log_warning "前端服务未运行 (localhost:3000)"
    fi
}

# 运行单元测试
run_unit_tests() {
    log_info "运行单元测试..."
    
    # 后端单元测试
    if [ -d "tests/unit/backend" ]; then
        log_info "执行后端单元测试..."
        python3 -m pytest tests/unit/backend/ -v --tb=short --cov=backend --cov-report=html:reports/backend_coverage
    fi
    
    # 算法服务单元测试
    if [ -d "tests/unit/algo" ]; then
        log_info "执行算法服务单元测试..."
        python3 -m pytest tests/unit/algo/ -v --tb=short --cov=algo --cov-report=html:reports/algo_coverage
    fi
    
    log_success "单元测试完成"
}

# 运行集成测试
run_integration_tests() {
    log_info "运行集成测试..."
    
    if [ -d "tests/integration" ]; then
        python3 -m pytest tests/integration/ -v --tb=short -s
    else
        log_warning "集成测试目录不存在"
    fi
    
    log_success "集成测试完成"
}

# 运行端到端测试
run_e2e_tests() {
    log_info "运行端到端测试..."
    
    # 检查Chrome浏览器（用于Selenium）
    if ! command -v google-chrome &> /dev/null && ! command -v chromium-browser &> /dev/null; then
        log_warning "Chrome浏览器未安装，跳过Web UI测试"
    fi
    
    if [ -d "tests/e2e" ]; then
        python3 -m pytest tests/e2e/ -v --tb=short -s --maxfail=5
    else
        log_warning "端到端测试目录不存在"
    fi
    
    log_success "端到端测试完成"
}

# 运行性能测试
run_performance_tests() {
    log_info "运行性能测试..."
    
    local test_type=${1:-"benchmark"}
    
    case $test_type in
        "benchmark")
            log_info "执行基准测试..."
            python3 scripts/performance/benchmark_test.py
            ;;
        "load")
            log_info "执行负载测试..."
            if command -v locust &> /dev/null; then
                locust -f scripts/performance/load_test.py --host http://localhost:8080 --users 50 --spawn-rate 5 --run-time 5m --headless
            else
                log_error "Locust 未安装，无法执行负载测试"
                return 1
            fi
            ;;
        "stress")
            log_info "执行压力测试..."
            python3 scripts/performance/stress_test.py
            ;;
        *)
            log_error "未知的性能测试类型: $test_type"
            return 1
            ;;
    esac
    
    log_success "性能测试完成"
}

# 生成测试报告
generate_report() {
    log_info "生成测试报告..."
    
    local report_dir="reports"
    mkdir -p $report_dir
    
    # 创建HTML报告
    cat > $report_dir/test_report.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>VoiceHelper 测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007cba; }
        .success { border-left-color: #28a745; }
        .warning { border-left-color: #ffc107; }
        .error { border-left-color: #dc3545; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>VoiceHelper 测试报告</h1>
        <p class="timestamp">生成时间: $(date)</p>
    </div>
    
    <div class="section success">
        <h2>测试执行摘要</h2>
        <p>测试已完成，详细结果请查看各个测试模块的报告。</p>
    </div>
    
    <div class="section">
        <h2>测试覆盖范围</h2>
        <ul>
            <li>单元测试：后端服务、算法引擎</li>
            <li>集成测试：API接口、跨服务调用</li>
            <li>端到端测试：完整业务流程</li>
            <li>性能测试：基准测试、负载测试、压力测试</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>相关文件</h2>
        <ul>
            <li><a href="backend_coverage/index.html">后端代码覆盖率报告</a></li>
            <li><a href="algo_coverage/index.html">算法服务代码覆盖率报告</a></li>
            <li><a href="../tests/">测试用例源码</a></li>
            <li><a href="../scripts/performance/">性能测试脚本</a></li>
        </ul>
    </div>
</body>
</html>
EOF
    
    log_success "测试报告已生成: $report_dir/test_report.html"
}

# 清理测试环境
cleanup() {
    log_info "清理测试环境..."
    
    # 清理临时文件
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # 清理测试数据库（如果有）
    if [ -f "test.db" ]; then
        rm test.db
    fi
    
    log_success "清理完成"
}

# 显示帮助信息
show_help() {
    echo "VoiceHelper 测试运行脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -c, --check             检查依赖和服务状态"
    echo "  -u, --unit              运行单元测试"
    echo "  -i, --integration       运行集成测试"
    echo "  -e, --e2e               运行端到端测试"
    echo "  -p, --performance TYPE  运行性能测试 (benchmark|load|stress)"
    echo "  -a, --all               运行所有测试"
    echo "  -r, --report            生成测试报告"
    echo "  --cleanup               清理测试环境"
    echo ""
    echo "示例:"
    echo "  $0 --check                    # 检查环境"
    echo "  $0 --unit                     # 运行单元测试"
    echo "  $0 --performance benchmark    # 运行基准测试"
    echo "  $0 --all                      # 运行所有测试"
}

# 主函数
main() {
    local run_check=false
    local run_unit=false
    local run_integration=false
    local run_e2e=false
    local run_performance=""
    local run_all=false
    local generate_report_flag=false
    local cleanup_flag=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--check)
                run_check=true
                shift
                ;;
            -u|--unit)
                run_unit=true
                shift
                ;;
            -i|--integration)
                run_integration=true
                shift
                ;;
            -e|--e2e)
                run_e2e=true
                shift
                ;;
            -p|--performance)
                run_performance="$2"
                shift 2
                ;;
            -a|--all)
                run_all=true
                shift
                ;;
            -r|--report)
                generate_report_flag=true
                shift
                ;;
            --cleanup)
                cleanup_flag=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定任何选项，显示帮助
    if [[ $run_check == false && $run_unit == false && $run_integration == false && 
          $run_e2e == false && -z $run_performance && $run_all == false && 
          $generate_report_flag == false && $cleanup_flag == false ]]; then
        show_help
        exit 0
    fi
    
    # 创建报告目录
    mkdir -p reports
    
    # 执行操作
    if [[ $cleanup_flag == true ]]; then
        cleanup
    fi
    
    if [[ $run_check == true || $run_all == true ]]; then
        check_dependencies
        check_services
    fi
    
    if [[ $run_unit == true || $run_all == true ]]; then
        run_unit_tests
    fi
    
    if [[ $run_integration == true || $run_all == true ]]; then
        run_integration_tests
    fi
    
    if [[ $run_e2e == true || $run_all == true ]]; then
        run_e2e_tests
    fi
    
    if [[ -n $run_performance ]]; then
        run_performance_tests "$run_performance"
    elif [[ $run_all == true ]]; then
        run_performance_tests "benchmark"
    fi
    
    if [[ $generate_report_flag == true || $run_all == true ]]; then
        generate_report
    fi
    
    log_success "所有测试任务完成！"
}

# 执行主函数
main "$@"
