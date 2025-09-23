#!/bin/bash

# VoiceHelper 测试运行脚本
# 提供便捷的测试执行命令

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTING_DIR="$PROJECT_ROOT/tools/testing"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "VoiceHelper 测试运行脚本"
    echo ""
    echo "用法: $0 [选项] [测试类型]"
    echo ""
    echo "测试类型:"
    echo "  all                    运行所有测试 (默认)"
    echo "  critical               只运行关键测试"
    echo "  unit                   运行单元测试"
    echo "  integration            运行集成测试"
    echo "  performance            运行性能测试"
    echo "  e2e                    运行端到端测试"
    echo "  security               运行安全测试"
    echo "  voice                  运行语音测试"
    echo "  multimodal             运行多模态测试"
    echo ""
    echo "选项:"
    echo "  -h, --help             显示此帮助信息"
    echo "  -v, --verbose          详细输出"
    echo "  -q, --quiet            静默模式"
    echo "  --no-html              不生成HTML报告"
    echo "  --no-json              不生成JSON报告"
    echo "  --output-dir DIR       指定输出目录"
    echo "  --timeout SECONDS      设置超时时间"
    echo ""
    echo "示例:"
    echo "  $0                     # 运行所有测试"
    echo "  $0 critical            # 只运行关键测试"
    echo "  $0 unit -v             # 详细模式运行单元测试"
    echo "  $0 performance --timeout 600  # 运行性能测试，超时10分钟"
}

# 检查依赖
check_dependencies() {
    print_info "检查测试依赖..."
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        print_error "需要Python 3.8+，当前版本: $python_version"
        exit 1
    fi
    
    # 检查pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        print_warning "pytest 未安装，正在安装测试依赖..."
        pip3 install -r "$PROJECT_ROOT/requirements-test.txt" || {
            print_error "安装测试依赖失败"
            exit 1
        }
    fi
    
    print_success "依赖检查通过"
}

# 设置测试环境
setup_test_env() {
    print_info "设置测试环境..."
    
    # 创建报告目录
    mkdir -p "$PROJECT_ROOT/reports/testing"
    
    # 设置环境变量
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export TEST_MODE=1
    
    print_success "测试环境设置完成"
}

# 运行单个测试套件
run_single_suite() {
    local suite_name=$1
    local suite_path=$2
    local verbose=$3
    
    print_info "运行测试套件: $suite_name"
    
    if [[ ! -f "$suite_path" ]]; then
        print_warning "测试文件不存在: $suite_path"
        return 1
    fi
    
    local pytest_args=("$suite_path")
    
    if [[ $verbose -eq 1 ]]; then
        pytest_args+=("-v" "-s")
    else
        pytest_args+=("--tb=short")
    fi
    
    pytest_args+=("--timeout=300")
    
    if python3 -m pytest "${pytest_args[@]}"; then
        print_success "测试套件 $suite_name 通过"
        return 0
    else
        print_error "测试套件 $suite_name 失败"
        return 1
    fi
}

# 运行综合测试
run_comprehensive_tests() {
    local test_type=$1
    local verbose=$2
    local quiet=$3
    local no_html=$4
    local no_json=$5
    local output_dir=$6
    local timeout=$7
    
    print_info "运行综合测试: $test_type"
    
    local runner_args=()
    
    case $test_type in
        "critical")
            runner_args+=("--critical-only")
            ;;
        "unit")
            runner_args+=("--suites" "security" "error_handling" "voice_processing" "multimodal_fusion")
            ;;
        "integration")
            runner_args+=("--suites" "service_integration")
            ;;
        "performance")
            runner_args+=("--suites" "performance")
            ;;
        "e2e")
            runner_args+=("--suites" "business_workflows")
            ;;
        "security")
            runner_args+=("--suites" "security")
            ;;
        "voice")
            runner_args+=("--suites" "voice_processing")
            ;;
        "multimodal")
            runner_args+=("--suites" "multimodal_fusion")
            ;;
        "all"|*)
            # 运行所有测试，不添加额外参数
            ;;
    esac
    
    if [[ $no_html -eq 1 ]]; then
        runner_args+=("--no-html")
    fi
    
    if [[ $no_json -eq 1 ]]; then
        runner_args+=("--no-json")
    fi
    
    if [[ -n $output_dir ]]; then
        runner_args+=("--output-dir" "$output_dir")
    fi
    
    # 运行综合测试运行器
    if python3 "$TESTING_DIR/test_runner_comprehensive.py" "${runner_args[@]}"; then
        print_success "综合测试完成"
        return 0
    else
        print_error "综合测试失败"
        return 1
    fi
}

# 主函数
main() {
    local test_type="all"
    local verbose=0
    local quiet=0
    local no_html=0
    local no_json=0
    local output_dir=""
    local timeout=600
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                verbose=1
                shift
                ;;
            -q|--quiet)
                quiet=1
                shift
                ;;
            --no-html)
                no_html=1
                shift
                ;;
            --no-json)
                no_json=1
                shift
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --timeout)
                timeout="$2"
                shift 2
                ;;
            all|critical|unit|integration|performance|e2e|security|voice|multimodal)
                test_type="$1"
                shift
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 静默模式下重定向输出
    if [[ $quiet -eq 1 ]]; then
        exec 1>/dev/null
    fi
    
    print_info "开始VoiceHelper测试执行"
    print_info "测试类型: $test_type"
    
    # 检查依赖
    check_dependencies
    
    # 设置测试环境
    setup_test_env
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 运行测试
    local exit_code=0
    
    case $test_type in
        "unit"|"integration"|"performance"|"e2e"|"all"|"critical"|"security"|"voice"|"multimodal")
            run_comprehensive_tests "$test_type" $verbose $quiet $no_html $no_json "$output_dir" $timeout
            exit_code=$?
            ;;
        *)
            print_error "不支持的测试类型: $test_type"
            exit_code=1
            ;;
    esac
    
    # 计算执行时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "所有测试完成，耗时: ${duration}秒"
        
        # 显示报告位置
        if [[ $no_html -eq 0 ]]; then
            report_path="$PROJECT_ROOT/reports/testing/comprehensive_test_report.html"
            if [[ -f "$report_path" ]]; then
                print_info "HTML报告: $report_path"
            fi
        fi
        
        if [[ $no_json -eq 0 ]]; then
            json_path="$PROJECT_ROOT/reports/testing/comprehensive_test_report.json"
            if [[ -f "$json_path" ]]; then
                print_info "JSON报告: $json_path"
            fi
        fi
    else
        print_error "测试执行失败，耗时: ${duration}秒"
    fi
    
    exit $exit_code
}

# 运行主函数
main "$@"
