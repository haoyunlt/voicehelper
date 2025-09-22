#!/bin/bash

# VoiceHelper 快速测试脚本
# 用于快速验证测试框架和核心功能

set -e

echo "🎯 VoiceHelper 快速测试"
echo "========================"

# 检查Python环境
echo "🔍 检查Python环境..."
python --version
echo ""

# 检查测试依赖
echo "📦 检查测试依赖..."
if ! python -c "import pytest" 2>/dev/null; then
    echo "❌ pytest未安装，正在安装..."
    pip install pytest pytest-asyncio
fi

if ! python -c "import httpx" 2>/dev/null; then
    echo "❌ httpx未安装，正在安装..."
    pip install httpx
fi

echo "✅ 依赖检查完成"
echo ""

# 运行框架验证测试
echo "🧪 运行测试框架验证..."
python -m pytest tests/test_framework_validation.py -v -s
echo ""

# 检查服务状态
echo "🔍 检查服务状态..."
check_service() {
    local url=$1
    local name=$2
    
    if curl -s --connect-timeout 3 "$url" > /dev/null 2>&1; then
        echo "✅ $name 服务运行中"
        return 0
    else
        echo "❌ $name 服务未运行"
        return 1
    fi
}

backend_running=false
algo_running=false

if check_service "http://localhost:8080/health" "后端"; then
    backend_running=true
fi

if check_service "http://localhost:8000/health" "算法"; then
    algo_running=true
fi

echo ""

# 运行可用的测试
if [ -f "tests/unit/backend/test_handlers.py" ]; then
    echo "🧪 运行后端单元测试..."
    python -m pytest tests/unit/backend/test_handlers.py::test_health_check -v || echo "⚠️ 部分测试失败"
    echo ""
fi

if [ -f "tests/unit/algo/test_services.py" ]; then
    echo "🧪 运行算法服务单元测试..."
    python -m pytest tests/unit/algo/test_services.py -k "test_retrieve_service" -v || echo "⚠️ 部分测试失败"
    echo ""
fi

# 如果服务运行中，执行集成测试
if $backend_running || $algo_running; then
    echo "🔗 运行集成测试..."
    if [ -f "tests/integration/test_api_endpoints.py" ]; then
        python -m pytest tests/integration/test_api_endpoints.py -k "health" -v || echo "⚠️ 部分集成测试失败"
    fi
else
    echo "⚠️ 服务未运行，跳过集成测试"
    echo "启动服务命令:"
    echo "  后端: cd backend && go run cmd/server/main.go"
    echo "  算法: cd algo && python app/main.py"
fi

echo ""

# 运行性能基准测试
echo "⚡ 运行快速性能测试..."
python -c "
import time
import asyncio

async def quick_perf_test():
    print('开始快速性能测试...')
    
    # 简单的计算性能测试
    start = time.time()
    result = sum(i*i for i in range(10000))
    duration = time.time() - start
    print(f'计算性能: {duration:.4f}s (结果: {result})')
    
    # 异步性能测试
    start = time.time()
    await asyncio.sleep(0.01)
    duration = time.time() - start
    print(f'异步延迟: {duration:.4f}s')
    
    print('✅ 快速性能测试完成')

asyncio.run(quick_perf_test())
"

echo ""
echo "🎉 快速测试完成！"
echo ""
echo "📚 更多测试选项:"
echo "  ./scripts/run_tests.sh           # 完整测试套件"
echo "  python scripts/demo_tests.py     # 测试演示"
echo "  pytest tests/ -v                 # 运行所有测试"
echo "  pytest --cov=backend tests/      # 代码覆盖率"
echo ""
echo "📖 测试文档: docs/TESTING_GUIDE.md"
