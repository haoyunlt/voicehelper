#!/usr/bin/env python3
"""
VoiceHelper 测试演示脚本
展示如何运行各种类型的测试并查看结果
"""

import subprocess
import sys
import time
import os
from pathlib import Path


def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"命令: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.stdout:
            print("📤 输出:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ 错误信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 执行成功")
        else:
            print(f"❌ 执行失败 (退出码: {result.returncode})")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
        return False
    except Exception as e:
        print(f"💥 执行异常: {e}")
        return False


def check_environment():
    """检查测试环境"""
    print("🔍 检查测试环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        return False
    
    # 检查必要的包
    required_packages = [
        "pytest", "requests", "aiohttp", "websockets"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements-test.txt")
        return False
    
    return True


def demo_unit_tests():
    """演示单元测试"""
    print("\n" + "="*60)
    print("📋 单元测试演示")
    print("="*60)
    
    # 运行后端单元测试
    if os.path.exists("tests/unit/backend"):
        run_command(
            "python -m pytest tests/unit/backend/test_handlers.py::TestIntegrationHandler::test_list_services_success -v",
            "运行后端处理器测试"
        )
    
    # 运行算法服务单元测试
    if os.path.exists("tests/unit/algo"):
        run_command(
            "python -m pytest tests/unit/algo/test_services.py::TestRetrieveService::test_build_prompt -v",
            "运行算法服务测试"
        )


def demo_integration_tests():
    """演示集成测试"""
    print("\n" + "="*60)
    print("🔗 集成测试演示")
    print("="*60)
    
    # 检查服务是否运行
    print("检查服务状态...")
    backend_running = run_command("curl -s http://localhost:8080/health", "检查后端服务")
    algo_running = run_command("curl -s http://localhost:8000/health", "检查算法服务")
    
    if not (backend_running or algo_running):
        print("⚠️ 服务未运行，集成测试可能失败")
        print("请先启动服务:")
        print("  后端: cd backend && go run cmd/server/main.go")
        print("  算法: cd algo && python app/main.py")
    
    # 运行集成测试
    if os.path.exists("tests/integration"):
        run_command(
            "python -m pytest tests/integration/test_api_endpoints.py::TestBackendAPIIntegration::test_health_check -v -s",
            "运行API集成测试"
        )


def demo_performance_tests():
    """演示性能测试"""
    print("\n" + "="*60)
    print("⚡ 性能测试演示")
    print("="*60)
    
    # 基准测试
    if os.path.exists("scripts/performance/benchmark_test.py"):
        print("运行快速基准测试...")
        # 创建一个简化的基准测试
        benchmark_code = '''
import asyncio
import aiohttp
import time

async def quick_benchmark():
    """快速基准测试"""
    print("开始快速基准测试...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # 测试健康检查
            start_time = time.time()
            async with session.get("http://localhost:8080/health", timeout=5) as response:
                response_time = time.time() - start_time
                print(f"健康检查响应时间: {response_time:.3f}s")
                print(f"状态码: {response.status}")
                
                if response.status == 200:
                    print("✅ 基准测试通过")
                else:
                    print("❌ 基准测试失败")
                    
    except Exception as e:
        print(f"❌ 基准测试异常: {e}")

if __name__ == "__main__":
    asyncio.run(quick_benchmark())
'''
        
        # 写入临时文件并执行
        with open("temp_benchmark.py", "w") as f:
            f.write(benchmark_code)
        
        run_command("python temp_benchmark.py", "快速基准测试")
        
        # 清理临时文件
        if os.path.exists("temp_benchmark.py"):
            os.remove("temp_benchmark.py")


def demo_test_reports():
    """演示测试报告生成"""
    print("\n" + "="*60)
    print("📊 测试报告演示")
    print("="*60)
    
    # 创建报告目录
    os.makedirs("reports", exist_ok=True)
    
    # 生成覆盖率报告
    if os.path.exists("tests/unit"):
        run_command(
            "python -m pytest tests/unit/ --cov=backend --cov=algo --cov-report=html:reports/coverage --cov-report=term -q",
            "生成代码覆盖率报告"
        )
    
    # 生成HTML测试报告
    run_command(
        "python -m pytest tests/ --html=reports/test_report.html --self-contained-html -q",
        "生成HTML测试报告"
    )
    
    print("\n📁 报告文件位置:")
    if os.path.exists("reports/coverage/index.html"):
        print(f"  覆盖率报告: {os.path.abspath('reports/coverage/index.html')}")
    if os.path.exists("reports/test_report.html"):
        print(f"  测试报告: {os.path.abspath('reports/test_report.html')}")


def show_test_structure():
    """显示测试结构"""
    print("\n" + "="*60)
    print("📁 测试目录结构")
    print("="*60)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        """打印目录树"""
        if current_depth >= max_depth:
            return
            
        if not os.path.exists(directory):
            return
            
        items = sorted(os.listdir(directory))
        for i, item in enumerate(items):
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item}")
            
            if os.path.isdir(item_path) and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item_path, next_prefix, max_depth, current_depth + 1)
    
    # 显示测试目录结构
    if os.path.exists("tests"):
        print("tests/")
        print_tree("tests", "", max_depth=4)
    
    # 显示脚本目录结构
    if os.path.exists("scripts"):
        print("\nscripts/")
        print_tree("scripts", "", max_depth=3)


def main():
    """主函数"""
    print("🎯 VoiceHelper 测试演示")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请先安装依赖")
        return
    
    # 显示测试结构
    show_test_structure()
    
    print("\n🚀 开始测试演示...")
    
    # 演示各种测试
    demo_unit_tests()
    demo_integration_tests()
    demo_performance_tests()
    demo_test_reports()
    
    print("\n" + "="*60)
    print("🎉 测试演示完成！")
    print("="*60)
    
    print("\n📚 更多测试命令:")
    print("  ./scripts/run_tests.sh --help        # 查看测试脚本帮助")
    print("  pytest --help                        # 查看pytest帮助")
    print("  python -m pytest tests/ -v           # 运行所有测试")
    print("  python -m pytest -m unit             # 只运行单元测试")
    print("  python -m pytest --cov=backend       # 生成覆盖率报告")
    
    print("\n📖 详细文档:")
    print("  docs/TESTING_GUIDE.md                # 测试指南")
    print("  pytest.ini                           # pytest配置")
    print("  requirements-test.txt                # 测试依赖")


if __name__ == "__main__":
    main()
