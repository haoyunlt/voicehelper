#!/usr/bin/env python3
"""
VoiceHelper 统一测试运行器
整合所有测试类型，提供一站式测试解决方案
"""

import asyncio
import os
import sys
import subprocess
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRunner:
    """统一测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.results = {
            'test_session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
        
    def check_dependencies(self) -> bool:
        """检查测试依赖"""
        logger.info("🔍 检查测试依赖...")
        
        # 检查Python包
        required_packages = [
            'pytest', 'requests', 'psutil', 'aiohttp'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"缺少必要的Python包: {', '.join(missing_packages)}")
            logger.info("请运行: pip install " + " ".join(missing_packages))
            return False
        
        # 检查服务可用性
        services_check = self._check_services()
        if not services_check['all_available']:
            logger.warning("⚠️ 部分服务不可用，某些测试可能会失败")
            for service, status in services_check['services'].items():
                if not status['available']:
                    logger.warning(f"  {service}: {status.get('error', 'Unknown error')}")
        
        return True
    
    def _check_services(self) -> Dict[str, Any]:
        """检查服务状态"""
        import requests
        
        services = {
            'backend': 'http://localhost:8080/health',
            'algorithm': 'http://localhost:8000/health',
            'frontend': 'http://localhost:3000'
        }
        
        results = {'services': {}, 'all_available': True}
        
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                results['services'][service_name] = {
                    'available': response.status_code == 200,
                    'status_code': response.status_code
                }
            except Exception as e:
                results['services'][service_name] = {
                    'available': False,
                    'error': str(e)
                }
                results['all_available'] = False
        
        return results
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        logger.info("🧪 运行单元测试...")
        
        start_time = time.time()
        
        # 查找单元测试文件
        unit_test_files = list(self.tests_dir.glob("**/test_*.py"))
        unit_test_files.extend(list(self.tests_dir.glob("unit/**/*.py")))
        
        if not unit_test_files:
            logger.warning("未找到单元测试文件")
            return {
                'status': 'skipped',
                'reason': 'No unit test files found',
                'duration': 0
            }
        
        try:
            # 使用pytest运行单元测试
            cmd = [
                sys.executable, '-m', 'pytest',
                str(self.tests_dir),
                '-v',
                '--tb=short',
                '--json-report',
                f'--json-report-file={self.tests_dir}/unit_test_results.json'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            duration = time.time() - start_time
            
            # 尝试读取pytest的JSON报告
            json_report_file = self.tests_dir / 'unit_test_results.json'
            test_details = {}
            if json_report_file.exists():
                try:
                    with open(json_report_file, 'r') as f:
                        pytest_report = json.load(f)
                        test_details = {
                            'total': pytest_report.get('summary', {}).get('total', 0),
                            'passed': pytest_report.get('summary', {}).get('passed', 0),
                            'failed': pytest_report.get('summary', {}).get('failed', 0),
                            'skipped': pytest_report.get('summary', {}).get('skipped', 0)
                        }
                except:
                    pass
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'test_details': test_details
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'duration': time.time() - start_time,
                'error': 'Unit tests timed out after 5 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        logger.info("🔗 运行集成测试...")
        
        start_time = time.time()
        
        # 运行模块测试
        module_test_file = self.tests_dir / 'module_test_runner.py'
        if not module_test_file.exists():
            logger.warning("未找到模块测试文件")
            return {
                'status': 'skipped',
                'reason': 'Module test runner not found',
                'duration': 0
            }
        
        try:
            result = subprocess.run(
                [sys.executable, str(module_test_file)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            duration = time.time() - start_time
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'duration': time.time() - start_time,
                'error': 'Integration tests timed out after 5 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    async def run_e2e_tests(self) -> Dict[str, Any]:
        """运行E2E测试"""
        logger.info("🎭 运行E2E测试...")
        
        start_time = time.time()
        
        # 检查E2E测试目录
        e2e_dir = self.tests_dir / 'e2e'
        if not e2e_dir.exists():
            logger.warning("未找到E2E测试目录")
            return {
                'status': 'skipped',
                'reason': 'E2E test directory not found',
                'duration': 0
            }
        
        try:
            # 运行Playwright测试
            result = subprocess.run(
                ['npx', 'playwright', 'test', '--reporter=json'],
                cwd=str(e2e_dir),
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            duration = time.time() - start_time
            
            # 尝试解析Playwright的JSON输出
            test_details = {}
            try:
                if result.stdout:
                    # Playwright的JSON输出可能包含多行，我们需要找到实际的JSON
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.startswith('{') and line.endswith('}'):
                            playwright_report = json.loads(line)
                            if 'stats' in playwright_report:
                                stats = playwright_report['stats']
                                test_details = {
                                    'total': stats.get('total', 0),
                                    'passed': stats.get('passed', 0),
                                    'failed': stats.get('failed', 0),
                                    'skipped': stats.get('skipped', 0)
                                }
                                break
            except:
                pass
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'test_details': test_details
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'duration': time.time() - start_time,
                'error': 'E2E tests timed out after 10 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """运行性能测试"""
        logger.info("⚡ 运行性能测试...")
        
        start_time = time.time()
        
        # 运行统一性能测试
        performance_test_file = self.tests_dir / 'unified_performance_test.py'
        if not performance_test_file.exists():
            logger.warning("未找到统一性能测试文件")
            return {
                'status': 'skipped',
                'reason': 'Unified performance test not found',
                'duration': 0
            }
        
        try:
            result = subprocess.run(
                [sys.executable, str(performance_test_file), '--test-type', 'comprehensive'],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            duration = time.time() - start_time
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'duration': time.time() - start_time,
                'error': 'Performance tests timed out after 10 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    async def run_benchmark_tests(self) -> Dict[str, Any]:
        """运行基准测试"""
        logger.info("📊 运行基准测试...")
        
        start_time = time.time()
        
        # 运行统一基准测试
        benchmark_test_file = self.tests_dir / 'unified_benchmark_test.py'
        if not benchmark_test_file.exists():
            logger.warning("未找到统一基准测试文件")
            return {
                'status': 'skipped',
                'reason': 'Unified benchmark test not found',
                'duration': 0
            }
        
        try:
            result = subprocess.run(
                [sys.executable, str(benchmark_test_file)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            duration = time.time() - start_time
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'duration': time.time() - start_time,
                'error': 'Benchmark tests timed out after 10 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    async def run_all_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """运行所有测试"""
        if test_types is None:
            test_types = ['unit', 'integration', 'e2e', 'performance', 'benchmark']
        
        logger.info(f"🚀 开始运行测试套件: {', '.join(test_types)}")
        
        # 检查依赖
        if not self.check_dependencies():
            return {
                'status': 'failed',
                'error': 'Dependency check failed'
            }
        
        # 运行各类测试
        test_functions = {
            'unit': self.run_unit_tests,
            'integration': self.run_integration_tests,
            'e2e': self.run_e2e_tests,
            'performance': self.run_performance_tests,
            'benchmark': self.run_benchmark_tests
        }
        
        for test_type in test_types:
            if test_type in test_functions:
                logger.info(f"\n{'='*60}")
                logger.info(f"开始 {test_type.upper()} 测试")
                logger.info(f"{'='*60}")
                
                self.results['tests'][test_type] = await test_functions[test_type]()
                
                # 显示测试结果摘要
                result = self.results['tests'][test_type]
                status_emoji = "✅" if result['status'] == 'success' else "❌" if result['status'] == 'failed' else "⏭️"
                logger.info(f"{status_emoji} {test_type.upper()} 测试完成 - 状态: {result['status']} - 耗时: {result['duration']:.2f}s")
            else:
                logger.warning(f"未知的测试类型: {test_type}")
        
        # 生成测试摘要
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """生成测试摘要"""
        total_tests = len(self.results['tests'])
        successful_tests = sum(1 for test in self.results['tests'].values() if test['status'] == 'success')
        failed_tests = sum(1 for test in self.results['tests'].values() if test['status'] == 'failed')
        skipped_tests = sum(1 for test in self.results['tests'].values() if test['status'] == 'skipped')
        
        total_duration = sum(test['duration'] for test in self.results['tests'].values())
        
        self.results['end_time'] = datetime.now().isoformat()
        self.results['summary'] = {
            'total_test_types': total_tests,
            'successful': successful_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'total_duration': total_duration,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0
        }
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "="*80)
        print("🎯 VoiceHelper 测试套件执行摘要")
        print("="*80)
        
        summary = self.results['summary']
        print(f"测试会话ID: {self.results['test_session_id']}")
        print(f"开始时间: {self.results['start_time']}")
        print(f"结束时间: {self.results['end_time']}")
        print(f"总耗时: {summary['total_duration']:.2f}秒")
        
        print(f"\n📊 测试结果统计:")
        print(f"  总测试类型: {summary['total_test_types']}")
        print(f"  成功: {summary['successful']} ✅")
        print(f"  失败: {summary['failed']} ❌")
        print(f"  跳过: {summary['skipped']} ⏭️")
        print(f"  成功率: {summary['success_rate']:.1f}%")
        
        print(f"\n📋 详细结果:")
        for test_type, result in self.results['tests'].items():
            status_emoji = {
                'success': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'timeout': '⏰',
                'error': '💥'
            }.get(result['status'], '❓')
            
            print(f"  {status_emoji} {test_type.upper()}: {result['status']} ({result['duration']:.2f}s)")
            
            # 显示测试详情
            if 'test_details' in result and result['test_details']:
                details = result['test_details']
                if 'total' in details:
                    print(f"    总计: {details['total']}, 通过: {details.get('passed', 0)}, 失败: {details.get('failed', 0)}")
            
            # 显示错误信息
            if result['status'] in ['failed', 'error', 'timeout'] and 'error' in result:
                print(f"    错误: {result['error']}")
        
        # 总体评价
        if summary['success_rate'] == 100:
            print(f"\n🎉 所有测试都通过了！系统状态良好。")
        elif summary['success_rate'] >= 80:
            print(f"\n✅ 大部分测试通过，系统基本正常。")
        elif summary['success_rate'] >= 60:
            print(f"\n⚠️ 部分测试失败，建议检查相关功能。")
        else:
            print(f"\n❌ 多项测试失败，系统可能存在严重问题。")
    
    def save_results(self, filename: str = None) -> str:
        """保存测试结果"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存到: {filename}")
        return filename

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiceHelper 统一测试运行器")
    parser.add_argument("--tests", nargs='+', 
                       choices=['unit', 'integration', 'e2e', 'performance', 'benchmark'],
                       default=['unit', 'integration', 'performance'],
                       help="要运行的测试类型")
    parser.add_argument("--output", type=str, help="结果输出文件路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建测试运行器
    runner = TestRunner()
    
    try:
        # 运行测试
        results = await runner.run_all_tests(args.tests)
        
        # 显示摘要
        runner.print_summary()
        
        # 保存结果
        output_file = runner.save_results(args.output)
        
        # 返回状态码
        success_rate = results['summary']['success_rate']
        return 0 if success_rate >= 80 else 1
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
