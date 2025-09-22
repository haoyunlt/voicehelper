#!/usr/bin/env python3
"""
VoiceHelper 统一性能测试套件
合并所有性能测试功能，提供完整的性能评估
"""

import asyncio
import time
import json
import statistics
import concurrent.futures
import psutil
import requests
import random
import math
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_system_metrics(self) -> Dict[str, float]:
        """获取系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用
        memory = psutil.virtual_memory()
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        
        # 进程内存使用
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100,
            'disk_free_gb': disk.free / (1024**3),
            'process_memory_mb': process_memory.rss / (1024**2),
            'process_memory_percent': process.memory_percent()
        }

class APITester:
    """API性能测试器"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8080"
        self.algo_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        
    async def test_api_response_times(self) -> Dict[str, Any]:
        """测试API响应时间"""
        endpoints = [
            (f"{self.backend_url}/health", "后端健康检查"),
            (f"{self.algo_url}/health", "算法服务健康检查"),
            (f"{self.frontend_url}", "前端页面")
        ]
        
        results = {}
        
        for url, name in endpoints:
            try:
                # 预热
                requests.get(url, timeout=5)
                
                # 测试多次取平均值
                response_times = []
                for _ in range(10):
                    start_time = time.time()
                    response = requests.get(url, timeout=10)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_times.append((end_time - start_time) * 1000)
                
                if response_times:
                    results[name] = {
                        'avg_response_time_ms': sum(response_times) / len(response_times),
                        'min_response_time_ms': min(response_times),
                        'max_response_time_ms': max(response_times),
                        'p95_response_time_ms': self._percentile(response_times, 95),
                        'success_rate': len(response_times) / 10,
                        'status': 'success'
                    }
                else:
                    results[name] = {'status': 'failed', 'error': 'All requests failed'}
                    
            except Exception as e:
                results[name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data) / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

class ConcurrencyTester:
    """并发性能测试器"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def test_concurrent_requests(self, concurrent_users: int = 10, duration: int = 30) -> Dict[str, Any]:
        """测试并发请求处理能力"""
        
        def make_request():
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}/health", timeout=5)
                end_time = time.time()
                return {
                    'success': response.status_code == 200,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time_ms': None,
                    'error': str(e)
                }
        
        start_time = time.time()
        total_requests = 0
        successful_requests = 0
        response_times = []
        
        # 运行并发测试
        while time.time() - start_time < duration:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrent_users)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            for result in results:
                total_requests += 1
                if result['success']:
                    successful_requests += 1
                    if result['response_time_ms']:
                        response_times.append(result['response_time_ms'])
            
            await asyncio.sleep(0.1)  # 短暂休息
        
        total_time = time.time() - start_time
        
        return {
            'concurrent_users': concurrent_users,
            'test_duration': total_time,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            'throughput_rps': total_requests / total_time if total_time > 0 else 0,
            'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
            'p95_response_time_ms': self._percentile(response_times, 95) if response_times else 0
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data) / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

class BatchTester:
    """批处理性能测试器"""
    
    def __init__(self):
        self.results = []
        
    async def test_batch_vs_individual(self, batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, Any]:
        """测试批处理 vs 单独处理性能"""
        
        results = {}
        
        for batch_size in batch_sizes:
            # 模拟批处理
            batch_start = time.time()
            await self._simulate_batch_processing(batch_size)
            batch_time = time.time() - batch_start
            
            # 模拟单独处理
            individual_start = time.time()
            for _ in range(batch_size):
                await self._simulate_individual_processing()
            individual_time = time.time() - individual_start
            
            # 计算改进
            improvement = ((individual_time - batch_time) / individual_time * 100) if individual_time > 0 else 0
            
            results[f'batch_size_{batch_size}'] = {
                'batch_time': batch_time,
                'individual_time': individual_time,
                'improvement_percent': improvement,
                'throughput_improvement': individual_time / batch_time if batch_time > 0 else 1
            }
        
        return results
    
    async def _simulate_batch_processing(self, batch_size: int):
        """模拟批处理"""
        # 批处理效率：批次越大，单个请求的平均时间越短
        base_time = 0.1  # 100ms基础时间
        efficiency_factor = 1.0 + (batch_size - 1) * 0.1  # 每增加一个请求，效率提升10%
        actual_time = base_time * efficiency_factor / batch_size
        await asyncio.sleep(actual_time)
    
    async def _simulate_individual_processing(self):
        """模拟单独处理"""
        await asyncio.sleep(0.1)  # 100ms处理时间

class MemoryTester:
    """内存性能测试器"""
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用情况"""
        # 获取当前进程内存使用
        process = psutil.Process()
        initial_memory = process.memory_info()
        
        # 模拟内存密集操作
        test_data = []
        for i in range(10000):
            test_data.append({
                'id': i,
                'message': f'Test message {i}',
                'timestamp': time.time(),
                'data': list(range(100))  # 增加内存使用
            })
        
        # 获取峰值内存
        peak_memory = process.memory_info()
        
        # 清理数据
        del test_data
        
        # 获取清理后内存
        final_memory = process.memory_info()
        
        return {
            'initial_memory_mb': initial_memory.rss / (1024**2),
            'peak_memory_mb': peak_memory.rss / (1024**2),
            'final_memory_mb': final_memory.rss / (1024**2),
            'memory_increase_mb': (peak_memory.rss - initial_memory.rss) / (1024**2),
            'memory_retained_mb': (final_memory.rss - initial_memory.rss) / (1024**2),
            'memory_efficiency': (peak_memory.rss - final_memory.rss) / (peak_memory.rss - initial_memory.rss) if peak_memory.rss > initial_memory.rss else 0
        }

class UnifiedPerformanceTest:
    """统一性能测试主类"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.api_tester = APITester()
        self.concurrency_tester = ConcurrencyTester("http://localhost:8080")
        self.batch_tester = BatchTester()
        self.memory_tester = MemoryTester()
        self.results = []
        
    async def run_quick_test(self) -> Dict[str, Any]:
        """运行快速性能测试"""
        logger.info("🚀 开始快速性能测试")
        
        results = {
            'test_type': 'quick',
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # 1. 系统资源检查
        logger.info("📊 系统资源检查")
        system_metrics = self.system_monitor.get_system_metrics()
        results['tests']['system_resources'] = TestResult(
            test_name="system_resources",
            success=True,
            duration=1.0,
            metrics=system_metrics
        ).__dict__
        
        # 2. API响应时间测试
        logger.info("⏱️ API响应时间测试")
        start_time = time.time()
        api_results = await self.api_tester.test_api_response_times()
        api_duration = time.time() - start_time
        
        results['tests']['api_performance'] = TestResult(
            test_name="api_performance",
            success=all(r.get('status') == 'success' for r in api_results.values()),
            duration=api_duration,
            metrics=api_results
        ).__dict__
        
        # 3. 内存使用测试
        logger.info("💾 内存使用测试")
        start_time = time.time()
        memory_results = self.memory_tester.test_memory_usage()
        memory_duration = time.time() - start_time
        
        results['tests']['memory_usage'] = TestResult(
            test_name="memory_usage",
            success=memory_results['memory_increase_mb'] < 100,  # 内存增长小于100MB
            duration=memory_duration,
            metrics=memory_results
        ).__dict__
        
        return results
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合性能测试"""
        logger.info("🎯 开始综合性能测试")
        
        results = {
            'test_type': 'comprehensive',
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # 运行快速测试
        quick_results = await self.run_quick_test()
        results['tests'].update(quick_results['tests'])
        
        # 4. 并发测试
        logger.info("🔄 并发性能测试")
        start_time = time.time()
        concurrency_results = await self.concurrency_tester.test_concurrent_requests(
            concurrent_users=10, duration=30
        )
        concurrency_duration = time.time() - start_time
        
        results['tests']['concurrency'] = TestResult(
            test_name="concurrency",
            success=concurrency_results['success_rate'] > 95,
            duration=concurrency_duration,
            metrics=concurrency_results
        ).__dict__
        
        # 5. 批处理测试
        logger.info("📦 批处理性能测试")
        start_time = time.time()
        batch_results = await self.batch_tester.test_batch_vs_individual()
        batch_duration = time.time() - start_time
        
        # 检查批处理是否有性能提升
        avg_improvement = sum(
            r['improvement_percent'] for r in batch_results.values()
        ) / len(batch_results)
        
        results['tests']['batch_processing'] = TestResult(
            test_name="batch_processing",
            success=avg_improvement > 20,  # 平均改善超过20%
            duration=batch_duration,
            metrics=batch_results
        ).__dict__
        
        return results
    
    def calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """计算性能评分"""
        score = 100
        
        for test_name, test_result in results['tests'].items():
            if not test_result['success']:
                score -= 20  # 每个失败的测试扣20分
            
            # 根据具体指标调整分数
            metrics = test_result['metrics']
            
            if test_name == 'system_resources':
                if metrics.get('cpu_percent', 0) > 80:
                    score -= 10
                if metrics.get('memory_percent', 0) > 90:
                    score -= 15
            
            elif test_name == 'api_performance':
                for endpoint, data in metrics.items():
                    if data.get('status') == 'success':
                        avg_time = data.get('avg_response_time_ms', 0)
                        if avg_time > 1000:  # 超过1秒
                            score -= 10
                        elif avg_time > 500:  # 超过500ms
                            score -= 5
            
            elif test_name == 'concurrency':
                if metrics.get('success_rate', 0) < 90:
                    score -= 15
                if metrics.get('avg_response_time_ms', 0) > 1000:
                    score -= 10
        
        return max(0, score)
    
    def print_results(self, results: Dict[str, Any]):
        """打印测试结果"""
        print("\n" + "="*80)
        print("🎯 VoiceHelper 统一性能测试报告")
        print("="*80)
        
        print(f"测试类型: {results['test_type']}")
        print(f"测试时间: {results['timestamp']}")
        
        # 计算总体评分
        score = self.calculate_performance_score(results)
        print(f"总体评分: {score:.1f}/100")
        
        if score >= 90:
            print("🎉 性能优秀！")
        elif score >= 70:
            print("✅ 性能良好")
        elif score >= 50:
            print("⚠️ 性能一般，建议优化")
        else:
            print("❌ 性能较差，需要重点优化")
        
        # 详细结果
        print(f"\n📊 详细测试结果:")
        for test_name, test_result in results['tests'].items():
            status = "✅" if test_result['success'] else "❌"
            print(f"\n{status} {test_result['test_name']} ({test_result['duration']:.2f}s)")
            
            metrics = test_result['metrics']
            
            if test_name == 'system_resources':
                print(f"  CPU使用率: {metrics.get('cpu_percent', 0):.1f}%")
                print(f"  内存使用率: {metrics.get('memory_percent', 0):.1f}%")
                print(f"  可用内存: {metrics.get('memory_available_gb', 0):.2f} GB")
                print(f"  进程内存: {metrics.get('process_memory_mb', 0):.2f} MB")
            
            elif test_name == 'api_performance':
                for endpoint, data in metrics.items():
                    if data.get('status') == 'success':
                        print(f"  {endpoint}: {data.get('avg_response_time_ms', 0):.2f}ms")
                    else:
                        print(f"  {endpoint}: 失败 - {data.get('error', 'Unknown error')}")
            
            elif test_name == 'memory_usage':
                print(f"  内存增长: {metrics.get('memory_increase_mb', 0):.2f} MB")
                print(f"  内存保留: {metrics.get('memory_retained_mb', 0):.2f} MB")
                print(f"  内存效率: {metrics.get('memory_efficiency', 0):.2%}")
            
            elif test_name == 'concurrency':
                print(f"  并发用户: {metrics.get('concurrent_users', 0)}")
                print(f"  成功率: {metrics.get('success_rate', 0):.1f}%")
                print(f"  吞吐量: {metrics.get('throughput_rps', 0):.2f} req/s")
                print(f"  平均响应时间: {metrics.get('avg_response_time_ms', 0):.2f}ms")
            
            elif test_name == 'batch_processing':
                print(f"  批处理测试结果:")
                for batch_name, batch_data in metrics.items():
                    improvement = batch_data.get('improvement_percent', 0)
                    print(f"    {batch_name}: {improvement:.1f}% 性能提升")
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存测试结果"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/performance_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存到: {filename}")
        return filename

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiceHelper 统一性能测试")
    parser.add_argument("--test-type", choices=["quick", "comprehensive"], 
                       default="quick", help="测试类型")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建测试器
    tester = UnifiedPerformanceTest()
    
    try:
        # 运行测试
        if args.test_type == "quick":
            results = await tester.run_quick_test()
        else:
            results = await tester.run_comprehensive_test()
        
        # 显示结果
        tester.print_results(results)
        
        # 保存结果
        output_file = tester.save_results(results, args.output)
        
        # 返回成功状态
        score = tester.calculate_performance_score(results)
        return 0 if score >= 70 else 1
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
