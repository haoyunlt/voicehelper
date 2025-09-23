#!/usr/bin/env python3
"""
VoiceHelper 统一基准测试套件
建立系统性能基线，用于回归测试和性能对比
"""

import asyncio
import time
import json
import statistics
import requests
import psutil
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import concurrent.futures
import threading

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    timestamp: str
    duration: float
    success: bool
    metrics: Dict[str, Any]
    baseline_metrics: Optional[Dict[str, Any]] = None
    comparison: Optional[Dict[str, Any]] = None

class ServiceHealthChecker:
    """服务健康检查器"""
    
    def __init__(self):
        self.services = {
            'backend': 'http://localhost:8080/health',
            'algorithm': 'http://localhost:8000/health',
            'frontend': 'http://localhost:3000'
        }
    
    def check_all_services(self) -> Dict[str, Any]:
        """检查所有服务健康状态"""
        results = {}
        
        for service_name, url in self.services.items():
            try:
                start_time = time.time()
                response = requests.get(url, timeout=10)
                end_time = time.time()
                
                results[service_name] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'status_code': response.status_code,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'available': True
                }
                
                # 尝试解析响应
                try:
                    if response.headers.get('content-type', '').startswith('application/json'):
                        results[service_name]['response_data'] = response.json()
                except:
                    pass
                    
            except requests.exceptions.RequestException as e:
                results[service_name] = {
                    'status': 'unavailable',
                    'error': str(e),
                    'available': False
                }
        
        return results

class PerformanceBaseline:
    """性能基线测试器"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8080"
        self.algo_url = "http://localhost:8000"
        
    def establish_api_baseline(self, iterations: int = 50) -> Dict[str, Any]:
        """建立API性能基线"""
        endpoints = [
            (f"{self.backend_url}/health", "backend_health"),
            (f"{self.backend_url}/api/v1/ping", "backend_ping"),
            (f"{self.algo_url}/health", "algorithm_health")
        ]
        
        baseline_data = {}
        
        for url, endpoint_name in endpoints:
            logger.info(f"建立 {endpoint_name} 基线...")
            
            response_times = []
            success_count = 0
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=5)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_times.append((end_time - start_time) * 1000)
                        success_count += 1
                        
                except Exception as e:
                    logger.debug(f"请求失败: {e}")
                    continue
            
            if response_times:
                baseline_data[endpoint_name] = {
                    'avg_response_time_ms': statistics.mean(response_times),
                    'median_response_time_ms': statistics.median(response_times),
                    'min_response_time_ms': min(response_times),
                    'max_response_time_ms': max(response_times),
                    'p95_response_time_ms': self._percentile(response_times, 95),
                    'p99_response_time_ms': self._percentile(response_times, 99),
                    'std_dev_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    'success_rate': (success_count / iterations) * 100,
                    'total_requests': iterations,
                    'successful_requests': success_count
                }
            else:
                baseline_data[endpoint_name] = {
                    'status': 'failed',
                    'error': 'No successful requests'
                }
        
        return baseline_data
    
    def establish_throughput_baseline(self, duration: int = 60) -> Dict[str, Any]:
        """建立吞吐量基线"""
        url = f"{self.backend_url}/health"
        
        def make_request():
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                end_time = time.time()
                return {
                    'success': response.status_code == 200,
                    'response_time': (end_time - start_time) * 1000
                }
            except:
                return {'success': False, 'response_time': None}
        
        logger.info(f"建立吞吐量基线 ({duration}秒)...")
        
        start_time = time.time()
        total_requests = 0
        successful_requests = 0
        response_times = []
        
        # 单线程基线测试
        while time.time() - start_time < duration:
            result = make_request()
            total_requests += 1
            
            if result['success']:
                successful_requests += 1
                if result['response_time']:
                    response_times.append(result['response_time'])
        
        actual_duration = time.time() - start_time
        
        return {
            'duration_seconds': actual_duration,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests,
            'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            'throughput_rps': total_requests / actual_duration if actual_duration > 0 else 0,
            'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
            'p95_response_time_ms': self._percentile(response_times, 95) if response_times else 0
        }
    
    def establish_concurrency_baseline(self, max_concurrent: int = 20) -> Dict[str, Any]:
        """建立并发处理基线"""
        url = f"{self.backend_url}/health"
        
        def test_concurrency_level(concurrent_users: int) -> Dict[str, Any]:
            def make_request():
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=10)
                    end_time = time.time()
                    return {
                        'success': response.status_code == 200,
                        'response_time_ms': (end_time - start_time) * 1000
                    }
                except:
                    return {'success': False, 'response_time_ms': None}
            
            # 执行并发测试
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrent_users * 2)]  # 每个用户发2个请求
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            successful_results = [r for r in results if r['success']]
            response_times = [r['response_time_ms'] for r in successful_results if r['response_time_ms']]
            
            return {
                'concurrent_users': concurrent_users,
                'total_requests': len(results),
                'successful_requests': len(successful_results),
                'success_rate': (len(successful_results) / len(results) * 100) if results else 0,
                'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
                'p95_response_time_ms': self._percentile(response_times, 95) if response_times else 0
            }
        
        logger.info(f"建立并发处理基线 (最大 {max_concurrent} 并发)...")
        
        concurrency_results = {}
        
        # 测试不同并发级别
        for concurrent_users in [1, 2, 5, 10, 15, 20]:
            if concurrent_users <= max_concurrent:
                logger.info(f"测试 {concurrent_users} 并发用户...")
                concurrency_results[f'concurrent_{concurrent_users}'] = test_concurrency_level(concurrent_users)
        
        return concurrency_results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data) / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

class SystemResourceBaseline:
    """系统资源基线测试器"""
    
    def establish_resource_baseline(self, duration: int = 30) -> Dict[str, Any]:
        """建立系统资源使用基线"""
        logger.info(f"建立系统资源基线 ({duration}秒)...")
        
        cpu_samples = []
        memory_samples = []
        disk_io_samples = []
        network_io_samples = []
        
        # 获取初始网络和磁盘IO
        initial_net_io = psutil.net_io_counters()
        initial_disk_io = psutil.disk_io_counters()
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_samples.append(cpu_percent)
            
            # 内存使用
            memory = psutil.virtual_memory()
            memory_samples.append(memory.percent)
            
            # 磁盘IO
            current_disk_io = psutil.disk_io_counters()
            if current_disk_io and initial_disk_io:
                disk_read_rate = (current_disk_io.read_bytes - initial_disk_io.read_bytes) / (time.time() - start_time)
                disk_write_rate = (current_disk_io.write_bytes - initial_disk_io.write_bytes) / (time.time() - start_time)
                disk_io_samples.append({'read_rate': disk_read_rate, 'write_rate': disk_write_rate})
            
            # 网络IO
            current_net_io = psutil.net_io_counters()
            if current_net_io and initial_net_io:
                net_recv_rate = (current_net_io.bytes_recv - initial_net_io.bytes_recv) / (time.time() - start_time)
                net_sent_rate = (current_net_io.bytes_sent - initial_net_io.bytes_sent) / (time.time() - start_time)
                network_io_samples.append({'recv_rate': net_recv_rate, 'sent_rate': net_sent_rate})
        
        return {
            'cpu': {
                'avg_percent': statistics.mean(cpu_samples) if cpu_samples else 0,
                'max_percent': max(cpu_samples) if cpu_samples else 0,
                'min_percent': min(cpu_samples) if cpu_samples else 0,
                'std_dev': statistics.stdev(cpu_samples) if len(cpu_samples) > 1 else 0
            },
            'memory': {
                'avg_percent': statistics.mean(memory_samples) if memory_samples else 0,
                'max_percent': max(memory_samples) if memory_samples else 0,
                'min_percent': min(memory_samples) if memory_samples else 0,
                'std_dev': statistics.stdev(memory_samples) if len(memory_samples) > 1 else 0
            },
            'disk_io': {
                'avg_read_rate_bps': statistics.mean([s['read_rate'] for s in disk_io_samples]) if disk_io_samples else 0,
                'avg_write_rate_bps': statistics.mean([s['write_rate'] for s in disk_io_samples]) if disk_io_samples else 0
            },
            'network_io': {
                'avg_recv_rate_bps': statistics.mean([s['recv_rate'] for s in network_io_samples]) if network_io_samples else 0,
                'avg_sent_rate_bps': statistics.mean([s['sent_rate'] for s in network_io_samples]) if network_io_samples else 0
            }
        }

class UnifiedBenchmarkTest:
    """统一基准测试主类"""
    
    def __init__(self):
        self.health_checker = ServiceHealthChecker()
        self.performance_baseline = PerformanceBaseline()
        self.resource_baseline = SystemResourceBaseline()
        
    def run_full_benchmark(self) -> Dict[str, Any]:
        """运行完整基准测试"""
        logger.info("🎯 开始建立VoiceHelper性能基线")
        
        benchmark_results = {
            'benchmark_id': f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'tests': {}
        }
        
        # 1. 服务健康检查
        logger.info("🔍 服务健康检查...")
        start_time = time.time()
        health_results = self.health_checker.check_all_services()
        health_duration = time.time() - start_time
        
        benchmark_results['tests']['service_health'] = BenchmarkResult(
            test_name="service_health",
            timestamp=datetime.now().isoformat(),
            duration=health_duration,
            success=all(service.get('available', False) for service in health_results.values()),
            metrics=health_results
        ).__dict__
        
        # 检查服务是否可用，如果不可用则跳过后续测试
        if not benchmark_results['tests']['service_health']['success']:
            logger.warning("⚠️ 部分服务不可用，跳过性能基线测试")
            return benchmark_results
        
        # 2. API性能基线
        logger.info("📊 建立API性能基线...")
        start_time = time.time()
        api_baseline = self.performance_baseline.establish_api_baseline()
        api_duration = time.time() - start_time
        
        benchmark_results['tests']['api_baseline'] = BenchmarkResult(
            test_name="api_baseline",
            timestamp=datetime.now().isoformat(),
            duration=api_duration,
            success=all(endpoint.get('success_rate', 0) > 90 for endpoint in api_baseline.values() if isinstance(endpoint, dict)),
            metrics=api_baseline
        ).__dict__
        
        # 3. 吞吐量基线
        logger.info("🚀 建立吞吐量基线...")
        start_time = time.time()
        throughput_baseline = self.performance_baseline.establish_throughput_baseline(duration=30)
        throughput_duration = time.time() - start_time
        
        benchmark_results['tests']['throughput_baseline'] = BenchmarkResult(
            test_name="throughput_baseline",
            timestamp=datetime.now().isoformat(),
            duration=throughput_duration,
            success=throughput_baseline.get('success_rate', 0) > 95,
            metrics=throughput_baseline
        ).__dict__
        
        # 4. 并发处理基线
        logger.info("🔄 建立并发处理基线...")
        start_time = time.time()
        concurrency_baseline = self.performance_baseline.establish_concurrency_baseline()
        concurrency_duration = time.time() - start_time
        
        benchmark_results['tests']['concurrency_baseline'] = BenchmarkResult(
            test_name="concurrency_baseline",
            timestamp=datetime.now().isoformat(),
            duration=concurrency_duration,
            success=all(level.get('success_rate', 0) > 90 for level in concurrency_baseline.values()),
            metrics=concurrency_baseline
        ).__dict__
        
        # 5. 系统资源基线
        logger.info("💻 建立系统资源基线...")
        start_time = time.time()
        resource_baseline = self.resource_baseline.establish_resource_baseline()
        resource_duration = time.time() - start_time
        
        benchmark_results['tests']['resource_baseline'] = BenchmarkResult(
            test_name="resource_baseline",
            timestamp=datetime.now().isoformat(),
            duration=resource_duration,
            success=resource_baseline['cpu']['avg_percent'] < 80 and resource_baseline['memory']['avg_percent'] < 90,
            metrics=resource_baseline
        ).__dict__
        
        return benchmark_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': os.name,
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'python_version': sys.version
        }
    
    def save_baseline(self, results: Dict[str, Any], filename: str = None) -> str:
        """保存基线数据"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tests/baseline_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"基线数据已保存到: {filename}")
        return filename
    
    def load_baseline(self, filename: str) -> Dict[str, Any]:
        """加载基线数据"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"基线文件不存在: {filename}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"基线文件格式错误: {filename}")
            return {}
    
    def compare_with_baseline(self, current_results: Dict[str, Any], baseline_file: str) -> Dict[str, Any]:
        """与基线对比"""
        baseline = self.load_baseline(baseline_file)
        if not baseline:
            return {}
        
        comparison = {
            'baseline_timestamp': baseline.get('timestamp'),
            'current_timestamp': current_results.get('timestamp'),
            'comparisons': {}
        }
        
        # 对比各项测试结果
        for test_name in current_results.get('tests', {}):
            if test_name in baseline.get('tests', {}):
                current_metrics = current_results['tests'][test_name]['metrics']
                baseline_metrics = baseline['tests'][test_name]['metrics']
                
                comparison['comparisons'][test_name] = self._compare_metrics(
                    current_metrics, baseline_metrics
                )
        
        return comparison
    
    def _compare_metrics(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """对比指标"""
        comparison = {
            'improvements': [],
            'regressions': [],
            'unchanged': []
        }
        
        # 这里简化处理，实际应该根据不同测试类型进行详细对比
        for key in current:
            if key in baseline and isinstance(current[key], (int, float)) and isinstance(baseline[key], (int, float)):
                if current[key] != baseline[key]:
                    change_percent = ((current[key] - baseline[key]) / baseline[key] * 100) if baseline[key] != 0 else 0
                    
                    # 根据指标类型判断是改善还是回归
                    # 响应时间类指标：越小越好
                    if 'time' in key.lower() or 'latency' in key.lower():
                        if change_percent < 0:
                            comparison['improvements'].append({
                                'metric': key,
                                'change_percent': abs(change_percent),
                                'current': current[key],
                                'baseline': baseline[key]
                            })
                        else:
                            comparison['regressions'].append({
                                'metric': key,
                                'change_percent': change_percent,
                                'current': current[key],
                                'baseline': baseline[key]
                            })
                    # 成功率、吞吐量类指标：越大越好
                    elif 'rate' in key.lower() or 'throughput' in key.lower():
                        if change_percent > 0:
                            comparison['improvements'].append({
                                'metric': key,
                                'change_percent': change_percent,
                                'current': current[key],
                                'baseline': baseline[key]
                            })
                        else:
                            comparison['regressions'].append({
                                'metric': key,
                                'change_percent': abs(change_percent),
                                'current': current[key],
                                'baseline': baseline[key]
                            })
                else:
                    comparison['unchanged'].append(key)
        
        return comparison
    
    def print_benchmark_results(self, results: Dict[str, Any]):
        """打印基准测试结果"""
        print("\n" + "="*80)
        print("📊 VoiceHelper 基准测试报告")
        print("="*80)
        
        print(f"基准ID: {results.get('benchmark_id')}")
        print(f"测试时间: {results.get('timestamp')}")
        
        # 系统信息
        system_info = results.get('system_info', {})
        print(f"\n💻 系统信息:")
        print(f"  CPU核心数: {system_info.get('cpu_count')} 物理 / {system_info.get('cpu_count_logical')} 逻辑")
        print(f"  内存总量: {system_info.get('memory_total_gb', 0):.2f} GB")
        print(f"  磁盘总量: {system_info.get('disk_total_gb', 0):.2f} GB")
        
        # 测试结果
        print(f"\n📋 基准测试结果:")
        
        for test_name, test_result in results.get('tests', {}).items():
            status = "✅" if test_result['success'] else "❌"
            duration = test_result['duration']
            print(f"\n{status} {test_name} ({duration:.2f}s)")
            
            metrics = test_result['metrics']
            
            if test_name == 'service_health':
                for service, data in metrics.items():
                    if data.get('available'):
                        print(f"  {service}: ✅ {data.get('response_time_ms', 0):.2f}ms")
                    else:
                        print(f"  {service}: ❌ {data.get('error', 'Unknown error')}")
            
            elif test_name == 'api_baseline':
                for endpoint, data in metrics.items():
                    if isinstance(data, dict) and 'avg_response_time_ms' in data:
                        print(f"  {endpoint}:")
                        print(f"    平均响应时间: {data['avg_response_time_ms']:.2f}ms")
                        print(f"    P95响应时间: {data['p95_response_time_ms']:.2f}ms")
                        print(f"    成功率: {data['success_rate']:.1f}%")
            
            elif test_name == 'throughput_baseline':
                print(f"  吞吐量: {metrics.get('throughput_rps', 0):.2f} req/s")
                print(f"  成功率: {metrics.get('success_rate', 0):.1f}%")
                print(f"  平均响应时间: {metrics.get('avg_response_time_ms', 0):.2f}ms")
            
            elif test_name == 'concurrency_baseline':
                for level_name, level_data in metrics.items():
                    concurrent_users = level_data.get('concurrent_users', 0)
                    success_rate = level_data.get('success_rate', 0)
                    avg_time = level_data.get('avg_response_time_ms', 0)
                    print(f"  {concurrent_users}并发: {success_rate:.1f}% 成功率, {avg_time:.2f}ms 平均响应")
            
            elif test_name == 'resource_baseline':
                cpu_data = metrics.get('cpu', {})
                memory_data = metrics.get('memory', {})
                print(f"  CPU使用率: {cpu_data.get('avg_percent', 0):.1f}% (最大: {cpu_data.get('max_percent', 0):.1f}%)")
                print(f"  内存使用率: {memory_data.get('avg_percent', 0):.1f}% (最大: {memory_data.get('max_percent', 0):.1f}%)")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiceHelper 统一基准测试")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--compare", type=str, help="与指定基线文件对比")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建基准测试器
    benchmark = UnifiedBenchmarkTest()
    
    try:
        # 运行基准测试
        results = benchmark.run_full_benchmark()
        
        # 如果指定了对比基线
        if args.compare:
            comparison = benchmark.compare_with_baseline(results, args.compare)
            results['comparison'] = comparison
        
        # 显示结果
        benchmark.print_benchmark_results(results)
        
        # 保存结果
        output_file = benchmark.save_baseline(results, args.output)
        
        # 检查是否所有测试都成功
        all_success = all(test['success'] for test in results.get('tests', {}).values())
        return 0 if all_success else 1
        
    except Exception as e:
        logger.error(f"基准测试执行失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
