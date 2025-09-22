"""
基准测试脚本
建立系统性能基线，用于性能回归检测和优化效果验证
"""

import asyncio
import aiohttp
import time
import json
import statistics
import csv
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import psutil
import os
import subprocess
import platform


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    timestamp: datetime
    response_time: float
    throughput: float
    success_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    concurrent_users: int
    test_duration: float
    system_info: Dict[str, Any]


class SystemProfiler:
    """系统性能分析器"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'python_version': platform.python_version(),
        }
    
    @staticmethod
    def get_current_usage() -> Dict[str, float]:
        """获取当前资源使用情况"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
        }


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.algo_url = "http://localhost:8000"
        self.results: List[BenchmarkResult] = []
        self.system_profiler = SystemProfiler()
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def benchmark_health_check(self, iterations: int = 100) -> BenchmarkResult:
        """基准测试：健康检查接口"""
        self.logger.info(f"开始健康检查基准测试 ({iterations}次)")
        
        response_times = []
        success_count = 0
        start_time = time.time()
        
        system_usage_before = self.system_profiler.get_current_usage()
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                request_start = time.time()
                
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        
                        if response.status == 200:
                            success_count += 1
                            
                except Exception as e:
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    self.logger.warning(f"请求失败: {e}")
                
                # 控制请求频率
                await asyncio.sleep(0.01)
        
        system_usage_after = self.system_profiler.get_current_usage()
        total_time = time.time() - start_time
        
        return self._create_benchmark_result(
            test_name="health_check",
            response_times=response_times,
            success_count=success_count,
            total_requests=iterations,
            total_time=total_time,
            concurrent_users=1,
            system_usage_before=system_usage_before,
            system_usage_after=system_usage_after
        )
    
    async def benchmark_chat_completion(self, iterations: int = 50) -> BenchmarkResult:
        """基准测试：聊天完成接口"""
        self.logger.info(f"开始聊天完成基准测试 ({iterations}次)")
        
        response_times = []
        success_count = 0
        start_time = time.time()
        
        system_usage_before = self.system_profiler.get_current_usage()
        
        # 测试消息模板
        test_messages = [
            "你好，请介绍一下VoiceHelper系统",
            "系统的主要功能有哪些？",
            "如何使用语音功能？",
            "技术架构是怎样的？",
            "性能指标如何？"
        ]
        
        headers = {
            'Authorization': 'Bearer benchmark_test_token',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                message = test_messages[i % len(test_messages)]
                
                chat_data = {
                    'conversation_id': f'benchmark_test_{i}',
                    'messages': [
                        {
                            'role': 'user',
                            'content': f"{message} (基准测试 #{i+1})"
                        }
                    ],
                    'top_k': 3,
                    'temperature': 0.3,
                    'max_tokens': 200
                }
                
                request_start = time.time()
                
                try:
                    async with session.post(
                        f"{self.base_url}/api/v1/chat/completions",
                        headers=headers,
                        json=chat_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        # 读取响应内容
                        if response.content_type == 'text/plain':
                            # SSE流式响应
                            async for line in response.content:
                                pass  # 读取所有数据
                        else:
                            await response.text()
                        
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        
                        if response.status == 200:
                            success_count += 1
                            
                except Exception as e:
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    self.logger.warning(f"聊天请求失败: {e}")
                
                # 控制请求频率
                await asyncio.sleep(0.5)
        
        system_usage_after = self.system_profiler.get_current_usage()
        total_time = time.time() - start_time
        
        return self._create_benchmark_result(
            test_name="chat_completion",
            response_times=response_times,
            success_count=success_count,
            total_requests=iterations,
            total_time=total_time,
            concurrent_users=1,
            system_usage_before=system_usage_before,
            system_usage_after=system_usage_after
        )
    
    async def benchmark_document_query(self, iterations: int = 30) -> BenchmarkResult:
        """基准测试：文档查询接口"""
        self.logger.info(f"开始文档查询基准测试 ({iterations}次)")
        
        response_times = []
        success_count = 0
        start_time = time.time()
        
        system_usage_before = self.system_profiler.get_current_usage()
        
        # 测试查询列表
        test_queries = [
            "VoiceHelper的核心功能是什么？",
            "系统架构设计原理",
            "如何保证系统安全性？",
            "性能优化策略有哪些？",
            "支持的技术栈介绍",
            "部署和运维方案",
            "监控和告警机制",
            "扩展性设计考虑"
        ]
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                query = test_queries[i % len(test_queries)]
                
                query_data = {
                    'messages': [
                        {
                            'role': 'user',
                            'content': f"{query} (基准测试查询 #{i+1})"
                        }
                    ],
                    'top_k': 5,
                    'temperature': 0.3,
                    'max_tokens': 300
                }
                
                request_start = time.time()
                
                try:
                    async with session.post(
                        f"{self.algo_url}/query",
                        json=query_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        # 读取NDJSON响应
                        response_text = await response.text()
                        
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                        
                        if response.status == 200 and response_text.strip():
                            success_count += 1
                            
                except Exception as e:
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    self.logger.warning(f"查询请求失败: {e}")
                
                # 控制请求频率
                await asyncio.sleep(1.0)
        
        system_usage_after = self.system_profiler.get_current_usage()
        total_time = time.time() - start_time
        
        return self._create_benchmark_result(
            test_name="document_query",
            response_times=response_times,
            success_count=success_count,
            total_requests=iterations,
            total_time=total_time,
            concurrent_users=1,
            system_usage_before=system_usage_before,
            system_usage_after=system_usage_after
        )
    
    async def benchmark_concurrent_requests(self, concurrent_users: int = 10, duration: int = 60) -> BenchmarkResult:
        """基准测试：并发请求"""
        self.logger.info(f"开始并发请求基准测试 ({concurrent_users}并发用户, {duration}秒)")
        
        response_times = []
        success_count = 0
        total_requests = 0
        start_time = time.time()
        
        system_usage_before = self.system_profiler.get_current_usage()
        
        async def worker(session: aiohttp.ClientSession, worker_id: int):
            """工作协程"""
            nonlocal response_times, success_count, total_requests
            
            worker_start = time.time()
            
            while time.time() - worker_start < duration:
                request_start = time.time()
                
                try:
                    # 随机选择请求类型
                    import random
                    request_type = random.choice(['health', 'ping', 'chat'])
                    
                    if request_type == 'health':
                        async with session.get(f"{self.base_url}/health") as response:
                            await response.text()
                            if response.status == 200:
                                success_count += 1
                    
                    elif request_type == 'ping':
                        headers = {'Authorization': 'Bearer benchmark_test_token'}
                        async with session.get(f"{self.base_url}/api/v1/ping", headers=headers) as response:
                            await response.text()
                            if response.status == 200:
                                success_count += 1
                    
                    elif request_type == 'chat':
                        headers = {
                            'Authorization': 'Bearer benchmark_test_token',
                            'Content-Type': 'application/json'
                        }
                        chat_data = {
                            'conversation_id': f'benchmark_concurrent_{worker_id}',
                            'messages': [
                                {
                                    'role': 'user',
                                    'content': f'并发测试消息 worker-{worker_id} #{total_requests}'
                                }
                            ]
                        }
                        async with session.post(
                            f"{self.base_url}/api/v1/chat/completions",
                            headers=headers,
                            json=chat_data,
                            timeout=aiohttp.ClientTimeout(total=15)
                        ) as response:
                            # 读取响应
                            if response.content_type.startswith('text/'):
                                async for line in response.content:
                                    pass
                            else:
                                await response.text()
                            
                            if response.status == 200:
                                success_count += 1
                    
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    total_requests += 1
                    
                except Exception as e:
                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    total_requests += 1
                    self.logger.debug(f"Worker {worker_id} 请求失败: {e}")
                
                # 短暂休息
                await asyncio.sleep(0.1)
        
        # 启动并发工作协程
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(worker(session, i))
                for i in range(concurrent_users)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        system_usage_after = self.system_profiler.get_current_usage()
        total_time = time.time() - start_time
        
        return self._create_benchmark_result(
            test_name="concurrent_requests",
            response_times=response_times,
            success_count=success_count,
            total_requests=total_requests,
            total_time=total_time,
            concurrent_users=concurrent_users,
            system_usage_before=system_usage_before,
            system_usage_after=system_usage_after
        )
    
    def _create_benchmark_result(
        self,
        test_name: str,
        response_times: List[float],
        success_count: int,
        total_requests: int,
        total_time: float,
        concurrent_users: int,
        system_usage_before: Dict[str, float],
        system_usage_after: Dict[str, float]
    ) -> BenchmarkResult:
        """创建基准测试结果"""
        
        if not response_times:
            response_times = [0.0]
        
        sorted_times = sorted(response_times)
        
        return BenchmarkResult(
            test_name=test_name,
            timestamp=datetime.now(),
            response_time=statistics.mean(response_times),
            throughput=total_requests / total_time if total_time > 0 else 0,
            success_rate=success_count / total_requests * 100 if total_requests > 0 else 0,
            error_rate=(total_requests - success_count) / total_requests * 100 if total_requests > 0 else 0,
            cpu_usage=(system_usage_before['cpu_percent'] + system_usage_after['cpu_percent']) / 2,
            memory_usage=(system_usage_before['memory_percent'] + system_usage_after['memory_percent']) / 2,
            p50_response_time=sorted_times[len(sorted_times) // 2],
            p95_response_time=sorted_times[int(len(sorted_times) * 0.95)],
            p99_response_time=sorted_times[int(len(sorted_times) * 0.99)],
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            requests_per_second=total_requests / total_time if total_time > 0 else 0,
            concurrent_users=concurrent_users,
            test_duration=total_time,
            system_info=self.system_profiler.get_system_info()
        )
    
    async def run_full_benchmark_suite(self) -> List[BenchmarkResult]:
        """运行完整基准测试套件"""
        self.logger.info("开始运行完整基准测试套件")
        
        # 系统预热
        self.logger.info("系统预热中...")
        await self.benchmark_health_check(10)
        await asyncio.sleep(5)
        
        # 运行各项基准测试
        tests = [
            ("健康检查基准测试", self.benchmark_health_check, 100),
            ("聊天完成基准测试", self.benchmark_chat_completion, 30),
            ("文档查询基准测试", self.benchmark_document_query, 20),
            ("并发请求基准测试 (10用户)", lambda: self.benchmark_concurrent_requests(10, 60)),
            ("并发请求基准测试 (25用户)", lambda: self.benchmark_concurrent_requests(25, 60)),
        ]
        
        results = []
        
        for test_name, test_func, *args in tests:
            self.logger.info(f"执行: {test_name}")
            
            try:
                if args:
                    result = await test_func(args[0])
                else:
                    result = await test_func()
                
                results.append(result)
                self.logger.info(f"{test_name} 完成")
                
                # 测试间休息
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"{test_name} 失败: {e}")
        
        self.results.extend(results)
        return results
    
    def export_benchmark_results(self, filename_prefix: str = "benchmark"):
        """导出基准测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出CSV格式
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            if self.results:
                fieldnames = list(asdict(self.results[0]).keys())
                # 排除复杂的system_info字段
                fieldnames = [f for f in fieldnames if f != 'system_info']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = asdict(result)
                    row.pop('system_info', None)  # 移除复杂字段
                    row['timestamp'] = result.timestamp.isoformat()
                    writer.writerow(row)
        
        # 导出JSON格式（包含完整信息）
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w') as jsonfile:
            results_data = []
            for result in self.results:
                result_dict = asdict(result)
                result_dict['timestamp'] = result.timestamp.isoformat()
                results_data.append(result_dict)
            
            json.dump({
                'benchmark_results': results_data,
                'summary': self.generate_summary()
            }, jsonfile, indent=2)
        
        self.logger.info(f"基准测试结果已导出: {csv_filename}, {json_filename}")
        return csv_filename, json_filename
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成基准测试摘要"""
        if not self.results:
            return {}
        
        # 按测试类型分组
        test_groups = {}
        for result in self.results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)
        
        summary = {
            'total_tests': len(self.results),
            'test_types': len(test_groups),
            'overall_stats': {
                'avg_response_time': statistics.mean([r.response_time for r in self.results]),
                'avg_success_rate': statistics.mean([r.success_rate for r in self.results]),
                'avg_throughput': statistics.mean([r.throughput for r in self.results]),
                'avg_cpu_usage': statistics.mean([r.cpu_usage for r in self.results]),
                'avg_memory_usage': statistics.mean([r.memory_usage for r in self.results])
            },
            'test_type_stats': {}
        }
        
        # 各测试类型统计
        for test_type, results in test_groups.items():
            summary['test_type_stats'][test_type] = {
                'count': len(results),
                'avg_response_time': statistics.mean([r.response_time for r in results]),
                'avg_success_rate': statistics.mean([r.success_rate for r in results]),
                'avg_throughput': statistics.mean([r.throughput for r in results]),
                'p95_response_time': statistics.mean([r.p95_response_time for r in results]),
                'p99_response_time': statistics.mean([r.p99_response_time for r in results])
            }
        
        # 系统信息
        if self.results:
            summary['system_info'] = self.results[0].system_info
        
        return summary
    
    def print_summary(self):
        """打印基准测试摘要"""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("基准测试摘要报告")
        print("="*60)
        
        print(f"总测试数: {summary['total_tests']}")
        print(f"测试类型: {summary['test_types']}")
        
        if 'overall_stats' in summary:
            stats = summary['overall_stats']
            print(f"\n整体统计:")
            print(f"  平均响应时间: {stats['avg_response_time']:.3f}s")
            print(f"  平均成功率: {stats['avg_success_rate']:.1f}%")
            print(f"  平均吞吐量: {stats['avg_throughput']:.2f} 请求/秒")
            print(f"  平均CPU使用率: {stats['avg_cpu_usage']:.1f}%")
            print(f"  平均内存使用率: {stats['avg_memory_usage']:.1f}%")
        
        if 'test_type_stats' in summary:
            print(f"\n各测试类型统计:")
            for test_type, stats in summary['test_type_stats'].items():
                print(f"  {test_type}:")
                print(f"    测试次数: {stats['count']}")
                print(f"    平均响应时间: {stats['avg_response_time']:.3f}s")
                print(f"    平均成功率: {stats['avg_success_rate']:.1f}%")
                print(f"    平均吞吐量: {stats['avg_throughput']:.2f} 请求/秒")
                print(f"    P95响应时间: {stats['p95_response_time']:.3f}s")
        
        # 性能基线建议
        print(f"\n性能基线建议:")
        if 'overall_stats' in summary:
            stats = summary['overall_stats']
            print(f"  响应时间阈值: < {stats['avg_response_time'] * 1.5:.3f}s")
            print(f"  成功率阈值: > {max(90, stats['avg_success_rate'] * 0.95):.1f}%")
            print(f"  吞吐量阈值: > {stats['avg_throughput'] * 0.8:.2f} 请求/秒")


async def main():
    """主函数"""
    print("VoiceHelper 基准测试工具")
    print("="*50)
    
    # 检查服务可用性
    benchmark_suite = BenchmarkSuite()
    
    try:
        # 简单连通性测试
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{benchmark_suite.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print("⚠️ 后端服务不可用，请检查服务状态")
                    return
    except Exception as e:
        print(f"⚠️ 无法连接到后端服务: {e}")
        return
    
    # 选择测试模式
    print("选择测试模式:")
    print("1. 快速基准测试 (推荐)")
    print("2. 完整基准测试套件")
    print("3. 自定义测试")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        # 快速基准测试
        print("执行快速基准测试...")
        
        results = []
        results.append(await benchmark_suite.benchmark_health_check(50))
        results.append(await benchmark_suite.benchmark_chat_completion(10))
        results.append(await benchmark_suite.benchmark_concurrent_requests(5, 30))
        
        benchmark_suite.results = results
        
    elif choice == "2":
        # 完整基准测试套件
        print("执行完整基准测试套件...")
        await benchmark_suite.run_full_benchmark_suite()
        
    elif choice == "3":
        # 自定义测试
        print("自定义测试配置:")
        
        test_health = input("健康检查测试 (y/n): ").lower() == 'y'
        test_chat = input("聊天完成测试 (y/n): ").lower() == 'y'
        test_query = input("文档查询测试 (y/n): ").lower() == 'y'
        test_concurrent = input("并发测试 (y/n): ").lower() == 'y'
        
        results = []
        
        if test_health:
            iterations = int(input("健康检查测试次数 (默认100): ") or "100")
            results.append(await benchmark_suite.benchmark_health_check(iterations))
        
        if test_chat:
            iterations = int(input("聊天测试次数 (默认30): ") or "30")
            results.append(await benchmark_suite.benchmark_chat_completion(iterations))
        
        if test_query:
            iterations = int(input("查询测试次数 (默认20): ") or "20")
            results.append(await benchmark_suite.benchmark_document_query(iterations))
        
        if test_concurrent:
            users = int(input("并发用户数 (默认10): ") or "10")
            duration = int(input("测试持续时间/秒 (默认60): ") or "60")
            results.append(await benchmark_suite.benchmark_concurrent_requests(users, duration))
        
        benchmark_suite.results = results
    
    else:
        print("无效选择")
        return
    
    # 显示结果
    benchmark_suite.print_summary()
    
    # 导出结果
    if input("\n是否导出结果? (y/n): ").lower() == 'y':
        benchmark_suite.export_benchmark_results()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"基准测试执行错误: {e}")
        logging.exception("详细错误信息:")
