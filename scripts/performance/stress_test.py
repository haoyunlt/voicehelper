"""
压力测试脚本
测试系统在极限负载下的表现，找出系统的性能瓶颈和崩溃点
"""

import asyncio
import aiohttp
import time
import json
import random
import threading
import queue
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics
import csv
from datetime import datetime
import signal
import sys


@dataclass
class StressTestResult:
    """压力测试结果"""
    timestamp: float
    request_type: str
    response_time: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取系统资源使用情况
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 获取网络统计
                net_io = psutil.net_io_counters()
                
                metric = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'disk_percent': disk.percent,
                    'network_sent_mb': net_io.bytes_sent / (1024**2),
                    'network_recv_mb': net_io.bytes_recv / (1024**2)
                }
                
                self.metrics.append(metric)
                
            except Exception as e:
                logging.error(f"系统监控错误: {e}")
            
            time.sleep(1)
    
    def get_current_metrics(self):
        """获取当前指标"""
        if self.metrics:
            return self.metrics[-1]
        return {}


class AsyncStressTester:
    """异步压力测试器"""
    
    def __init__(self, base_url: str, max_concurrent: int = 1000):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.results = []
        self.system_monitor = SystemMonitor()
        self.test_running = False
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def make_request(self, session: aiohttp.ClientSession, request_config: Dict) -> StressTestResult:
        """发起单个请求"""
        start_time = time.time()
        current_metrics = self.system_monitor.get_current_metrics()
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with session.request(
                method=request_config['method'],
                url=f"{self.base_url}{request_config['path']}",
                json=request_config.get('data'),
                headers=request_config.get('headers', {}),
                timeout=timeout
            ) as response:
                
                response_time = time.time() - start_time
                
                # 读取响应内容（用于验证）
                try:
                    if response.content_type == 'application/json':
                        await response.json()
                    else:
                        await response.text()
                except:
                    pass  # 忽略响应解析错误
                
                return StressTestResult(
                    timestamp=start_time,
                    request_type=request_config['name'],
                    response_time=response_time,
                    status_code=response.status,
                    success=200 <= response.status < 400,
                    memory_usage=current_metrics.get('memory_percent', 0),
                    cpu_usage=current_metrics.get('cpu_percent', 0)
                )
                
        except asyncio.TimeoutError:
            return StressTestResult(
                timestamp=start_time,
                request_type=request_config['name'],
                response_time=time.time() - start_time,
                status_code=408,
                success=False,
                error_message="Request timeout",
                memory_usage=current_metrics.get('memory_percent', 0),
                cpu_usage=current_metrics.get('cpu_percent', 0)
            )
            
        except Exception as e:
            return StressTestResult(
                timestamp=start_time,
                request_type=request_config['name'],
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                error_message=str(e),
                memory_usage=current_metrics.get('memory_percent', 0),
                cpu_usage=current_metrics.get('cpu_percent', 0)
            )
    
    async def run_concurrent_requests(self, request_configs: List[Dict], concurrent_users: int, duration_seconds: int):
        """运行并发请求"""
        self.test_running = True
        self.system_monitor.start_monitoring()
        
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                
                start_time = time.time()
                tasks = []
                
                self.logger.info(f"开始压力测试: {concurrent_users}并发用户, 持续{duration_seconds}秒")
                
                # 创建并发任务
                while time.time() - start_time < duration_seconds and self.test_running:
                    
                    # 控制并发数量
                    if len(tasks) >= concurrent_users:
                        # 等待一些任务完成
                        done_tasks = []
                        for task in tasks[:concurrent_users // 2]:
                            if task.done():
                                done_tasks.append(task)
                                result = await task
                                self.results.append(result)
                        
                        # 移除已完成的任务
                        for task in done_tasks:
                            tasks.remove(task)
                    
                    # 随机选择请求类型
                    request_config = random.choice(request_configs)
                    
                    # 创建新任务
                    task = asyncio.create_task(
                        self.make_request(session, request_config)
                    )
                    tasks.append(task)
                    
                    # 控制请求频率
                    await asyncio.sleep(0.01)  # 10ms间隔
                
                # 等待所有剩余任务完成
                self.logger.info("等待剩余请求完成...")
                for task in tasks:
                    try:
                        result = await asyncio.wait_for(task, timeout=30)
                        self.results.append(result)
                    except asyncio.TimeoutError:
                        task.cancel()
                        self.logger.warning("任务超时被取消")
                
        except Exception as e:
            self.logger.error(f"压力测试执行错误: {e}")
        
        finally:
            self.system_monitor.stop_monitoring()
            self.test_running = False
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """获取测试统计信息"""
        if not self.results:
            return {}
        
        # 基础统计
        total_requests = len(self.results)
        successful_requests = [r for r in self.results if r.success]
        failed_requests = [r for r in self.results if not r.success]
        
        response_times = [r.response_time for r in self.results]
        successful_response_times = [r.response_time for r in successful_requests]
        
        # 按请求类型分组统计
        request_type_stats = {}
        for result in self.results:
            if result.request_type not in request_type_stats:
                request_type_stats[result.request_type] = {
                    'total': 0,
                    'success': 0,
                    'failed': 0,
                    'response_times': []
                }
            
            stats = request_type_stats[result.request_type]
            stats['total'] += 1
            stats['response_times'].append(result.response_time)
            
            if result.success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        # 计算统计指标
        statistics_data = {
            'total_requests': total_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / total_requests * 100 if total_requests > 0 else 0,
            'error_rate': len(failed_requests) / total_requests * 100 if total_requests > 0 else 0,
        }
        
        if response_times:
            statistics_data.update({
                'avg_response_time': statistics.mean(response_times),
                'median_response_time': statistics.median(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': self._percentile(response_times, 95),
                'p99_response_time': self._percentile(response_times, 99),
                'response_time_std': statistics.stdev(response_times) if len(response_times) > 1 else 0
            })
        
        if successful_response_times:
            statistics_data.update({
                'avg_successful_response_time': statistics.mean(successful_response_times),
                'p95_successful_response_time': self._percentile(successful_response_times, 95)
            })
        
        # 吞吐量计算
        if self.results:
            test_duration = max(r.timestamp for r in self.results) - min(r.timestamp for r in self.results)
            if test_duration > 0:
                statistics_data['throughput_rps'] = total_requests / test_duration
                statistics_data['successful_throughput_rps'] = len(successful_requests) / test_duration
        
        # 系统资源统计
        if self.system_monitor.metrics:
            cpu_usage = [m['cpu_percent'] for m in self.system_monitor.metrics]
            memory_usage = [m['memory_percent'] for m in self.system_monitor.metrics]
            
            statistics_data.update({
                'avg_cpu_usage': statistics.mean(cpu_usage),
                'max_cpu_usage': max(cpu_usage),
                'avg_memory_usage': statistics.mean(memory_usage),
                'max_memory_usage': max(memory_usage)
            })
        
        statistics_data['request_type_stats'] = request_type_stats
        
        return statistics_data
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def export_results(self, filename: str):
        """导出测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出详细结果
        csv_filename = f"{filename}_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            writer.writerow([
                'Timestamp', 'Request Type', 'Response Time', 'Status Code',
                'Success', 'Error Message', 'Memory Usage %', 'CPU Usage %'
            ])
            
            # 写入数据
            for result in self.results:
                writer.writerow([
                    datetime.fromtimestamp(result.timestamp).isoformat(),
                    result.request_type,
                    result.response_time,
                    result.status_code,
                    result.success,
                    result.error_message or '',
                    result.memory_usage,
                    result.cpu_usage
                ])
        
        # 导出统计摘要
        stats = self.get_test_statistics()
        summary_filename = f"{filename}_summary_{timestamp}.json"
        with open(summary_filename, 'w') as jsonfile:
            json.dump(stats, jsonfile, indent=2)
        
        self.logger.info(f"结果已导出: {csv_filename}, {summary_filename}")
        
        return csv_filename, summary_filename


class VoiceHelperStressTest:
    """VoiceHelper压力测试主类"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8080"
        self.algo_url = "http://localhost:8000"
        self.tester = None
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print("\n收到中断信号，正在停止测试...")
        if self.tester:
            self.tester.test_running = False
        sys.exit(0)
    
    def get_request_configs(self) -> List[Dict]:
        """获取请求配置"""
        return [
            # 健康检查
            {
                'name': 'health_check',
                'method': 'GET',
                'path': '/health',
                'headers': {}
            },
            
            # Ping接口
            {
                'name': 'ping',
                'method': 'GET',
                'path': '/api/v1/ping',
                'headers': {'Authorization': 'Bearer stress_test_token'}
            },
            
            # 聊天完成
            {
                'name': 'chat_completion',
                'method': 'POST',
                'path': '/api/v1/chat/completions',
                'headers': {
                    'Authorization': 'Bearer stress_test_token',
                    'Content-Type': 'application/json'
                },
                'data': {
                    'conversation_id': f'stress_test_{random.randint(1, 1000)}',
                    'messages': [
                        {
                            'role': 'user',
                            'content': f'压力测试消息 {random.randint(1, 10000)}'
                        }
                    ],
                    'top_k': 3,
                    'temperature': 0.5
                }
            },
            
            # 集成服务列表
            {
                'name': 'integration_services',
                'method': 'GET',
                'path': '/api/v1/integrations/services',
                'headers': {'Authorization': 'Bearer stress_test_token'}
            },
            
            # 集成服务健康检查
            {
                'name': 'integration_health',
                'method': 'GET',
                'path': '/api/v1/integrations/health',
                'headers': {'Authorization': 'Bearer stress_test_token'}
            }
        ]
    
    def get_algo_request_configs(self) -> List[Dict]:
        """获取算法服务请求配置"""
        return [
            # 算法服务健康检查
            {
                'name': 'algo_health',
                'method': 'GET',
                'path': '/health',
                'headers': {}
            },
            
            # 文档查询
            {
                'name': 'document_query',
                'method': 'POST',
                'path': '/query',
                'headers': {'Content-Type': 'application/json'},
                'data': {
                    'messages': [
                        {
                            'role': 'user',
                            'content': f'压力测试查询 {random.randint(1, 1000)}'
                        }
                    ],
                    'top_k': 3,
                    'temperature': 0.3
                }
            }
        ]
    
    async def run_backend_stress_test(self, concurrent_users: int, duration_seconds: int):
        """运行后端压力测试"""
        print(f"开始后端压力测试: {concurrent_users}并发用户, 持续{duration_seconds}秒")
        
        self.tester = AsyncStressTester(self.backend_url, concurrent_users)
        request_configs = self.get_request_configs()
        
        await self.tester.run_concurrent_requests(
            request_configs, concurrent_users, duration_seconds
        )
        
        # 获取统计信息
        stats = self.tester.get_test_statistics()
        
        print("\n后端压力测试结果:")
        print("="*50)
        self._print_statistics(stats)
        
        # 导出结果
        self.tester.export_results("backend_stress_test")
        
        return stats
    
    async def run_algo_stress_test(self, concurrent_users: int, duration_seconds: int):
        """运行算法服务压力测试"""
        print(f"开始算法服务压力测试: {concurrent_users}并发用户, 持续{duration_seconds}秒")
        
        self.tester = AsyncStressTester(self.algo_url, concurrent_users)
        request_configs = self.get_algo_request_configs()
        
        await self.tester.run_concurrent_requests(
            request_configs, concurrent_users, duration_seconds
        )
        
        # 获取统计信息
        stats = self.tester.get_test_statistics()
        
        print("\n算法服务压力测试结果:")
        print("="*50)
        self._print_statistics(stats)
        
        # 导出结果
        self.tester.export_results("algo_stress_test")
        
        return stats
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """打印统计信息"""
        print(f"总请求数: {stats.get('total_requests', 0)}")
        print(f"成功请求: {stats.get('successful_requests', 0)}")
        print(f"失败请求: {stats.get('failed_requests', 0)}")
        print(f"成功率: {stats.get('success_rate', 0):.2f}%")
        print(f"错误率: {stats.get('error_rate', 0):.2f}%")
        
        if 'avg_response_time' in stats:
            print(f"平均响应时间: {stats['avg_response_time']:.3f}s")
            print(f"中位数响应时间: {stats['median_response_time']:.3f}s")
            print(f"P95响应时间: {stats['p95_response_time']:.3f}s")
            print(f"P99响应时间: {stats['p99_response_time']:.3f}s")
            print(f"最大响应时间: {stats['max_response_time']:.3f}s")
        
        if 'throughput_rps' in stats:
            print(f"吞吐量: {stats['throughput_rps']:.2f} 请求/秒")
        
        if 'avg_cpu_usage' in stats:
            print(f"平均CPU使用率: {stats['avg_cpu_usage']:.1f}%")
            print(f"最大CPU使用率: {stats['max_cpu_usage']:.1f}%")
            print(f"平均内存使用率: {stats['avg_memory_usage']:.1f}%")
            print(f"最大内存使用率: {stats['max_memory_usage']:.1f}%")
        
        # 打印各请求类型统计
        if 'request_type_stats' in stats:
            print("\n各请求类型统计:")
            for req_type, type_stats in stats['request_type_stats'].items():
                success_rate = type_stats['success'] / type_stats['total'] * 100
                avg_time = statistics.mean(type_stats['response_times'])
                print(f"  {req_type}: {type_stats['total']}次, 成功率{success_rate:.1f}%, 平均{avg_time:.3f}s")
    
    async def run_escalating_stress_test(self):
        """运行递增压力测试"""
        print("开始递增压力测试...")
        
        # 测试配置：逐步增加并发用户数
        test_phases = [
            (10, 60),   # 10用户, 1分钟
            (25, 60),   # 25用户, 1分钟
            (50, 60),   # 50用户, 1分钟
            (100, 60),  # 100用户, 1分钟
            (200, 60),  # 200用户, 1分钟
            (500, 60),  # 500用户, 1分钟
            (1000, 60), # 1000用户, 1分钟
        ]
        
        all_results = []
        
        for phase_num, (users, duration) in enumerate(test_phases, 1):
            print(f"\n阶段 {phase_num}: {users}并发用户")
            
            try:
                # 后端测试
                backend_stats = await self.run_backend_stress_test(users, duration)
                
                # 等待系统恢复
                print("等待系统恢复...")
                await asyncio.sleep(30)
                
                # 算法服务测试
                algo_stats = await self.run_algo_stress_test(users // 2, duration)  # 算法服务并发数减半
                
                phase_result = {
                    'phase': phase_num,
                    'concurrent_users': users,
                    'duration': duration,
                    'backend_stats': backend_stats,
                    'algo_stats': algo_stats
                }
                
                all_results.append(phase_result)
                
                # 检查是否达到系统极限
                if (backend_stats.get('error_rate', 0) > 50 or 
                    backend_stats.get('avg_response_time', 0) > 10):
                    print("⚠️ 系统已达到极限，停止测试")
                    break
                
            except Exception as e:
                print(f"阶段 {phase_num} 测试失败: {e}")
                break
            
            # 阶段间休息
            if phase_num < len(test_phases):
                print("阶段间休息60秒...")
                await asyncio.sleep(60)
        
        # 导出综合报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"escalating_stress_test_report_{timestamp}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n递增压力测试完成，报告已保存: {report_filename}")
        
        return all_results


async def main():
    """主函数"""
    stress_test = VoiceHelperStressTest()
    
    print("VoiceHelper 压力测试工具")
    print("="*50)
    
    # 选择测试模式
    test_mode = input("选择测试模式 (1: 后端测试, 2: 算法服务测试, 3: 递增压力测试): ")
    
    if test_mode == "1":
        users = int(input("并发用户数 (默认100): ") or "100")
        duration = int(input("测试持续时间(秒) (默认300): ") or "300")
        await stress_test.run_backend_stress_test(users, duration)
        
    elif test_mode == "2":
        users = int(input("并发用户数 (默认50): ") or "50")
        duration = int(input("测试持续时间(秒) (默认300): ") or "300")
        await stress_test.run_algo_stress_test(users, duration)
        
    elif test_mode == "3":
        await stress_test.run_escalating_stress_test()
        
    else:
        print("无效的测试模式")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试执行错误: {e}")
        logging.exception("详细错误信息:")
