"""
综合性能和负载测试用例
测试覆盖：并发处理、内存使用、响应时间、吞吐量、资源利用率、压力测试
"""

import pytest
import asyncio
import time
import psutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch
import numpy as np
import json
import gc
import tracemalloc
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_time: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    error_rate: float
    concurrent_users: int
    timestamp: float


class TestConcurrencyPerformance:
    """并发性能测试"""
    
    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self):
        """测试异步并发请求处理"""
        class AsyncRequestHandler:
            def __init__(self):
                self.request_count = 0
                self.processing_times = []
                self.semaphore = asyncio.Semaphore(100)  # 限制并发数
            
            async def handle_request(self, request_id):
                """处理单个请求"""
                async with self.semaphore:
                    start_time = time.time()
                    
                    # 模拟请求处理
                    await asyncio.sleep(0.01 + np.random.exponential(0.02))  # 模拟变化的处理时间
                    
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    self.request_count += 1
                    
                    return {
                        "request_id": request_id,
                        "processing_time": processing_time,
                        "timestamp": time.time()
                    }
            
            async def load_test(self, concurrent_requests=1000, duration=10):
                """负载测试"""
                start_time = time.time()
                completed_requests = []
                
                # 创建并发请求
                tasks = []
                request_id = 0
                
                while time.time() - start_time < duration:
                    # 批量创建任务
                    batch_size = min(50, concurrent_requests)
                    batch_tasks = []
                    
                    for _ in range(batch_size):
                        task = asyncio.create_task(self.handle_request(request_id))
                        batch_tasks.append(task)
                        request_id += 1
                    
                    # 等待批次完成
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # 处理结果
                    for result in batch_results:
                        if isinstance(result, Exception):
                            continue
                        completed_requests.append(result)
                    
                    # 短暂休息避免过载
                    await asyncio.sleep(0.001)
                
                total_time = time.time() - start_time
                
                return {
                    "total_requests": len(completed_requests),
                    "total_time": total_time,
                    "throughput": len(completed_requests) / total_time,
                    "avg_response_time": np.mean(self.processing_times),
                    "p95_response_time": np.percentile(self.processing_times, 95),
                    "p99_response_time": np.percentile(self.processing_times, 99),
                    "max_response_time": max(self.processing_times),
                    "min_response_time": min(self.processing_times)
                }
        
        # 执行负载测试
        handler = AsyncRequestHandler()
        results = await handler.load_test(concurrent_requests=500, duration=5)
        
        # 验证性能指标
        assert results["throughput"] > 50  # 至少50 RPS
        assert results["avg_response_time"] < 0.1  # 平均响应时间小于100ms
        assert results["p95_response_time"] < 0.2  # 95%请求小于200ms
        assert results["total_requests"] > 200  # 至少处理200个请求
        
        print(f"并发测试结果:")
        print(f"  总请求数: {results['total_requests']}")
        print(f"  吞吐量: {results['throughput']:.2f} RPS")
        print(f"  平均响应时间: {results['avg_response_time']:.3f}s")
        print(f"  P95响应时间: {results['p95_response_time']:.3f}s")
    
    def test_thread_pool_performance(self):
        """测试线程池性能"""
        def cpu_intensive_task(n):
            """CPU密集型任务"""
            result = 0
            for i in range(n):
                result += i ** 2
            return result
        
        def io_intensive_task(duration):
            """IO密集型任务"""
            time.sleep(duration)
            return f"IO task completed in {duration}s"
        
        # 测试不同线程池大小的性能
        task_counts = [10, 50, 100]
        thread_pool_sizes = [1, 5, 10, 20]
        
        performance_results = []
        
        for pool_size in thread_pool_sizes:
            for task_count in task_counts:
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=pool_size) as executor:
                    # 提交CPU密集型任务
                    cpu_futures = [
                        executor.submit(cpu_intensive_task, 10000)
                        for _ in range(task_count // 2)
                    ]
                    
                    # 提交IO密集型任务
                    io_futures = [
                        executor.submit(io_intensive_task, 0.01)
                        for _ in range(task_count // 2)
                    ]
                    
                    # 等待所有任务完成
                    all_futures = cpu_futures + io_futures
                    completed = 0
                    for future in all_futures:
                        try:
                            future.result(timeout=10)
                            completed += 1
                        except Exception as e:
                            print(f"Task failed: {e}")
                
                total_time = time.time() - start_time
                throughput = completed / total_time
                
                performance_results.append({
                    "pool_size": pool_size,
                    "task_count": task_count,
                    "completed_tasks": completed,
                    "total_time": total_time,
                    "throughput": throughput
                })
        
        # 分析最优配置
        best_result = max(performance_results, key=lambda x: x["throughput"])
        
        assert best_result["throughput"] > 10  # 至少10 tasks/s
        assert best_result["completed_tasks"] >= best_result["task_count"] * 0.95  # 95%成功率
        
        print(f"线程池性能测试最佳配置:")
        print(f"  线程池大小: {best_result['pool_size']}")
        print(f"  任务数量: {best_result['task_count']}")
        print(f"  吞吐量: {best_result['throughput']:.2f} tasks/s")
    
    @pytest.mark.asyncio
    async def test_connection_pool_performance(self):
        """测试连接池性能"""
        class MockConnection:
            def __init__(self, connection_id):
                self.id = connection_id
                self.is_busy = False
                self.created_at = time.time()
                self.last_used = time.time()
            
            async def execute_query(self, query):
                """执行查询"""
                self.is_busy = True
                await asyncio.sleep(0.001 + np.random.exponential(0.002))  # 模拟查询时间
                self.last_used = time.time()
                self.is_busy = False
                return f"Query result for: {query}"
        
        class ConnectionPool:
            def __init__(self, max_connections=10):
                self.max_connections = max_connections
                self.connections = []
                self.available_connections = asyncio.Queue()
                self.connection_counter = 0
                self.stats = {
                    "total_requests": 0,
                    "pool_hits": 0,
                    "pool_misses": 0,
                    "wait_times": []
                }
            
            async def get_connection(self):
                """获取连接"""
                start_wait = time.time()
                
                try:
                    # 尝试从池中获取可用连接
                    connection = await asyncio.wait_for(
                        self.available_connections.get(), 
                        timeout=1.0
                    )
                    self.stats["pool_hits"] += 1
                except asyncio.TimeoutError:
                    # 创建新连接
                    if len(self.connections) < self.max_connections:
                        connection = MockConnection(self.connection_counter)
                        self.connection_counter += 1
                        self.connections.append(connection)
                        self.stats["pool_misses"] += 1
                    else:
                        raise Exception("Connection pool exhausted")
                
                wait_time = time.time() - start_wait
                self.stats["wait_times"].append(wait_time)
                self.stats["total_requests"] += 1
                
                return connection
            
            async def return_connection(self, connection):
                """归还连接"""
                if not connection.is_busy:
                    await self.available_connections.put(connection)
            
            async def execute_with_pool(self, query):
                """使用连接池执行查询"""
                connection = await self.get_connection()
                try:
                    result = await connection.execute_query(query)
                    return result
                finally:
                    await self.return_connection(connection)
        
        # 测试不同连接池大小的性能
        pool_sizes = [5, 10, 20]
        concurrent_requests = 100
        
        for pool_size in pool_sizes:
            pool = ConnectionPool(max_connections=pool_size)
            
            start_time = time.time()
            
            # 创建并发查询任务
            tasks = [
                pool.execute_with_pool(f"SELECT * FROM table WHERE id = {i}")
                for i in range(concurrent_requests)
            ]
            
            # 执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            successful_requests = sum(1 for r in results if not isinstance(r, Exception))
            
            # 计算性能指标
            throughput = successful_requests / total_time
            hit_rate = pool.stats["pool_hits"] / pool.stats["total_requests"] if pool.stats["total_requests"] > 0 else 0
            avg_wait_time = np.mean(pool.stats["wait_times"]) if pool.stats["wait_times"] else 0
            
            print(f"连接池大小 {pool_size} 的性能:")
            print(f"  成功请求: {successful_requests}/{concurrent_requests}")
            print(f"  吞吐量: {throughput:.2f} queries/s")
            print(f"  连接池命中率: {hit_rate:.2%}")
            print(f"  平均等待时间: {avg_wait_time:.4f}s")
            
            # 验证性能要求
            assert successful_requests >= concurrent_requests * 0.95  # 95%成功率
            assert throughput > 50  # 至少50 queries/s
            assert avg_wait_time < 0.1  # 平均等待时间小于100ms


class TestMemoryPerformance:
    """内存性能测试"""
    
    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        def memory_intensive_operation(size_mb=10):
            """内存密集型操作"""
            # 分配大量内存
            data = []
            for i in range(size_mb):
                # 每次分配1MB数据
                chunk = [0] * (1024 * 1024 // 8)  # 8字节per int
                data.append(chunk)
            
            # 进行一些操作
            total = 0
            for chunk in data:
                total += len(chunk)
            
            return total, data
        
        # 开始内存跟踪
        tracemalloc.start()
        process = psutil.Process()
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_snapshot = tracemalloc.take_snapshot()
        
        # 执行内存密集型操作
        sizes = [5, 10, 20, 50]
        memory_usage_data = []
        
        for size in sizes:
            gc.collect()  # 强制垃圾回收
            
            before_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            total, data = memory_intensive_operation(size)
            
            after_memory = process.memory_info().rss / 1024 / 1024
            operation_time = time.time() - start_time
            
            memory_increase = after_memory - before_memory
            
            memory_usage_data.append({
                "allocated_size_mb": size,
                "actual_increase_mb": memory_increase,
                "operation_time": operation_time,
                "efficiency": size / memory_increase if memory_increase > 0 else 0
            })
            
            # 清理数据
            del data
            gc.collect()
        
        # 获取最终内存快照
        final_snapshot = tracemalloc.take_snapshot()
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        
        # 分析内存使用模式
        for data in memory_usage_data:
            print(f"分配 {data['allocated_size_mb']}MB:")
            print(f"  实际增长: {data['actual_increase_mb']:.2f}MB")
            print(f"  操作时间: {data['operation_time']:.3f}s")
            print(f"  内存效率: {data['efficiency']:.2f}")
        
        # 验证内存使用合理性
        for data in memory_usage_data:
            # 实际内存增长不应该超过分配大小的3倍（考虑Python开销）
            assert data["actual_increase_mb"] < data["allocated_size_mb"] * 3
            # 操作时间应该合理
            assert data["operation_time"] < 5.0
        
        tracemalloc.stop()
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        class PotentialLeakyService:
            def __init__(self):
                self.cache = {}
                self.request_history = []
                self.connections = []
            
            async def process_request(self, request_id, data):
                """处理请求（可能有内存泄漏）"""
                # 缓存数据（可能导致内存泄漏）
                self.cache[request_id] = data
                
                # 记录请求历史（可能导致内存泄漏）
                self.request_history.append({
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "data_size": len(str(data))
                })
                
                # 模拟创建连接（可能忘记关闭）
                mock_connection = {"id": request_id, "created_at": time.time()}
                self.connections.append(mock_connection)
                
                # 处理数据
                await asyncio.sleep(0.001)
                
                return f"Processed request {request_id}"
            
            def cleanup_old_data(self, max_age=60):
                """清理旧数据"""
                current_time = time.time()
                
                # 清理旧缓存
                old_cache_keys = [
                    key for key, value in self.cache.items()
                    if hasattr(value, 'timestamp') and current_time - value.get('timestamp', 0) > max_age
                ]
                for key in old_cache_keys:
                    del self.cache[key]
                
                # 清理旧历史记录
                self.request_history = [
                    record for record in self.request_history
                    if current_time - record["timestamp"] <= max_age
                ]
                
                # 清理旧连接
                self.connections = [
                    conn for conn in self.connections
                    if current_time - conn["created_at"] <= max_age
                ]
        
        # 内存泄漏检测测试
        service = PotentialLeakyService()
        process = psutil.Process()
        
        memory_samples = []
        
        # 执行多轮请求，监控内存使用
        for round_num in range(5):
            # 记录当前内存使用
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # 执行一批请求
            tasks = []
            for i in range(100):
                request_id = f"req_{round_num}_{i}"
                data = {"content": "x" * 1000, "timestamp": time.time()}  # 1KB数据
                task = service.process_request(request_id, data)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # 每两轮清理一次（模拟不完善的清理策略）
            if round_num % 2 == 1:
                service.cleanup_old_data(max_age=30)
                gc.collect()
        
        # 分析内存增长趋势
        memory_growth = [
            memory_samples[i] - memory_samples[0] 
            for i in range(len(memory_samples))
        ]
        
        print("内存使用情况:")
        for i, (memory, growth) in enumerate(zip(memory_samples, memory_growth)):
            print(f"  轮次 {i}: {memory:.2f}MB (增长: {growth:.2f}MB)")
        
        # 检测内存泄漏
        final_growth = memory_growth[-1]
        max_acceptable_growth = 50  # 50MB
        
        if final_growth > max_acceptable_growth:
            print(f"警告: 检测到潜在内存泄漏，内存增长 {final_growth:.2f}MB")
            
            # 分析泄漏源
            print(f"缓存大小: {len(service.cache)}")
            print(f"历史记录数: {len(service.request_history)}")
            print(f"连接数: {len(service.connections)}")
        
        # 验证内存使用在合理范围内
        assert final_growth < max_acceptable_growth * 2, f"内存泄漏严重: {final_growth:.2f}MB"
    
    def test_garbage_collection_performance(self):
        """测试垃圾回收性能"""
        def create_circular_references(count=1000):
            """创建循环引用对象"""
            objects = []
            for i in range(count):
                obj = {"id": i, "data": [0] * 100}
                if objects:
                    obj["ref"] = objects[-1]  # 创建引用
                    objects[-1]["back_ref"] = obj  # 创建循环引用
                objects.append(obj)
            return objects
        
        # 测试不同垃圾回收策略的性能
        gc_stats = []
        
        # 禁用自动垃圾回收
        gc.disable()
        
        try:
            for iteration in range(3):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # 创建大量对象
                objects_list = []
                for batch in range(10):
                    objects = create_circular_references(500)
                    objects_list.append(objects)
                
                after_creation_memory = psutil.Process().memory_info().rss / 1024 / 1024
                after_creation_time = time.time()
                
                # 删除引用
                del objects_list
                
                # 手动触发垃圾回收
                gc_start = time.time()
                collected = gc.collect()
                gc_time = time.time() - gc_start
                
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                total_time = time.time() - start_time
                
                gc_stats.append({
                    "iteration": iteration,
                    "creation_time": after_creation_time - start_time,
                    "gc_time": gc_time,
                    "total_time": total_time,
                    "memory_before": start_memory,
                    "memory_after_creation": after_creation_memory,
                    "memory_after_gc": final_memory,
                    "memory_freed": after_creation_memory - final_memory,
                    "objects_collected": collected
                })
        
        finally:
            gc.enable()
        
        # 分析垃圾回收性能
        avg_gc_time = np.mean([stat["gc_time"] for stat in gc_stats])
        avg_memory_freed = np.mean([stat["memory_freed"] for stat in gc_stats])
        
        print("垃圾回收性能统计:")
        for stat in gc_stats:
            print(f"  迭代 {stat['iteration']}:")
            print(f"    创建时间: {stat['creation_time']:.3f}s")
            print(f"    GC时间: {stat['gc_time']:.3f}s")
            print(f"    释放内存: {stat['memory_freed']:.2f}MB")
            print(f"    回收对象: {stat['objects_collected']}")
        
        print(f"平均GC时间: {avg_gc_time:.3f}s")
        print(f"平均释放内存: {avg_memory_freed:.2f}MB")
        
        # 验证垃圾回收效果
        assert avg_gc_time < 1.0, f"垃圾回收时间过长: {avg_gc_time:.3f}s"
        assert avg_memory_freed > 0, "垃圾回收未释放内存"


class TestResponseTimePerformance:
    """响应时间性能测试"""
    
    @pytest.mark.asyncio
    async def test_api_response_time_distribution(self):
        """测试API响应时间分布"""
        class APIService:
            def __init__(self):
                self.response_times = []
                self.error_count = 0
            
            async def handle_request(self, request_type="normal", complexity=1):
                """处理API请求"""
                start_time = time.time()
                
                try:
                    # 根据请求类型模拟不同的处理时间
                    if request_type == "fast":
                        base_time = 0.001
                    elif request_type == "normal":
                        base_time = 0.01
                    elif request_type == "slow":
                        base_time = 0.1
                    elif request_type == "complex":
                        base_time = 0.05 * complexity
                    else:
                        base_time = 0.01
                    
                    # 添加随机变化
                    processing_time = base_time + np.random.exponential(base_time * 0.5)
                    
                    # 模拟偶发的慢请求
                    if np.random.random() < 0.05:  # 5%概率
                        processing_time *= 5
                    
                    await asyncio.sleep(processing_time)
                    
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    
                    return {
                        "success": True,
                        "response_time": response_time,
                        "request_type": request_type
                    }
                
                except Exception as e:
                    self.error_count += 1
                    return {
                        "success": False,
                        "error": str(e),
                        "response_time": time.time() - start_time
                    }
            
            def get_performance_stats(self):
                """获取性能统计"""
                if not self.response_times:
                    return {}
                
                times = np.array(self.response_times)
                
                return {
                    "total_requests": len(self.response_times),
                    "error_count": self.error_count,
                    "error_rate": self.error_count / (len(self.response_times) + self.error_count),
                    "mean_response_time": np.mean(times),
                    "median_response_time": np.median(times),
                    "p90_response_time": np.percentile(times, 90),
                    "p95_response_time": np.percentile(times, 95),
                    "p99_response_time": np.percentile(times, 99),
                    "max_response_time": np.max(times),
                    "min_response_time": np.min(times),
                    "std_response_time": np.std(times)
                }
        
        # 执行混合负载测试
        api_service = APIService()
        
        # 创建不同类型的请求
        request_mix = [
            ("fast", 1, 200),      # 200个快速请求
            ("normal", 1, 500),    # 500个普通请求
            ("slow", 1, 100),      # 100个慢请求
            ("complex", 3, 50),    # 50个复杂请求
        ]
        
        all_tasks = []
        for request_type, complexity, count in request_mix:
            for _ in range(count):
                task = api_service.handle_request(request_type, complexity)
                all_tasks.append(task)
        
        # 随机打乱请求顺序
        np.random.shuffle(all_tasks)
        
        # 执行所有请求
        start_time = time.time()
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 分析结果
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_requests = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and not r.get("success"))]
        
        stats = api_service.get_performance_stats()
        throughput = len(successful_requests) / total_time
        
        print("API响应时间性能统计:")
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  成功请求: {len(successful_requests)}")
        print(f"  失败请求: {len(failed_requests)}")
        print(f"  错误率: {stats['error_rate']:.2%}")
        print(f"  吞吐量: {throughput:.2f} RPS")
        print(f"  平均响应时间: {stats['mean_response_time']:.3f}s")
        print(f"  中位数响应时间: {stats['median_response_time']:.3f}s")
        print(f"  P90响应时间: {stats['p90_response_time']:.3f}s")
        print(f"  P95响应时间: {stats['p95_response_time']:.3f}s")
        print(f"  P99响应时间: {stats['p99_response_time']:.3f}s")
        print(f"  最大响应时间: {stats['max_response_time']:.3f}s")
        
        # 验证性能要求
        assert stats["error_rate"] < 0.01, f"错误率过高: {stats['error_rate']:.2%}"
        assert stats["p95_response_time"] < 0.5, f"P95响应时间过长: {stats['p95_response_time']:.3f}s"
        assert stats["mean_response_time"] < 0.1, f"平均响应时间过长: {stats['mean_response_time']:.3f}s"
        assert throughput > 100, f"吞吐量过低: {throughput:.2f} RPS"
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self):
        """测试数据库查询性能"""
        class MockDatabase:
            def __init__(self):
                self.query_times = {}
                self.connection_pool_size = 10
                self.active_connections = 0
            
            async def execute_query(self, query_type, complexity=1):
                """执行数据库查询"""
                # 模拟连接池限制
                if self.active_connections >= self.connection_pool_size:
                    await asyncio.sleep(0.01)  # 等待连接可用
                
                self.active_connections += 1
                
                try:
                    # 根据查询类型模拟不同的执行时间
                    base_times = {
                        "select_simple": 0.001,
                        "select_complex": 0.01,
                        "select_join": 0.02,
                        "insert": 0.005,
                        "update": 0.008,
                        "delete": 0.003,
                        "aggregate": 0.015
                    }
                    
                    base_time = base_times.get(query_type, 0.01)
                    query_time = base_time * complexity + np.random.exponential(base_time * 0.3)
                    
                    await asyncio.sleep(query_time)
                    
                    if query_type not in self.query_times:
                        self.query_times[query_type] = []
                    self.query_times[query_type].append(query_time)
                    
                    return {
                        "success": True,
                        "query_type": query_type,
                        "execution_time": query_time,
                        "rows_affected": complexity * 10
                    }
                
                finally:
                    self.active_connections -= 1
            
            def get_query_stats(self):
                """获取查询统计"""
                stats = {}
                for query_type, times in self.query_times.items():
                    times_array = np.array(times)
                    stats[query_type] = {
                        "count": len(times),
                        "mean_time": np.mean(times_array),
                        "median_time": np.median(times_array),
                        "p95_time": np.percentile(times_array, 95),
                        "max_time": np.max(times_array),
                        "min_time": np.min(times_array)
                    }
                return stats
        
        # 执行数据库性能测试
        db = MockDatabase()
        
        # 定义查询组合
        query_workload = [
            ("select_simple", 1, 300),    # 300个简单查询
            ("select_complex", 2, 100),   # 100个复杂查询
            ("select_join", 3, 50),       # 50个连接查询
            ("insert", 1, 150),           # 150个插入
            ("update", 2, 100),           # 100个更新
            ("delete", 1, 50),            # 50个删除
            ("aggregate", 4, 30),         # 30个聚合查询
        ]
        
        # 创建查询任务
        all_queries = []
        for query_type, complexity, count in query_workload:
            for _ in range(count):
                query = db.execute_query(query_type, complexity)
                all_queries.append(query)
        
        # 随机打乱查询顺序模拟真实负载
        np.random.shuffle(all_queries)
        
        # 执行查询
        start_time = time.time()
        results = await asyncio.gather(*all_queries, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 分析结果
        successful_queries = [r for r in results if isinstance(r, dict) and r.get("success")]
        query_stats = db.get_query_stats()
        
        total_queries = len(successful_queries)
        qps = total_queries / total_time
        
        print("数据库查询性能统计:")
        print(f"  总查询数: {total_queries}")
        print(f"  总执行时间: {total_time:.3f}s")
        print(f"  QPS: {qps:.2f}")
        
        for query_type, stats in query_stats.items():
            print(f"  {query_type}:")
            print(f"    查询数: {stats['count']}")
            print(f"    平均时间: {stats['mean_time']:.4f}s")
            print(f"    P95时间: {stats['p95_time']:.4f}s")
            print(f"    最大时间: {stats['max_time']:.4f}s")
        
        # 验证性能要求
        assert qps > 500, f"QPS过低: {qps:.2f}"
        
        # 验证各类查询的性能
        performance_requirements = {
            "select_simple": 0.01,   # 简单查询 < 10ms
            "select_complex": 0.05,  # 复杂查询 < 50ms
            "insert": 0.02,          # 插入 < 20ms
            "update": 0.03,          # 更新 < 30ms
        }
        
        for query_type, max_time in performance_requirements.items():
            if query_type in query_stats:
                actual_p95 = query_stats[query_type]["p95_time"]
                assert actual_p95 < max_time, f"{query_type} P95时间过长: {actual_p95:.4f}s > {max_time}s"


class TestStressAndLoadTesting:
    """压力和负载测试"""
    
    @pytest.mark.asyncio
    async def test_gradual_load_increase(self):
        """测试渐进式负载增加"""
        class LoadTestService:
            def __init__(self):
                self.active_requests = 0
                self.max_concurrent = 0
                self.request_history = []
                self.error_count = 0
            
            async def handle_request(self, request_id):
                """处理请求"""
                self.active_requests += 1
                self.max_concurrent = max(self.max_concurrent, self.active_requests)
                
                start_time = time.time()
                
                try:
                    # 模拟请求处理，处理时间随并发数增加而增加
                    base_time = 0.01
                    load_factor = min(self.active_requests / 100, 2.0)  # 最多2倍延迟
                    processing_time = base_time * (1 + load_factor)
                    
                    await asyncio.sleep(processing_time + np.random.exponential(0.005))
                    
                    response_time = time.time() - start_time
                    
                    self.request_history.append({
                        "request_id": request_id,
                        "response_time": response_time,
                        "concurrent_requests": self.active_requests,
                        "timestamp": time.time()
                    })
                    
                    return {"success": True, "response_time": response_time}
                
                except Exception as e:
                    self.error_count += 1
                    return {"success": False, "error": str(e)}
                
                finally:
                    self.active_requests -= 1
        
        # 执行渐进式负载测试
        service = LoadTestService()
        load_phases = [
            (10, 5),   # 10个并发用户，持续5秒
            (25, 5),   # 25个并发用户，持续5秒
            (50, 5),   # 50个并发用户，持续5秒
            (100, 5),  # 100个并发用户，持续5秒
            (150, 5),  # 150个并发用户，持续5秒
        ]
        
        phase_results = []
        
        for phase_num, (concurrent_users, duration) in enumerate(load_phases):
            print(f"执行负载阶段 {phase_num + 1}: {concurrent_users} 并发用户，{duration}秒")
            
            phase_start = time.time()
            request_id_counter = phase_num * 1000
            
            # 创建并发任务
            active_tasks = set()
            completed_requests = []
            
            while time.time() - phase_start < duration:
                # 维持目标并发数
                while len(active_tasks) < concurrent_users:
                    request_id_counter += 1
                    task = asyncio.create_task(
                        service.handle_request(f"req_{request_id_counter}")
                    )
                    active_tasks.add(task)
                
                # 检查完成的任务
                done_tasks = {task for task in active_tasks if task.done()}
                for task in done_tasks:
                    try:
                        result = await task
                        completed_requests.append(result)
                    except Exception as e:
                        print(f"Task failed: {e}")
                    active_tasks.remove(task)
                
                await asyncio.sleep(0.01)  # 短暂休息
            
            # 等待剩余任务完成
            if active_tasks:
                remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, dict):
                        completed_requests.append(result)
            
            # 分析阶段结果
            successful_requests = [r for r in completed_requests if r.get("success")]
            failed_requests = [r for r in completed_requests if not r.get("success")]
            
            if successful_requests:
                response_times = [r["response_time"] for r in successful_requests]
                
                phase_result = {
                    "phase": phase_num + 1,
                    "concurrent_users": concurrent_users,
                    "duration": duration,
                    "total_requests": len(completed_requests),
                    "successful_requests": len(successful_requests),
                    "failed_requests": len(failed_requests),
                    "error_rate": len(failed_requests) / len(completed_requests),
                    "throughput": len(successful_requests) / duration,
                    "avg_response_time": np.mean(response_times),
                    "p95_response_time": np.percentile(response_times, 95),
                    "max_concurrent": service.max_concurrent
                }
                
                phase_results.append(phase_result)
                
                print(f"  完成请求: {len(successful_requests)}")
                print(f"  失败请求: {len(failed_requests)}")
                print(f"  错误率: {phase_result['error_rate']:.2%}")
                print(f"  吞吐量: {phase_result['throughput']:.2f} RPS")
                print(f"  平均响应时间: {phase_result['avg_response_time']:.3f}s")
                print(f"  P95响应时间: {phase_result['p95_response_time']:.3f}s")
        
        # 分析负载测试结果
        print("\n负载测试总结:")
        for result in phase_results:
            print(f"阶段 {result['phase']} ({result['concurrent_users']} 用户):")
            print(f"  吞吐量: {result['throughput']:.2f} RPS")
            print(f"  错误率: {result['error_rate']:.2%}")
            print(f"  P95响应时间: {result['p95_response_time']:.3f}s")
        
        # 验证系统在负载下的表现
        for result in phase_results:
            # 错误率应该保持在合理范围内
            assert result["error_rate"] < 0.05, f"阶段 {result['phase']} 错误率过高: {result['error_rate']:.2%}"
            
            # 响应时间不应该过长
            assert result["p95_response_time"] < 1.0, f"阶段 {result['phase']} P95响应时间过长: {result['p95_response_time']:.3f}s"
        
        # 检查系统是否能处理最高负载
        highest_load_result = phase_results[-1]
        assert highest_load_result["throughput"] > 50, f"最高负载下吞吐量过低: {highest_load_result['throughput']:.2f} RPS"
    
    def test_resource_exhaustion_scenarios(self):
        """测试资源耗尽场景"""
        class ResourceLimitedService:
            def __init__(self, max_memory_mb=100, max_connections=50, max_cpu_time=1.0):
                self.max_memory_mb = max_memory_mb
                self.max_connections = max_connections
                self.max_cpu_time = max_cpu_time
                
                self.current_memory_mb = 0
                self.current_connections = 0
                self.active_operations = []
            
            def allocate_memory(self, size_mb):
                """分配内存"""
                if self.current_memory_mb + size_mb > self.max_memory_mb:
                    raise Exception(f"Memory limit exceeded: {self.current_memory_mb + size_mb} > {self.max_memory_mb}")
                
                self.current_memory_mb += size_mb
                return f"Allocated {size_mb}MB"
            
            def create_connection(self):
                """创建连接"""
                if self.current_connections >= self.max_connections:
                    raise Exception(f"Connection limit exceeded: {self.current_connections} >= {self.max_connections}")
                
                self.current_connections += 1
                connection_id = f"conn_{self.current_connections}"
                return connection_id
            
            def cpu_intensive_operation(self, duration):
                """CPU密集型操作"""
                if duration > self.max_cpu_time:
                    raise Exception(f"CPU time limit exceeded: {duration} > {self.max_cpu_time}")
                
                start_time = time.time()
                # 模拟CPU密集型计算
                result = 0
                while time.time() - start_time < duration:
                    result += 1
                
                return result
            
            def release_memory(self, size_mb):
                """释放内存"""
                self.current_memory_mb = max(0, self.current_memory_mb - size_mb)
            
            def close_connection(self, connection_id):
                """关闭连接"""
                self.current_connections = max(0, self.current_connections - 1)
        
        # 测试各种资源耗尽场景
        service = ResourceLimitedService()
        
        # 1. 测试内存限制
        print("测试内存限制:")
        memory_allocations = []
        try:
            for i in range(15):  # 尝试分配150MB（超过100MB限制）
                try:
                    result = service.allocate_memory(10)
                    memory_allocations.append(10)
                    print(f"  分配 10MB: 成功 (总计: {service.current_memory_mb}MB)")
                except Exception as e:
                    print(f"  分配 10MB: 失败 - {e}")
                    break
        finally:
            # 清理内存
            for size in memory_allocations:
                service.release_memory(size)
        
        # 2. 测试连接限制
        print("\n测试连接限制:")
        connections = []
        try:
            for i in range(60):  # 尝试创建60个连接（超过50个限制）
                try:
                    conn_id = service.create_connection()
                    connections.append(conn_id)
                    if i % 10 == 0:
                        print(f"  创建连接 {i+1}: 成功 (总计: {service.current_connections})")
                except Exception as e:
                    print(f"  创建连接 {i+1}: 失败 - {e}")
                    break
        finally:
            # 清理连接
            for conn_id in connections:
                service.close_connection(conn_id)
        
        # 3. 测试CPU时间限制
        print("\n测试CPU时间限制:")
        cpu_durations = [0.1, 0.5, 0.8, 1.2, 1.5]  # 最后两个应该失败
        
        for duration in cpu_durations:
            try:
                start_time = time.time()
                result = service.cpu_intensive_operation(duration)
                actual_time = time.time() - start_time
                print(f"  CPU操作 {duration}s: 成功 (实际: {actual_time:.3f}s, 结果: {result})")
            except Exception as e:
                print(f"  CPU操作 {duration}s: 失败 - {e}")
        
        # 验证资源限制正常工作
        assert service.current_memory_mb == 0, "内存未完全释放"
        assert service.current_connections == 0, "连接未完全关闭"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
