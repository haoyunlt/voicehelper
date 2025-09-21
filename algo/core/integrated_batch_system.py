"""
集成批量化系统

整合批量处理、请求合并和动态调整功能
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from .batch_processor import LLMBatchProcessor
from .request_merger import AdvancedRequestMerger, MergeableRequest
from .dynamic_batcher import AdaptiveBatchProcessor, PerformanceMonitor, BatchMetrics

logger = logging.getLogger(__name__)


@dataclass
class ProcessingRequest:
    """处理请求"""
    id: str
    content: str
    model: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 0=normal, 1=high, -1=low
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None
    
    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()
    
    def to_mergeable_request(self) -> MergeableRequest:
        """转换为可合并请求"""
        return MergeableRequest(
            id=self.id,
            content=self.content,
            model=self.model,
            parameters=self.parameters,
            future=self.future
        )


@dataclass
class BatchingConfig:
    """批处理配置"""
    # 基础配置
    initial_batch_size: int = 4
    min_batch_size: int = 1
    max_batch_size: int = 32
    max_wait_time: float = 0.1
    
    # 合并配置
    enable_request_merging: bool = True
    similarity_threshold: float = 0.85
    merge_window: float = 5.0
    max_merge_group_size: int = 10
    
    # 动态调整配置
    enable_dynamic_adjustment: bool = True
    adjustment_factor: float = 0.2
    monitor_interval: float = 5.0
    
    # 性能配置
    enable_performance_monitoring: bool = True
    history_size: int = 100
    
    # 优先级配置
    enable_priority_scheduling: bool = True
    high_priority_batch_size: int = 2
    priority_timeout_multiplier: float = 0.5


class IntegratedBatchSystem:
    """集成批量化系统"""
    
    def __init__(
        self,
        processing_func: Callable[[List[Dict[str, Any]]], Awaitable[List[Any]]],
        config: Optional[BatchingConfig] = None
    ):
        self.processing_func = processing_func
        self.config = config or BatchingConfig()
        
        # 核心组件
        self.request_merger = AdvancedRequestMerger(
            similarity_threshold=self.config.similarity_threshold,
            merge_window=self.config.merge_window,
            max_group_size=self.config.max_merge_group_size
        ) if self.config.enable_request_merging else None
        
        self.performance_monitor = PerformanceMonitor(
            history_size=self.config.history_size
        ) if self.config.enable_performance_monitoring else None
        
        # 处理队列 (按优先级分组)
        self.high_priority_queue = asyncio.Queue()
        self.normal_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        
        # 批处理状态
        self._running = False
        self._processing_tasks: List[asyncio.Task] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # 动态批次大小
        self.current_batch_size = self.config.initial_batch_size
        self.batch_size_history = []
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'processed_requests': 0,
            'merged_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'avg_queue_time': 0.0,
            'throughput': 0.0,
            'batch_efficiency': 0.0
        }
    
    async def start(self):
        """启动批处理系统"""
        if self._running:
            return
        
        self._running = True
        
        # 启动性能监控
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
        
        # 启动处理任务
        num_workers = 2  # 可配置
        for i in range(num_workers):
            task = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self._processing_tasks.append(task)
        
        # 启动监控任务
        if self.config.enable_dynamic_adjustment:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Integrated batch system started with {num_workers} workers")
    
    async def stop(self):
        """停止批处理系统"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止处理任务
        for task in self._processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        self._processing_tasks.clear()
        
        # 停止监控任务
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 停止性能监控
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        logger.info("Integrated batch system stopped")
    
    async def process_request(
        self,
        content: str,
        model: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        timeout: float = 30.0,
        request_id: Optional[str] = None
    ) -> Any:
        """
        处理单个请求
        
        Args:
            content: 请求内容
            model: 模型名称
            parameters: 模型参数
            priority: 优先级 (1=high, 0=normal, -1=low)
            timeout: 超时时间
            request_id: 请求ID
        
        Returns:
            处理结果
        """
        if not self._running:
            await self.start()
        
        # 创建请求
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000)}_{id(content)}"
        
        request = ProcessingRequest(
            id=request_id,
            content=content,
            model=model,
            parameters=parameters or {},
            priority=priority,
            timeout=timeout
        )
        
        # 根据优先级入队
        queue = self._get_queue_by_priority(priority)
        await queue.put(request)
        
        self.stats['total_requests'] += 1
        
        try:
            # 等待结果
            result = await asyncio.wait_for(request.future, timeout=timeout)
            self.stats['processed_requests'] += 1
            return result
        except asyncio.TimeoutError:
            self.stats['failed_requests'] += 1
            raise asyncio.TimeoutError(f"Request {request_id} timed out after {timeout}s")
        except Exception as e:
            self.stats['failed_requests'] += 1
            raise e
    
    def _get_queue_by_priority(self, priority: int) -> asyncio.Queue:
        """根据优先级获取队列"""
        if priority > 0:
            return self.high_priority_queue
        elif priority < 0:
            return self.low_priority_queue
        else:
            return self.normal_priority_queue
    
    async def _processing_worker(self, worker_id: str):
        """处理工作线程"""
        logger.info(f"Processing worker {worker_id} started")
        
        while self._running:
            try:
                # 收集批次
                batch = await self._collect_batch(worker_id)
                
                if batch:
                    await self._process_batch(batch, worker_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    async def _collect_batch(self, worker_id: str) -> List[ProcessingRequest]:
        """收集批次"""
        batch = []
        batch_start = time.time()
        
        # 确定目标批次大小
        target_size = self._get_target_batch_size()
        
        while len(batch) < target_size and self._running:
            # 计算剩余等待时间
            elapsed = time.time() - batch_start
            remaining_wait = self.config.max_wait_time - elapsed
            
            if remaining_wait <= 0:
                break
            
            # 按优先级获取请求
            request = await self._get_next_request(remaining_wait)
            
            if request:
                batch.append(request)
            else:
                break
        
        return batch
    
    def _get_target_batch_size(self) -> int:
        """获取目标批次大小"""
        # 检查高优先级队列
        if not self.high_priority_queue.empty():
            return min(self.config.high_priority_batch_size, self.current_batch_size)
        
        return self.current_batch_size
    
    async def _get_next_request(self, timeout: float) -> Optional[ProcessingRequest]:
        """按优先级获取下一个请求"""
        queues = [
            self.high_priority_queue,
            self.normal_priority_queue,
            self.low_priority_queue
        ]
        
        for queue in queues:
            if not queue.empty():
                try:
                    return await asyncio.wait_for(queue.get(), timeout=min(timeout, 0.01))
                except asyncio.TimeoutError:
                    continue
        
        # 如果所有队列都空，等待新请求
        try:
            # 使用 asyncio.wait 来同时等待多个队列
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(queue.get())
                    for queue in queues
                ],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if done:
                # 取消其他等待的任务
                for task in pending:
                    task.cancel()
                
                # 返回第一个完成的结果
                return await done.pop()
            
        except asyncio.TimeoutError:
            pass
        
        return None
    
    async def _process_batch(self, batch: List[ProcessingRequest], worker_id: str):
        """处理批次"""
        if not batch:
            return
        
        batch_start = time.time()
        
        try:
            # 请求合并 (如果启用)
            if self.request_merger and len(batch) > 1:
                batch, merge_mapping = await self._merge_requests(batch)
            else:
                merge_mapping = {}
            
            # 准备处理请求
            processing_requests = [
                {
                    'id': req.id,
                    'content': req.content,
                    'model': req.model,
                    'parameters': req.parameters
                }
                for req in batch
            ]
            
            # 执行批处理
            processing_start = time.time()
            results = await self.processing_func(processing_requests)
            processing_time = time.time() - processing_start
            
            # 分发结果
            await self._distribute_results(batch, results, merge_mapping)
            
            # 记录指标
            if self.performance_monitor:
                queue_time = processing_start - batch_start
                metrics = BatchMetrics(
                    batch_size=len(batch),
                    processing_time=processing_time,
                    queue_wait_time=queue_time,
                    throughput=len(batch) / processing_time if processing_time > 0 else 0,
                    latency_p95=processing_time,
                    error_rate=0.0
                )
                self.performance_monitor.record_batch_metrics(metrics)
            
            # 更新统计
            self._update_stats(batch, processing_time, batch_start)
            
            logger.debug(
                f"Worker {worker_id} processed batch of {len(batch)} requests "
                f"in {processing_time:.3f}s"
            )
            
        except Exception as e:
            logger.error(f"Batch processing error in worker {worker_id}: {e}")
            
            # 处理错误
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
            
            # 记录错误指标
            if self.performance_monitor:
                processing_time = time.time() - processing_start
                queue_time = processing_start - batch_start
                metrics = BatchMetrics(
                    batch_size=len(batch),
                    processing_time=processing_time,
                    queue_wait_time=queue_time,
                    throughput=0.0,
                    latency_p95=processing_time,
                    error_rate=1.0
                )
                self.performance_monitor.record_batch_metrics(metrics)
    
    async def _merge_requests(
        self,
        batch: List[ProcessingRequest]
    ) -> Tuple[List[ProcessingRequest], Dict[str, List[str]]]:
        """合并请求"""
        # 转换为可合并请求
        mergeable_requests = [req.to_mergeable_request() for req in batch]
        
        # 执行合并
        unique_requests, merge_mapping = self.request_merger.merge_requests_advanced(
            mergeable_requests
        )
        
        # 转换回处理请求
        unique_batch = []
        for mergeable_req in unique_requests:
            # 找到对应的原始请求
            for original_req in batch:
                if original_req.id == mergeable_req.id:
                    unique_batch.append(original_req)
                    break
        
        # 更新统计
        merged_count = len(batch) - len(unique_batch)
        self.stats['merged_requests'] += merged_count
        
        if merged_count > 0:
            logger.debug(f"Merged {merged_count} requests, processing {len(unique_batch)} unique requests")
        
        return unique_batch, merge_mapping
    
    async def _distribute_results(
        self,
        batch: List[ProcessingRequest],
        results: List[Any],
        merge_mapping: Dict[str, List[str]]
    ):
        """分发结果"""
        # 创建结果映射
        result_map = {}
        for req, result in zip(batch, results):
            result_map[req.id] = result
        
        # 分发给合并的请求
        if self.request_merger and merge_mapping:
            for group_id, merged_ids in merge_mapping.items():
                # 找到代表请求的结果
                representative_result = None
                for req in batch:
                    if req.id not in merged_ids:  # 代表请求不在合并列表中
                        representative_result = result_map.get(req.id)
                        break
                
                if representative_result:
                    self.request_merger.distribute_response(
                        group_id, representative_result, merge_mapping
                    )
        
        # 分发给未合并的请求
        for req in batch:
            if not req.future.done():
                result = result_map.get(req.id)
                if result is not None:
                    req.future.set_result(result)
                else:
                    req.future.set_exception(
                        RuntimeError(f"No result for request {req.id}")
                    )
    
    def _update_stats(
        self,
        batch: List[ProcessingRequest],
        processing_time: float,
        batch_start: float
    ):
        """更新统计信息"""
        total_queue_time = sum(
            batch_start - req.created_at for req in batch
        ) / len(batch)
        
        # 更新平均值
        total_processed = self.stats['processed_requests']
        if total_processed > 0:
            self.stats['avg_processing_time'] = (
                self.stats['avg_processing_time'] * (total_processed - len(batch)) +
                processing_time * len(batch)
            ) / total_processed
            
            self.stats['avg_queue_time'] = (
                self.stats['avg_queue_time'] * (total_processed - len(batch)) +
                total_queue_time * len(batch)
            ) / total_processed
        
        # 计算吞吐量
        if processing_time > 0:
            self.stats['throughput'] = len(batch) / processing_time
        
        # 计算批次效率
        if len(batch) > 0:
            self.stats['batch_efficiency'] = len(batch) / self.current_batch_size
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.monitor_interval)
                
                if self.performance_monitor:
                    # 计算性能评分
                    performance_score = self.performance_monitor.calculate_performance_score()
                    
                    # 获取系统负载
                    system_metrics = self.performance_monitor.get_recent_system_metrics(1)
                    system_load = 0.5
                    
                    if system_metrics:
                        latest = system_metrics[-1]
                        system_load = (latest.cpu_usage + latest.memory_usage) / 200.0
                    
                    # 调整批次大小
                    self._adjust_batch_size(performance_score, system_load)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _adjust_batch_size(self, performance_score: float, system_load: float):
        """调整批次大小"""
        old_size = self.current_batch_size
        
        # 简单的调整逻辑
        if performance_score > 0.8 and system_load < 0.7:
            # 性能良好，增加批次大小
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        elif performance_score < 0.4 or system_load > 0.9:
            # 性能不佳，减少批次大小
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        
        if self.current_batch_size != old_size:
            logger.info(
                f"Batch size adjusted: {old_size} -> {self.current_batch_size} "
                f"(performance: {performance_score:.3f}, load: {system_load:.3f})"
            )
            
            self.batch_size_history.append({
                'timestamp': time.time(),
                'old_size': old_size,
                'new_size': self.current_batch_size,
                'performance_score': performance_score,
                'system_load': system_load
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加队列状态
        stats.update({
            'high_priority_queue_size': self.high_priority_queue.qsize(),
            'normal_priority_queue_size': self.normal_priority_queue.qsize(),
            'low_priority_queue_size': self.low_priority_queue.qsize(),
            'current_batch_size': self.current_batch_size,
            'batch_adjustments': len(self.batch_size_history)
        })
        
        # 添加合并统计
        if self.request_merger:
            merge_stats = self.request_merger.get_stats()
            stats.update({f"merge_{k}": v for k, v in merge_stats.items()})
        
        # 添加性能监控统计
        if self.performance_monitor:
            recent_batches = self.performance_monitor.get_recent_batch_metrics(10)
            if recent_batches:
                import statistics
                stats.update({
                    'recent_avg_throughput': statistics.mean([b.throughput for b in recent_batches]),
                    'recent_avg_latency': statistics.mean([b.latency_p95 for b in recent_batches]),
                    'recent_avg_error_rate': statistics.mean([b.error_rate for b in recent_batches])
                })
        
        return stats


# 使用示例
async def mock_integrated_processing(requests: List[Dict[str, Any]]) -> List[str]:
    """模拟集成处理"""
    batch_size = len(requests)
    
    # 模拟不同模型的处理时间
    model_times = {
        'gpt-3.5-turbo': 0.3,
        'gpt-4': 0.8,
        'claude-3': 0.5
    }
    
    # 计算处理时间
    avg_time = sum(model_times.get(req.get('model', 'gpt-3.5-turbo'), 0.3) for req in requests) / batch_size
    batch_efficiency = min(1.5, batch_size / 4.0)  # 批次效率
    processing_time = avg_time / batch_efficiency
    
    await asyncio.sleep(processing_time)
    
    return [
        f"Processed '{req.get('content', 'N/A')}' with {req.get('model', 'unknown')}"
        for req in requests
    ]


async def example_usage():
    """使用示例"""
    
    # 创建配置
    config = BatchingConfig(
        initial_batch_size=4,
        max_batch_size=16,
        max_wait_time=0.1,
        enable_request_merging=True,
        enable_dynamic_adjustment=True,
        similarity_threshold=0.8
    )
    
    # 创建集成批处理系统
    batch_system = IntegratedBatchSystem(
        processing_func=mock_integrated_processing,
        config=config
    )
    
    await batch_system.start()
    
    # 模拟不同类型的请求
    async def generate_requests():
        requests = []
        
        # 高优先级请求
        for i in range(5):
            task = asyncio.create_task(
                batch_system.process_request(
                    content=f"Urgent query {i}",
                    model="gpt-4",
                    priority=1,
                    timeout=10.0
                )
            )
            requests.append(task)
            await asyncio.sleep(0.01)
        
        # 普通请求 (一些相似的)
        for i in range(15):
            content = f"Translate 'Hello world {i % 3}' to Chinese"  # 制造相似请求
            task = asyncio.create_task(
                batch_system.process_request(
                    content=content,
                    model="gpt-3.5-turbo",
                    priority=0
                )
            )
            requests.append(task)
            await asyncio.sleep(0.02)
        
        # 低优先级请求
        for i in range(10):
            task = asyncio.create_task(
                batch_system.process_request(
                    content=f"Background task {i}",
                    model="claude-3",
                    priority=-1,
                    timeout=60.0
                )
            )
            requests.append(task)
            await asyncio.sleep(0.05)
        
        return requests
    
    # 启动请求生成
    print("Generating requests...")
    request_tasks = await generate_requests()
    
    # 监控统计
    async def monitor_stats():
        for i in range(8):
            await asyncio.sleep(2)
            stats = batch_system.get_stats()
            print(f"\n=== 统计信息 (第{i+1}次) ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    if 'rate' in key or 'efficiency' in key:
                        print(f"{key}: {value:.2%}")
                    else:
                        print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
    
    monitor_task = asyncio.create_task(monitor_stats())
    
    # 等待所有请求完成
    print("Waiting for requests to complete...")
    results = await asyncio.gather(*request_tasks, return_exceptions=True)
    
    # 统计结果
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    print(f"\n=== 请求完成 ===")
    print(f"总请求: {len(results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    
    # 等待监控完成
    await asyncio.sleep(2)
    
    # 停止系统
    await batch_system.stop()
    
    # 最终统计
    final_stats = batch_system.get_stats()
    print(f"\n=== 最终统计 ===")
    for key, value in final_stats.items():
        if isinstance(value, float):
            if 'rate' in key or 'efficiency' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_usage())
