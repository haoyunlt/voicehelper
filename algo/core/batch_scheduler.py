"""
VoiceHelper v1.20.0 - 企业级批处理调度器
基于开源最佳实践的高性能异步批处理系统
支持优先级队列、并发控制、负载均衡和监控
"""

import asyncio
import time
import logging
import uuid
import heapq
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class RequestPriority(Enum):
    """请求优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class RequestType(Enum):
    """请求类型"""
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    VOICE_SYNTHESIS = "voice_synthesis"
    EMOTION_ANALYSIS = "emotion_analysis"
    MULTIMODAL = "multimodal"

@dataclass
class ProcessRequest:
    """处理请求"""
    id: str
    type: RequestType
    priority: RequestPriority
    data: Any
    user_id: str
    timestamp: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """优先级队列排序：优先级高的先处理，时间早的先处理"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp

@dataclass
class BatchResult:
    """批处理结果"""
    batch_id: str
    requests: List[ProcessRequest]
    processing_time: float
    throughput: float
    success_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)

@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_batch_size: int = 32
    max_wait_time: float = 0.1  # 最大等待时间(秒)
    max_concurrent_batches: int = 4
    worker_pool_size: int = 8
    enable_metrics: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5  # 连续失败次数阈值
    circuit_breaker_timeout: float = 30.0  # 熔断器恢复时间

class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, threshold: int = 5, timeout: float = 30.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"

class BatchScheduler:
    """企业级批处理调度器"""
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.is_running = False
        
        # 优先级队列 - 使用堆实现
        self._request_queue: List[ProcessRequest] = []
        self._queue_lock = asyncio.Lock()
        
        # 结果存储
        self._results: Dict[str, Any] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # 统计信息
        self.total_requests = 0
        self.total_batches = 0
        self.successful_batches = 0
        self.failed_batches = 0
        self.average_batch_size = 0.0
        self.average_processing_time = 0.0
        
        # 并发控制
        self._batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        self._worker_semaphore = asyncio.Semaphore(self.config.worker_pool_size)
        
        # 熔断器
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        ) if self.config.enable_circuit_breaker else None
        
        # 任务管理
        self._scheduler_task: Optional[asyncio.Task] = None
        self._worker_tasks: List[asyncio.Task] = []
        
        # 指标收集
        self._metrics: Optional[Dict[str, List[Any]]] = defaultdict(list) if self.config.enable_metrics else None
        
    async def start(self):
        """启动调度器"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        self.is_running = True
        
        # 启动调度器主循环
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # 启动工作线程池
        for i in range(self.config.worker_pool_size):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        logger.info(f"Batch scheduler started with {self.config.worker_pool_size} workers")
    
    async def stop(self):
        """停止调度器"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # 停止调度器任务
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 停止工作任务
        for task in self._worker_tasks:
            task.cancel()
        
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        logger.info("Batch scheduler stopped")
    
    async def submit_request(self, request: ProcessRequest) -> asyncio.Future:
        """提交处理请求"""
        if not self.is_running:
            raise RuntimeError("Scheduler is not running")
        
        # 创建Future用于返回结果
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request.id] = future
        
        # 添加到优先级队列
        async with self._queue_lock:
            heapq.heappush(self._request_queue, request)
            self.total_requests += 1
        
        logger.debug(f"Request {request.id} submitted (priority: {request.priority.name})")
        return future
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running:
            try:
                batch = await self._collect_batch()
                if batch:
                    # 异步处理批次
                    asyncio.create_task(self._process_batch_async(batch))
                else:
                    # 没有请求时短暂休眠
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[ProcessRequest]:
        """收集一批请求"""
        batch: List[ProcessRequest] = []
        start_time = time.time()
        
        async with self._queue_lock:
            # 收集请求直到达到批次大小或超时
            while (len(batch) < self.config.max_batch_size and 
                   len(self._request_queue) > 0 and
                   (time.time() - start_time) < self.config.max_wait_time):
                
                request = heapq.heappop(self._request_queue)
                
                # 检查请求是否超时
                if request.timeout and (time.time() - request.timestamp) > request.timeout:
                    self._complete_request(request.id, None, "Request timeout")
                    continue
                
                batch.append(request)
        
        return batch
    
    async def _process_batch_async(self, batch: List[ProcessRequest]):
        """异步处理批次"""
        async with self._batch_semaphore:
            try:
                result = await self._process_batch(batch)
                self._record_batch_result(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                self.failed_batches += 1
                # 标记批次中的所有请求为失败
                for request in batch:
                    self._complete_request(request.id, None, str(e))
    
    async def _process_batch(self, requests: List[ProcessRequest]) -> BatchResult:
        """处理一批请求"""
        if not requests:
            return BatchResult("empty", [], 0.0, 0.0)
        
        # 检查熔断器
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise RuntimeError("Circuit breaker is open")
        
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        success_count = 0
        error_count = 0
        errors = []
        
        try:
            # 按类型分组处理
            grouped_requests = defaultdict(list)
            for request in requests:
                grouped_requests[request.type].append(request)
            
            # 并发处理不同类型的请求
            tasks = []
            for request_type, type_requests in grouped_requests.items():
                task = asyncio.create_task(
                    self._process_requests_by_type(request_type, type_requests)
                )
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_count += len(list(grouped_requests.values())[i])
                    errors.append(str(result))
                else:
                    success_count += len(list(grouped_requests.values())[i])
                    # 完成请求
                    for request in list(grouped_requests.values())[i]:
                        self._complete_request(request.id, result, None)
            
            processing_time = time.time() - start_time
            throughput = len(requests) / processing_time if processing_time > 0 else 0
            
            # 记录熔断器成功
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            return BatchResult(
                batch_id=batch_id,
                requests=requests,
                processing_time=processing_time,
                throughput=throughput,
                success_count=success_count,
                error_count=error_count,
                errors=errors
            )
            
        except Exception as e:
            # 记录熔断器失败
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            # 标记所有请求为失败
            for request in requests:
                self._complete_request(request.id, None, str(e))
            
            raise
    
    async def _process_requests_by_type(self, request_type: RequestType, requests: List[ProcessRequest]) -> Any:
        """按类型处理请求"""
        async with self._worker_semaphore:
            # 模拟不同类型的处理时间
            processing_times = {
                RequestType.TEXT_GENERATION: 0.2,
                RequestType.EMBEDDING: 0.05,
                RequestType.VOICE_SYNTHESIS: 0.3,
                RequestType.EMOTION_ANALYSIS: 0.1,
                RequestType.MULTIMODAL: 0.4
            }
            
            await asyncio.sleep(processing_times.get(request_type, 0.1))
            
            # 模拟处理结果
            return {
                "type": request_type.value,
                "processed_count": len(requests),
                "timestamp": time.time()
            }
    
    async def _worker_loop(self, worker_id: str):
        """工作线程循环"""
        logger.debug(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # 工作线程可以执行额外的后台任务
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def _complete_request(self, request_id: str, result: Any, error: Optional[str]):
        """完成请求"""
        future = self._pending_requests.pop(request_id, None)
        if future and not future.done():
            if error:
                future.set_exception(Exception(error))
            else:
                future.set_result(result)
    
    def _record_batch_result(self, result: BatchResult):
        """记录批次结果"""
        self.total_batches += 1
        
        if result.error_count == 0:
            self.successful_batches += 1
        else:
            self.failed_batches += 1
        
        # 更新平均值
        self.average_batch_size = (
            (self.average_batch_size * (self.total_batches - 1) + len(result.requests)) / 
            self.total_batches
        )
        
        self.average_processing_time = (
            (self.average_processing_time * (self.total_batches - 1) + result.processing_time) / 
            self.total_batches
        )
        
        # 记录指标
        if self._metrics is not None:
            self._metrics['batch_sizes'].append(len(result.requests))
            self._metrics['processing_times'].append(result.processing_time)
            self._metrics['throughputs'].append(result.throughput)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats: Dict[str, Any] = {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "success_rate": self.successful_batches / max(self.total_batches, 1),
            "average_batch_size": self.average_batch_size,
            "average_processing_time": self.average_processing_time,
            "is_running": self.is_running,
            "queue_size": len(self._request_queue),
            "pending_requests": len(self._pending_requests)
        }
        
        if self.circuit_breaker:
            stats["circuit_breaker_state"] = self.circuit_breaker.state
            stats["circuit_breaker_failures"] = self.circuit_breaker.failure_count
        
        return stats
    
    def get_metrics(self) -> Dict[str, List]:
        """获取详细指标"""
        return dict(self._metrics) if self._metrics else {}

# 保持向后兼容的简化版本
class SimpleBatchScheduler:
    """简化批处理调度器 - 向后兼容版本"""
    
    def __init__(self):
        self.is_running = False
        self.total_requests = 0
        self.total_batches = 0
        self.average_batch_size = 0.0
        self._batch_scheduler = BatchScheduler(SchedulerConfig(
            max_batch_size=8,
            max_wait_time=0.05,
            max_concurrent_batches=2,
            worker_pool_size=4,
            enable_metrics=False,
            enable_circuit_breaker=False
        ))
    
    async def start(self):
        """启动调度器"""
        await self._batch_scheduler.start()
        self.is_running = True
        logger.info("Simple batch scheduler started")
    
    async def stop(self):
        """停止调度器"""
        await self._batch_scheduler.stop()
        self.is_running = False
        logger.info("Simple batch scheduler stopped")
    
    async def submit_request(self, request: ProcessRequest) -> str:
        """提交处理请求 - 简化版本"""
        await self._batch_scheduler.submit_request(request)
        self.total_requests += 1
        return request.id
    
    async def process_batch(self, requests: List[ProcessRequest]) -> BatchResult:
        """处理一批请求 - 兼容旧接口"""
        return await self._batch_scheduler._process_batch(requests)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._batch_scheduler.get_statistics()
        return {
            "total_requests": stats["total_requests"],
            "total_batches": stats["total_batches"],
            "average_batch_size": stats["average_batch_size"],
            "is_running": self.is_running
        }
