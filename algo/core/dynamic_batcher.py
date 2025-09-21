"""
动态批次大小调整器

根据系统负载和响应时间自动调整批次大小
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque
import logging
import threading
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class BatchMetrics:
    """批次指标"""
    batch_size: int
    processing_time: float
    queue_wait_time: float
    throughput: float  # requests/second
    latency_p95: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    network_latency: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.batch_history: deque = deque(maxlen=history_size)
        self.system_history: deque = deque(maxlen=history_size)
        
        # 监控线程
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 1.0  # 1秒
    
    def start_monitoring(self):
        """启动系统监控"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_system)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """停止系统监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_system(self):
        """系统监控循环"""
        while self._monitoring:
            try:
                # 获取系统指标
                if HAS_PSUTIL:
                    cpu_usage = psutil.cpu_percent(interval=0.1)
                    memory_usage = psutil.virtual_memory().percent
                else:
                    # 模拟系统指标
                    cpu_usage = 50.0
                    memory_usage = 60.0
                
                # GPU使用率 (需要nvidia-ml-py库)
                gpu_usage = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = gpu_info.gpu
                except (ImportError, Exception):
                    pass
                
                metrics = SystemMetrics(
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    gpu_usage=gpu_usage
                )
                
                self.system_history.append(metrics)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
            
            time.sleep(self._monitor_interval)
    
    def record_batch_metrics(self, metrics: BatchMetrics):
        """记录批次指标"""
        self.batch_history.append(metrics)
    
    def get_recent_batch_metrics(self, count: int = 10) -> List[BatchMetrics]:
        """获取最近的批次指标"""
        return list(self.batch_history)[-count:]
    
    def get_recent_system_metrics(self, count: int = 10) -> List[SystemMetrics]:
        """获取最近的系统指标"""
        return list(self.system_history)[-count:]
    
    def calculate_performance_score(self) -> float:
        """计算性能评分 (0-1)"""
        if not self.batch_history:
            return 0.5
        
        recent_batches = self.get_recent_batch_metrics(5)
        
        # 计算各项指标的权重评分
        throughput_scores = [min(b.throughput / 10.0, 1.0) for b in recent_batches]
        latency_scores = [max(0, 1.0 - b.latency_p95 / 5.0) for b in recent_batches]
        error_scores = [max(0, 1.0 - b.error_rate) for b in recent_batches]
        
        # 加权平均
        weights = [0.4, 0.4, 0.2]  # 吞吐量、延迟、错误率
        scores = [
            statistics.mean(throughput_scores),
            statistics.mean(latency_scores),
            statistics.mean(error_scores)
        ]
        
        return sum(w * s for w, s in zip(weights, scores))


class DynamicBatchSizeController:
    """动态批次大小控制器"""
    
    def __init__(
        self,
        initial_batch_size: int = 4,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        adjustment_factor: float = 0.2,
        stability_threshold: int = 3
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adjustment_factor = adjustment_factor
        self.stability_threshold = stability_threshold
        
        # 控制状态
        self.consecutive_improvements = 0
        self.consecutive_degradations = 0
        self.last_performance_score = 0.5
        self.adjustment_history: deque = deque(maxlen=20)
        
        # 自适应参数
        self.learning_rate = 0.1
        self.exploration_probability = 0.1
        
    def adjust_batch_size(self, performance_score: float, system_load: float) -> int:
        """
        调整批次大小
        
        Args:
            performance_score: 性能评分 (0-1)
            system_load: 系统负载 (0-1)
        
        Returns:
            新的批次大小
        """
        # 记录调整历史
        self.adjustment_history.append({
            'batch_size': self.current_batch_size,
            'performance_score': performance_score,
            'system_load': system_load,
            'timestamp': time.time()
        })
        
        # 计算性能变化
        performance_delta = performance_score - self.last_performance_score
        
        # 决策逻辑
        new_batch_size = self._make_adjustment_decision(
            performance_delta, system_load, performance_score
        )
        
        # 更新状态
        self._update_control_state(performance_delta)
        self.last_performance_score = performance_score
        
        # 应用约束
        new_batch_size = max(self.min_batch_size, min(new_batch_size, self.max_batch_size))
        
        if new_batch_size != self.current_batch_size:
            logger.info(
                f"Batch size adjusted: {self.current_batch_size} -> {new_batch_size} "
                f"(performance: {performance_score:.3f}, load: {system_load:.3f})"
            )
            self.current_batch_size = new_batch_size
        
        return self.current_batch_size
    
    def _make_adjustment_decision(
        self,
        performance_delta: float,
        system_load: float,
        performance_score: float
    ) -> int:
        """做出调整决策"""
        
        # 1. 系统过载保护
        if system_load > 0.9:
            return max(self.min_batch_size, int(self.current_batch_size * 0.7))
        
        # 2. 性能显著下降
        if performance_delta < -0.1:
            if self.current_batch_size > self.min_batch_size:
                return int(self.current_batch_size * (1 - self.adjustment_factor))
            else:
                return self.current_batch_size
        
        # 3. 性能显著提升
        if performance_delta > 0.1:
            if self.current_batch_size < self.max_batch_size and system_load < 0.7:
                return int(self.current_batch_size * (1 + self.adjustment_factor))
            else:
                return self.current_batch_size
        
        # 4. 探索性调整
        if self._should_explore():
            direction = 1 if performance_score > 0.6 and system_load < 0.6 else -1
            adjustment = max(1, int(self.current_batch_size * 0.1)) * direction
            return self.current_batch_size + adjustment
        
        # 5. 基于历史的智能调整
        return self._intelligent_adjustment(performance_score, system_load)
    
    def _should_explore(self) -> bool:
        """是否应该进行探索性调整"""
        import random
        
        # 在稳定期间偶尔探索
        if (self.consecutive_improvements == 0 and 
            self.consecutive_degradations == 0 and
            random.random() < self.exploration_probability):
            return True
        
        return False
    
    def _intelligent_adjustment(self, performance_score: float, system_load: float) -> int:
        """基于历史数据的智能调整"""
        if len(self.adjustment_history) < 5:
            return self.current_batch_size
        
        # 分析历史数据找到最佳批次大小
        recent_history = list(self.adjustment_history)[-10:]
        
        # 按性能评分排序
        sorted_history = sorted(recent_history, key=lambda x: x['performance_score'], reverse=True)
        
        # 获取表现最好的批次大小
        best_batch_sizes = [h['batch_size'] for h in sorted_history[:3]]
        target_batch_size = int(statistics.mean(best_batch_sizes))
        
        # 渐进式调整
        if target_batch_size > self.current_batch_size:
            return min(target_batch_size, int(self.current_batch_size * 1.2))
        elif target_batch_size < self.current_batch_size:
            return max(target_batch_size, int(self.current_batch_size * 0.8))
        else:
            return self.current_batch_size
    
    def _update_control_state(self, performance_delta: float):
        """更新控制状态"""
        if performance_delta > 0.05:
            self.consecutive_improvements += 1
            self.consecutive_degradations = 0
        elif performance_delta < -0.05:
            self.consecutive_degradations += 1
            self.consecutive_improvements = 0
        else:
            # 性能稳定
            if self.consecutive_improvements > 0:
                self.consecutive_improvements = max(0, self.consecutive_improvements - 1)
            if self.consecutive_degradations > 0:
                self.consecutive_degradations = max(0, self.consecutive_degradations - 1)
    
    def get_adjustment_stats(self) -> Dict[str, Any]:
        """获取调整统计信息"""
        if not self.adjustment_history:
            return {}
        
        history = list(self.adjustment_history)
        batch_sizes = [h['batch_size'] for h in history]
        performance_scores = [h['performance_score'] for h in history]
        
        return {
            'current_batch_size': self.current_batch_size,
            'avg_batch_size': statistics.mean(batch_sizes),
            'min_batch_size_used': min(batch_sizes),
            'max_batch_size_used': max(batch_sizes),
            'avg_performance': statistics.mean(performance_scores),
            'consecutive_improvements': self.consecutive_improvements,
            'consecutive_degradations': self.consecutive_degradations,
            'total_adjustments': len(history)
        }


class AdaptiveBatchProcessor:
    """自适应批处理器"""
    
    def __init__(
        self,
        processing_func: Callable[[List[Dict[str, Any]]], Awaitable[List[Any]]],
        initial_batch_size: int = 4,
        max_wait_time: float = 0.1,
        monitor_interval: float = 5.0
    ):
        self.processing_func = processing_func
        self.max_wait_time = max_wait_time
        self.monitor_interval = monitor_interval
        
        # 组件
        self.performance_monitor = PerformanceMonitor()
        self.batch_controller = DynamicBatchSizeController(initial_batch_size)
        
        # 处理状态
        self._queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 当前批次
        self._current_batch: List[Dict[str, Any]] = []
        self._current_futures: List[asyncio.Future] = []
        self._batch_start_time = time.time()
    
    async def start(self):
        """启动自适应批处理器"""
        if self._running:
            return
        
        self._running = True
        self.performance_monitor.start_monitoring()
        
        # 启动处理任务
        self._processing_task = asyncio.create_task(self._processing_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Adaptive batch processor started")
    
    async def stop(self):
        """停止自适应批处理器"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止任务
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 处理剩余批次
        await self._process_current_batch()
        
        self.performance_monitor.stop_monitoring()
        logger.info("Adaptive batch processor stopped")
    
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """处理单个请求"""
        if not self._running:
            await self.start()
        
        future = asyncio.Future()
        await self._queue.put({'request': request, 'future': future})
        return await future
    
    async def _processing_loop(self):
        """处理循环"""
        while self._running:
            try:
                # 获取当前批次大小
                current_batch_size = self.batch_controller.current_batch_size
                
                # 收集请求到批次
                await self._collect_batch(current_batch_size)
                
                # 处理批次
                if self._current_batch:
                    await self._process_current_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self, target_size: int):
        """收集批次"""
        batch_start = time.time()
        
        while len(self._current_batch) < target_size and self._running:
            try:
                # 计算剩余等待时间
                elapsed = time.time() - batch_start
                remaining_wait = self.max_wait_time - elapsed
                
                if remaining_wait <= 0:
                    break
                
                # 等待新请求
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=min(remaining_wait, 0.01)
                )
                
                self._current_batch.append(item['request'])
                self._current_futures.append(item['future'])
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                # 超时，处理当前批次
                break
    
    async def _process_current_batch(self):
        """处理当前批次"""
        if not self._current_batch:
            return
        
        batch_size = len(self._current_batch)
        processing_start = time.time()
        queue_wait_time = processing_start - self._batch_start_time
        
        try:
            # 执行批处理
            results = await self.processing_func(self._current_batch)
            
            # 分发结果
            for future, result in zip(self._current_futures, results):
                if not future.done():
                    future.set_result(result)
            
            processing_time = time.time() - processing_start
            
            # 记录指标
            metrics = BatchMetrics(
                batch_size=batch_size,
                processing_time=processing_time,
                queue_wait_time=queue_wait_time,
                throughput=batch_size / processing_time if processing_time > 0 else 0,
                latency_p95=processing_time,  # 简化计算
                error_rate=0.0
            )
            
            self.performance_monitor.record_batch_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            
            # 处理错误
            for future in self._current_futures:
                if not future.done():
                    future.set_exception(e)
            
            # 记录错误指标
            processing_time = time.time() - processing_start
            metrics = BatchMetrics(
                batch_size=batch_size,
                processing_time=processing_time,
                queue_wait_time=queue_wait_time,
                throughput=0.0,
                latency_p95=processing_time,
                error_rate=1.0
            )
            
            self.performance_monitor.record_batch_metrics(metrics)
        
        finally:
            # 清理当前批次
            self._current_batch.clear()
            self._current_futures.clear()
            self._batch_start_time = time.time()
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await asyncio.sleep(self.monitor_interval)
                
                # 计算性能评分
                performance_score = self.performance_monitor.calculate_performance_score()
                
                # 获取系统负载
                system_metrics = self.performance_monitor.get_recent_system_metrics(1)
                system_load = 0.5  # 默认值
                
                if system_metrics:
                    latest = system_metrics[-1]
                    # 综合CPU和内存使用率
                    system_load = (latest.cpu_usage + latest.memory_usage) / 200.0
                    if latest.gpu_usage is not None:
                        system_load = (system_load * 2 + latest.gpu_usage / 100.0) / 3
                
                # 调整批次大小
                self.batch_controller.adjust_batch_size(performance_score, system_load)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        batch_stats = self.batch_controller.get_adjustment_stats()
        
        recent_batches = self.performance_monitor.get_recent_batch_metrics(10)
        if recent_batches:
            avg_throughput = statistics.mean([b.throughput for b in recent_batches])
            avg_latency = statistics.mean([b.latency_p95 for b in recent_batches])
            avg_error_rate = statistics.mean([b.error_rate for b in recent_batches])
        else:
            avg_throughput = avg_latency = avg_error_rate = 0.0
        
        return {
            **batch_stats,
            'avg_throughput': avg_throughput,
            'avg_latency': avg_latency,
            'avg_error_rate': avg_error_rate,
            'performance_score': self.performance_monitor.calculate_performance_score(),
            'queue_size': self._queue.qsize(),
            'current_batch_size_actual': len(self._current_batch)
        }


# 使用示例
async def mock_llm_batch_processing(requests: List[Dict[str, Any]]) -> List[str]:
    """模拟LLM批处理"""
    # 模拟处理时间 (批次越大，单个请求的平均时间越短)
    batch_size = len(requests)
    base_time = 0.5
    efficiency_factor = min(1.0, batch_size / 8.0)  # 批次效率
    processing_time = base_time * (1.0 + 0.1 * batch_size) * (2.0 - efficiency_factor)
    
    await asyncio.sleep(processing_time)
    
    return [f"Response to '{req.get('prompt', 'N/A')}'" for req in requests]


async def example_usage():
    """使用示例"""
    
    # 创建自适应批处理器
    processor = AdaptiveBatchProcessor(
        processing_func=mock_llm_batch_processing,
        initial_batch_size=4,
        max_wait_time=0.05,
        monitor_interval=2.0
    )
    
    await processor.start()
    
    # 模拟请求负载
    async def generate_requests():
        for i in range(50):
            # 变化的请求频率
            if i < 20:
                await asyncio.sleep(0.01)  # 高频
            elif i < 35:
                await asyncio.sleep(0.1)   # 中频
            else:
                await asyncio.sleep(0.05)  # 低频
            
            asyncio.create_task(
                processor.process_request({"prompt": f"Request {i}", "id": i})
            )
    
    # 启动请求生成
    request_task = asyncio.create_task(generate_requests())
    
    # 监控统计
    async def monitor_stats():
        for _ in range(10):
            await asyncio.sleep(3)
            stats = processor.get_stats()
            print(f"\n=== 统计信息 ===")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
    
    monitor_task = asyncio.create_task(monitor_stats())
    
    # 等待完成
    await request_task
    await asyncio.sleep(5)  # 等待处理完成
    
    # 停止处理器
    await processor.stop()
    
    # 最终统计
    final_stats = processor.get_stats()
    print(f"\n=== 最终统计 ===")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_usage())
