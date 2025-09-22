"""
VoiceHelper v1.20.0 - 自适应批处理调度器
实现智能请求调度、负载预测和资源优化
"""

import asyncio
import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict, deque

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
    estimated_processing_time: float = 0.0
    max_wait_time: float = 30.0  # 最大等待时间（秒）
    
    def __lt__(self, other):
        """用于优先级队列排序"""
        return (self.priority.value, -self.timestamp) > (other.priority.value, -other.timestamp)

@dataclass
class BatchGroup:
    """批处理组"""
    requests: List[ProcessRequest]
    batch_id: str = field(default_factory=lambda: f"batch_{int(time.time()*1000)}")
    created_at: float = field(default_factory=time.time)
    estimated_processing_time: float = 0.0
    
    def __post_init__(self):
        """计算预估处理时间"""
        if self.requests:
            self.estimated_processing_time = max(req.estimated_processing_time for req in self.requests)

@dataclass
class BatchConfig:
    """批处理配置"""
    max_batch_size: int = 32
    min_batch_size: int = 1
    max_wait_time: float = 0.1  # 100ms
    similarity_threshold: float = 0.8
    load_factor: float = 1.0

@dataclass
class ResourceStatus:
    """资源状态"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    queue_length: int
    active_batches: int
    average_processing_time: float

class LoadPredictor:
    """负载预测器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.load_history = deque(maxlen=history_size)
        self.time_series = deque(maxlen=history_size)
        
    async def predict_load(self) -> float:
        """预测系统负载"""
        current_time = time.time()
        
        # 收集当前负载数据
        current_load = await self._collect_current_load()
        
        # 更新历史数据
        self.load_history.append(current_load)
        self.time_series.append(current_time)
        
        # 预测未来负载
        if len(self.load_history) < 10:
            return current_load
        
        # 简单的移动平均预测
        recent_loads = list(self.load_history)[-10:]
        predicted_load = sum(recent_loads) / len(recent_loads)
        
        # 考虑趋势
        if len(recent_loads) >= 5:
            # 简单线性趋势计算
            x_values = list(range(len(recent_loads)))
            n = len(recent_loads)
            sum_x = sum(x_values)
            sum_y = sum(recent_loads)
            sum_xy = sum(x * y for x, y in zip(x_values, recent_loads))
            sum_x2 = sum(x * x for x in x_values)
            
            # 计算斜率
            trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            predicted_load += trend * 5  # 预测5个时间步后的负载
        
        return max(0.0, min(1.0, predicted_load))
    
    async def _collect_current_load(self) -> float:
        """收集当前负载数据"""
        # 模拟负载收集
        await asyncio.sleep(0.001)
        
        # 基于时间的模拟负载模式
        current_time = time.time()
        hour = (current_time % 86400) / 3600  # 当前小时
        
        # 模拟日常负载模式
        if 9 <= hour <= 18:  # 工作时间
            base_load = 0.7
        elif 19 <= hour <= 22:  # 晚上高峰
            base_load = 0.8
        else:  # 夜间低峰
            base_load = 0.3
        
        # 添加随机波动
        noise = random.gauss(0, 0.1)
        return max(0.0, min(1.0, base_load + noise))

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.gpu_history = deque(maxlen=100)
        
    def get_status(self) -> ResourceStatus:
        """获取资源状态"""
        # 模拟资源监控
        cpu_usage = random.uniform(0.2, 0.8)
        memory_usage = random.uniform(0.3, 0.7)
        gpu_usage = random.uniform(0.1, 0.9)
        
        # 更新历史
        self.cpu_history.append(cpu_usage)
        self.memory_history.append(memory_usage)
        self.gpu_history.append(gpu_usage)
        
        return ResourceStatus(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            queue_length=random.randint(0, 50),
            active_batches=random.randint(0, 10),
            average_processing_time=random.uniform(0.1, 2.0)
        )
    
    def is_resource_available(self, threshold: float = 0.8) -> bool:
        """检查资源是否可用"""
        status = self.get_status()
        return (status.cpu_usage < threshold and 
                status.memory_usage < threshold and 
                status.gpu_usage < threshold)

class BatchOptimizer:
    """批处理优化器"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.optimal_configs = {}
        
    def optimize_config(
        self, 
        load: float, 
        resources: ResourceStatus, 
        queue_length: int
    ) -> BatchConfig:
        """优化批处理配置"""
        
        # 基于负载调整批大小
        if load > 0.8:  # 高负载
            max_batch_size = 64
            max_wait_time = 0.05  # 50ms
        elif load > 0.5:  # 中等负载
            max_batch_size = 32
            max_wait_time = 0.1   # 100ms
        else:  # 低负载
            max_batch_size = 16
            max_wait_time = 0.2   # 200ms
        
        # 基于资源状态调整
        if resources.cpu_usage > 0.8:
            max_batch_size = min(max_batch_size, 16)
        
        if resources.memory_usage > 0.8:
            max_batch_size = min(max_batch_size, 8)
        
        # 基于队列长度调整
        if queue_length > 100:
            max_wait_time = min(max_wait_time, 0.05)
        elif queue_length < 10:
            max_wait_time = max(max_wait_time, 0.15)
        
        return BatchConfig(
            max_batch_size=max_batch_size,
            min_batch_size=max(1, max_batch_size // 8),
            max_wait_time=max_wait_time,
            similarity_threshold=0.8,
            load_factor=load
        )
    
    def record_performance(self, config: BatchConfig, throughput: float, latency: float):
        """记录性能数据"""
        key = (config.max_batch_size, config.max_wait_time)
        self.performance_history[key].append({
            "throughput": throughput,
            "latency": latency,
            "timestamp": time.time()
        })
        
        # 保持历史记录大小
        if len(self.performance_history[key]) > 100:
            self.performance_history[key] = self.performance_history[key][-50:]

class PriorityQueue:
    """优先级队列"""
    
    def __init__(self):
        self._queue = []
        self._index = 0
        self._lock = asyncio.Lock()
        
    async def put(self, request: ProcessRequest):
        """添加请求到队列"""
        async with self._lock:
            heapq.heappush(self._queue, (request.priority.value, self._index, request))
            self._index += 1
    
    async def get(self) -> Optional[ProcessRequest]:
        """从队列获取请求"""
        async with self._lock:
            if self._queue:
                _, _, request = heapq.heappop(self._queue)
                return request
            return None
    
    async def get_batch(self, max_size: int, timeout: float) -> List[ProcessRequest]:
        """批量获取请求"""
        requests = []
        deadline = time.time() + timeout
        
        while len(requests) < max_size and time.time() < deadline:
            try:
                # 尝试获取请求，设置短超时
                request = await asyncio.wait_for(self.get(), timeout=0.01)
                if request:
                    requests.append(request)
                else:
                    break
            except asyncio.TimeoutError:
                break
        
        return requests
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self._queue)
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self._queue) == 0

class AdaptiveBatchScheduler:
    """v1.20.0 自适应批处理调度器"""
    
    def __init__(self):
        self.load_predictor = LoadPredictor()
        self.resource_monitor = ResourceMonitor()
        self.batch_optimizer = BatchOptimizer()
        self.priority_queue = PriorityQueue()
        
        # 统计信息
        self.total_requests = 0
        self.total_batches = 0
        self.average_batch_size = 0.0
        self.average_wait_time = 0.0
        
        # 运行状态
        self.is_running = False
        self.scheduler_task = None
        
    async def start(self):
        """启动调度器"""
        if not self.is_running:
            self.is_running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("Adaptive batch scheduler started")
    
    async def stop(self):
        """停止调度器"""
        self.is_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Adaptive batch scheduler stopped")
    
    async def submit_request(self, request: ProcessRequest) -> str:
        """提交处理请求"""
        await self.priority_queue.put(request)
        self.total_requests += 1
        logger.debug(f"Request {request.id} submitted to queue")
        return request.id
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running:
            try:
                # 预测负载
                predicted_load = await self.load_predictor.predict_load()
                
                # 监控资源
                resource_status = self.resource_monitor.get_status()
                
                # 优化配置
                batch_config = self.batch_optimizer.optimize_config(
                    load=predicted_load,
                    resources=resource_status,
                    queue_length=self.priority_queue.size()
                )
                
                # 调度请求
                await self._schedule_requests(batch_config)
                
                # 短暂休眠
                await asyncio.sleep(0.01)  # 10ms
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _schedule_requests(self, config: BatchConfig):
        """调度请求"""
        if self.priority_queue.is_empty():
            return
        
        # 获取一批请求
        requests = await self.priority_queue.get_batch(
            max_size=config.max_batch_size,
            timeout=config.max_wait_time
        )
        
        if not requests:
            return
        
        # 按类型和相似性分组
        batch_groups = await self.group_requests(requests, config)
        
        # 处理每个批次组
        for batch_group in batch_groups:
            if batch_group.requests:
                asyncio.create_task(self._process_batch_group(batch_group))
                self.total_batches += 1
    
    async def group_requests(
        self, 
        requests: List[ProcessRequest], 
        config: BatchConfig
    ) -> List[BatchGroup]:
        """智能请求分组"""
        
        # 按请求类型分组
        type_groups = defaultdict(list)
        for request in requests:
            type_groups[request.type].append(request)
        
        batch_groups = []
        
        for request_type, type_requests in type_groups.items():
            # 按优先级排序
            type_requests.sort(key=lambda x: x.priority.value, reverse=True)
            
            # 创建批次组
            current_group = []
            for request in type_requests:
                # 检查是否可以加入当前组
                if self.can_group_together(current_group, request, config):
                    current_group.append(request)
                else:
                    # 创建新组
                    if current_group:
                        batch_groups.append(BatchGroup(current_group))
                    current_group = [request]
                
                # 检查组大小限制
                if len(current_group) >= config.max_batch_size:
                    batch_groups.append(BatchGroup(current_group))
                    current_group = []
            
            # 处理最后一组
            if current_group:
                batch_groups.append(BatchGroup(current_group))
        
        return batch_groups
    
    def can_group_together(
        self, 
        current_group: List[ProcessRequest], 
        new_request: ProcessRequest, 
        config: BatchConfig
    ) -> bool:
        """检查请求是否可以分组"""
        if not current_group:
            return True
        
        # 检查类型一致性
        if current_group[0].type != new_request.type:
            return False
        
        # 检查优先级差异
        priority_diff = abs(current_group[0].priority.value - new_request.priority.value)
        if priority_diff > 1:  # 优先级差异不能超过1级
            return False
        
        # 检查等待时间
        oldest_request = min(current_group, key=lambda x: x.timestamp)
        wait_time = time.time() - oldest_request.timestamp
        if wait_time > config.max_wait_time:
            return False
        
        # 检查相似性（基于用户ID和数据特征）
        similarity = self._calculate_similarity(current_group[0], new_request)
        if similarity < config.similarity_threshold:
            return False
        
        return True
    
    def _calculate_similarity(self, req1: ProcessRequest, req2: ProcessRequest) -> float:
        """计算请求相似性"""
        # 简化的相似性计算
        similarity = 0.0
        
        # 用户相似性
        if req1.user_id == req2.user_id:
            similarity += 0.3
        
        # 类型相似性
        if req1.type == req2.type:
            similarity += 0.4
        
        # 优先级相似性
        priority_diff = abs(req1.priority.value - req2.priority.value)
        similarity += 0.3 * (1 - priority_diff / 3)
        
        return min(1.0, max(0.0, similarity))
    
    async def _process_batch_group(self, batch_group: BatchGroup):
        """处理批次组"""
        start_time = time.time()
        
        try:
            logger.debug(f"Processing batch {batch_group.batch_id} with {len(batch_group.requests)} requests")
            
            # 模拟批处理
            await self._simulate_batch_processing(batch_group)
            
            # 记录性能
            processing_time = time.time() - start_time
            throughput = len(batch_group.requests) / processing_time
            
            # 更新统计
            self._update_statistics(batch_group, processing_time)
            
            logger.debug(f"Batch {batch_group.batch_id} completed in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def _simulate_batch_processing(self, batch_group: BatchGroup):
        """模拟批处理"""
        # 根据请求类型模拟不同的处理时间
        base_time = {
            RequestType.TEXT_GENERATION: 0.5,
            RequestType.EMBEDDING: 0.1,
            RequestType.VOICE_SYNTHESIS: 0.8,
            RequestType.EMOTION_ANALYSIS: 0.3,
            RequestType.MULTIMODAL: 1.0
        }
        
        if batch_group.requests:
            request_type = batch_group.requests[0].type
            processing_time = base_time.get(request_type, 0.5)
            
            # 批处理效率：批次越大，单个请求处理时间越短
            batch_efficiency = 1.0 - (len(batch_group.requests) - 1) * 0.02
            processing_time *= max(0.1, batch_efficiency)
            
            await asyncio.sleep(processing_time)
    
    def _update_statistics(self, batch_group: BatchGroup, processing_time: float):
        """更新统计信息"""
        batch_size = len(batch_group.requests)
        
        # 更新平均批大小
        self.average_batch_size = (
            (self.average_batch_size * (self.total_batches - 1) + batch_size) / 
            self.total_batches
        )
        
        # 更新平均等待时间
        wait_times = [time.time() - req.timestamp for req in batch_group.requests]
        avg_wait_time = sum(wait_times) / len(wait_times)
        
        self.average_wait_time = (
            (self.average_wait_time * (self.total_batches - 1) + avg_wait_time) / 
            self.total_batches
        )
    
    def get_statistics(self) -> Dict:
        """获取调度器统计信息"""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": self.average_batch_size,
            "average_wait_time_ms": self.average_wait_time * 1000,
            "queue_size": self.priority_queue.size(),
            "is_running": self.is_running,
            "throughput_rps": self.total_requests / max(1, time.time() - getattr(self, 'start_time', time.time()))
        }

# 全局实例
adaptive_batch_scheduler = AdaptiveBatchScheduler()

async def submit_batch_request(
    request_type: RequestType,
    data: Any,
    user_id: str = "default",
    priority: RequestPriority = RequestPriority.NORMAL
) -> str:
    """提交批处理请求的便捷函数"""
    request = ProcessRequest(
        id=f"req_{int(time.time()*1000)}_{random.randint(1000, 9999)}",
        type=request_type,
        priority=priority,
        data=data,
        user_id=user_id
    )
    
    return await adaptive_batch_scheduler.submit_request(request)

if __name__ == "__main__":
    # 测试代码
    async def test_adaptive_scheduler():
        # 启动调度器
        await adaptive_batch_scheduler.start()
        
        # 提交测试请求
        tasks = []
        for i in range(50):
            request_type = random.choice(list(RequestType))
            priority = random.choice(list(RequestPriority))
            
            task = submit_batch_request(
                request_type=request_type,
                data=f"test_data_{i}",
                user_id=f"user_{i % 10}",
                priority=priority
            )
            tasks.append(task)
        
        # 等待请求提交
        request_ids = await asyncio.gather(*tasks)
        print(f"Submitted {len(request_ids)} requests")
        
        # 等待处理
        await asyncio.sleep(5)
        
        # 获取统计信息
        stats = adaptive_batch_scheduler.get_statistics()
        print(f"\n调度器统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 停止调度器
        await adaptive_batch_scheduler.stop()
    
    # 运行测试
    asyncio.run(test_adaptive_scheduler())
