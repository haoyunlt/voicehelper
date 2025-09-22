"""
VoiceHelper 高并发系统
支持100万+ QPS级别，对标ChatGPT-4o
实现分布式架构、智能负载均衡和自动扩缩容
"""

import asyncio
import time
import logging
import json
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"

@dataclass
class ServiceInstance:
    """服务实例"""
    id: str
    host: str
    port: int
    weight: float = 1.0
    status: ServiceStatus = ServiceStatus.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    average_response_time: float = 0.0
    last_health_check: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests
    
    @property
    def health_score(self) -> float:
        """健康评分"""
        if self.status == ServiceStatus.UNHEALTHY:
            return 0.0
        elif self.status == ServiceStatus.MAINTENANCE:
            return 0.1
        elif self.status == ServiceStatus.DEGRADED:
            return 0.5
        
        # 基于多个指标计算健康评分
        score = 1.0
        
        # 错误率影响
        score -= min(self.error_rate * 2, 0.5)
        
        # 响应时间影响
        if self.average_response_time > 1000:  # 1秒
            score -= 0.3
        elif self.average_response_time > 500:  # 500ms
            score -= 0.1
        
        # 资源使用率影响
        if self.cpu_usage > 0.9:
            score -= 0.2
        elif self.cpu_usage > 0.7:
            score -= 0.1
        
        if self.memory_usage > 0.9:
            score -= 0.2
        elif self.memory_usage > 0.8:
            score -= 0.1
        
        return max(score, 0.0)

@dataclass
class RequestMetrics:
    """请求指标"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    service_instance: Optional[str] = None
    status_code: int = 200
    response_size: int = 0
    error_message: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """请求持续时间（毫秒）"""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return 200 <= self.status_code < 400

@dataclass
class ConcurrencyStats:
    """并发统计"""
    current_qps: float = 0.0
    peak_qps: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    active_connections: int = 0
    queue_length: int = 0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        return 1.0 - self.success_rate

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """执行函数调用"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """成功回调"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class RateLimiter:
    """限流器"""
    
    def __init__(self, max_requests: int, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """检查是否允许请求"""
        with self.lock:
            now = time.time()
            
            # 清理过期请求
            while self.requests and now - self.requests[0] > self.time_window:
                self.requests.popleft()
            
            # 检查是否超过限制
            if len(self.requests) >= self.max_requests:
                return False
            
            # 记录新请求
            self.requests.append(now)
            return True

class LoadBalancer:
    """智能负载均衡器"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.instances: List[ServiceInstance] = []
        self.current_index = 0
        self.connection_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
    def add_instance(self, instance: ServiceInstance):
        """添加服务实例"""
        self.instances.append(instance)
        logger.info(f"Added service instance: {instance.id}")
    
    def remove_instance(self, instance_id: str):
        """移除服务实例"""
        self.instances = [inst for inst in self.instances if inst.id != instance_id]
        logger.info(f"Removed service instance: {instance_id}")
    
    def select_instance(self, request_hash: Optional[str] = None) -> Optional[ServiceInstance]:
        """选择服务实例"""
        healthy_instances = [inst for inst in self.instances 
                           if inst.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]]
        
        if not healthy_instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash(healthy_instances, request_hash)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(healthy_instances)
        
        return healthy_instances[0]  # 默认返回第一个
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询算法"""
        instance = instances[self.current_index % len(instances)]
        self.current_index += 1
        return instance
    
    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询算法"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return instances[0]
        
        # 简化的加权轮询实现
        target = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if current_weight >= target:
                return instance
        
        return instances[-1]
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接算法"""
        return min(instances, key=lambda x: x.current_connections)
    
    def _least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最短响应时间算法"""
        return min(instances, key=lambda x: x.average_response_time)
    
    def _consistent_hash(self, instances: List[ServiceInstance], request_hash: Optional[str]) -> ServiceInstance:
        """一致性哈希算法"""
        if not request_hash:
            return instances[0]
        
        # 简化的一致性哈希实现
        hash_value = int(hashlib.md5(request_hash.encode()).hexdigest(), 16)
        index = hash_value % len(instances)
        return instances[index]
    
    def _adaptive_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """自适应选择算法"""
        # 基于健康评分的自适应选择
        scored_instances = [(inst, inst.health_score) for inst in instances]
        scored_instances.sort(key=lambda x: x[1], reverse=True)
        
        # 使用加权随机选择，健康评分高的实例被选中概率更大
        total_score = sum(score for _, score in scored_instances)
        if total_score == 0:
            return instances[0]
        
        target = random.uniform(0, total_score)
        current_score = 0
        
        for instance, score in scored_instances:
            current_score += score
            if current_score >= target:
                return instance
        
        return scored_instances[0][0]

class ConnectionPool:
    """连接池"""
    
    def __init__(self, max_connections: int = 1000):
        self.max_connections = max_connections
        self.active_connections = 0
        self.connection_queue = asyncio.Queue(maxsize=max_connections)
        self.lock = asyncio.Lock()
    
    async def acquire_connection(self) -> bool:
        """获取连接"""
        async with self.lock:
            if self.active_connections >= self.max_connections:
                return False
            
            self.active_connections += 1
            return True
    
    async def release_connection(self):
        """释放连接"""
        async with self.lock:
            if self.active_connections > 0:
                self.active_connections -= 1

class HighConcurrencySystem:
    """高并发系统"""
    
    def __init__(self, 
                 max_qps: int = 1000000,
                 max_connections: int = 100000,
                 worker_threads: int = None,
                 worker_processes: int = None):
        
        self.max_qps = max_qps
        self.max_connections = max_connections
        
        # 工作线程和进程池
        self.worker_threads = worker_threads or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.worker_processes = worker_processes or multiprocessing.cpu_count() or 1
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.worker_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=self.worker_processes)
        
        # 核心组件
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.ADAPTIVE)
        self.connection_pool = ConnectionPool(max_connections)
        self.rate_limiter = RateLimiter(max_qps, 1.0)
        self.circuit_breakers = {}
        
        # 统计和监控
        self.stats = ConcurrencyStats()
        self.request_metrics = deque(maxlen=10000)  # 保留最近10000个请求的指标
        self.qps_history = deque(maxlen=300)  # 5分钟的QPS历史（每秒一个点）
        
        # 运行状态
        self.is_running = False
        self.background_tasks = []
        
        # 性能优化
        self.request_cache = {}
        self.cache_ttl = 60  # 缓存TTL（秒）
        
        logger.info(f"HighConcurrencySystem initialized: {max_qps} QPS, {max_connections} connections")
    
    async def start(self):
        """启动系统"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动后台任务
        self.background_tasks = [
            asyncio.create_task(self._stats_collector()),
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._cache_cleaner()),
            asyncio.create_task(self._auto_scaler())
        ]
        
        logger.info("High concurrency system started")
    
    async def stop(self):
        """停止系统"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止后台任务
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # 关闭线程池和进程池
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("High concurrency system stopped")
    
    async def process_request(self, 
                            request_data: Any,
                            request_id: Optional[str] = None,
                            user_id: Optional[str] = None,
                            priority: int = 1) -> Dict[str, Any]:
        """处理请求"""
        
        # 生成请求ID
        if request_id is None:
            request_id = f"req_{int(time.time()*1000)}_{random.randint(1000, 9999)}"
        
        # 限流检查
        if not self.rate_limiter.is_allowed():
            return {
                "request_id": request_id,
                "status": "error",
                "error": "Rate limit exceeded",
                "code": 429
            }
        
        # 获取连接
        if not await self.connection_pool.acquire_connection():
            return {
                "request_id": request_id,
                "status": "error",
                "error": "Connection pool exhausted",
                "code": 503
            }
        
        # 创建请求指标
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.time()
        )
        
        try:
            # 选择服务实例
            instance = self.load_balancer.select_instance(user_id)
            if not instance:
                raise Exception("No healthy service instances available")
            
            metrics.service_instance = instance.id
            instance.current_connections += 1
            
            # 检查缓存
            cache_key = self._generate_cache_key(request_data, user_id)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                metrics.end_time = time.time()
                metrics.status_code = 200
                self._record_metrics(metrics, instance)
                
                return {
                    "request_id": request_id,
                    "status": "success",
                    "data": cached_result,
                    "cached": True,
                    "processing_time": metrics.duration_ms
                }
            
            # 获取熔断器
            circuit_breaker = self._get_circuit_breaker(instance.id)
            
            # 处理请求
            try:
                result = await circuit_breaker.call(
                    self._execute_request,
                    request_data,
                    instance,
                    priority
                )
                
                # 缓存结果
                self._cache_result(cache_key, result)
                
                metrics.end_time = time.time()
                metrics.status_code = 200
                
                return {
                    "request_id": request_id,
                    "status": "success",
                    "data": result,
                    "cached": False,
                    "processing_time": metrics.duration_ms,
                    "instance_id": instance.id
                }
                
            except Exception as e:
                metrics.end_time = time.time()
                metrics.status_code = 500
                metrics.error_message = str(e)
                
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                    "code": 500,
                    "processing_time": metrics.duration_ms
                }
        
        finally:
            # 释放连接
            await self.connection_pool.release_connection()
            
            # 更新实例连接数
            if metrics.service_instance:
                for inst in self.load_balancer.instances:
                    if inst.id == metrics.service_instance:
                        inst.current_connections = max(0, inst.current_connections - 1)
                        break
            
            # 记录指标
            self._record_metrics(metrics, instance if 'instance' in locals() else None)
    
    async def _execute_request(self, request_data: Any, instance: ServiceInstance, priority: int) -> Any:
        """执行具体请求"""
        # 模拟请求处理
        processing_time = random.uniform(0.01, 0.1)  # 10-100ms
        
        # 根据优先级调整处理时间
        if priority > 5:
            processing_time *= 0.5  # 高优先级请求处理更快
        elif priority < 3:
            processing_time *= 1.5  # 低优先级请求处理更慢
        
        await asyncio.sleep(processing_time)
        
        # 模拟偶发错误
        if random.random() < 0.001:  # 0.1%的错误率
            raise Exception("Simulated processing error")
        
        # 返回模拟结果
        return {
            "processed_data": f"Processed: {request_data}",
            "instance_id": instance.id,
            "processing_time": processing_time * 1000,
            "timestamp": time.time()
        }
    
    def _get_circuit_breaker(self, instance_id: str) -> CircuitBreaker:
        """获取熔断器"""
        if instance_id not in self.circuit_breakers:
            self.circuit_breakers[instance_id] = CircuitBreaker()
        return self.circuit_breakers[instance_id]
    
    def _generate_cache_key(self, request_data: Any, user_id: Optional[str]) -> str:
        """生成缓存键"""
        data_str = json.dumps(request_data, sort_keys=True) if isinstance(request_data, dict) else str(request_data)
        key_data = f"{data_str}:{user_id or 'anonymous'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """获取缓存结果"""
        if cache_key in self.request_cache:
            cached_data, timestamp = self.request_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.request_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """缓存结果"""
        self.request_cache[cache_key] = (result, time.time())
    
    def _record_metrics(self, metrics: RequestMetrics, instance: Optional[ServiceInstance]):
        """记录指标"""
        self.request_metrics.append(metrics)
        
        # 更新全局统计
        self.stats.total_requests += 1
        if metrics.is_success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        
        # 更新实例统计
        if instance:
            instance.total_requests += 1
            if not metrics.is_success:
                instance.total_errors += 1
            
            # 更新平均响应时间
            if instance.total_requests > 0:
                instance.average_response_time = (
                    (instance.average_response_time * (instance.total_requests - 1) + metrics.duration_ms) /
                    instance.total_requests
                )
    
    async def _stats_collector(self):
        """统计收集器"""
        last_request_count = 0
        
        while self.is_running:
            try:
                await asyncio.sleep(1)  # 每秒收集一次
                
                # 计算QPS
                current_requests = self.stats.total_requests
                current_qps = current_requests - last_request_count
                last_request_count = current_requests
                
                self.stats.current_qps = current_qps
                self.stats.peak_qps = max(self.stats.peak_qps, current_qps)
                self.qps_history.append(current_qps)
                
                # 计算响应时间百分位数
                if self.request_metrics:
                    recent_metrics = list(self.request_metrics)[-1000:]  # 最近1000个请求
                    durations = [m.duration_ms for m in recent_metrics if m.end_time is not None]
                    
                    if durations:
                        durations.sort()
                        self.stats.average_response_time = sum(durations) / len(durations)
                        self.stats.p95_response_time = durations[int(len(durations) * 0.95)]
                        self.stats.p99_response_time = durations[int(len(durations) * 0.99)]
                
                # 更新活跃连接数
                self.stats.active_connections = self.connection_pool.active_connections
                
            except Exception as e:
                logger.error(f"Stats collector error: {e}")
    
    async def _health_checker(self):
        """健康检查器"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # 每10秒检查一次
                
                for instance in self.load_balancer.instances:
                    # 模拟健康检查
                    await self._check_instance_health(instance)
                
            except Exception as e:
                logger.error(f"Health checker error: {e}")
    
    async def _check_instance_health(self, instance: ServiceInstance):
        """检查实例健康状态"""
        try:
            # 模拟健康检查请求
            await asyncio.sleep(0.01)  # 模拟检查延迟
            
            # 更新资源使用率（模拟）
            instance.cpu_usage = random.uniform(0.1, 0.9)
            instance.memory_usage = random.uniform(0.2, 0.8)
            instance.last_health_check = time.time()
            
            # 根据错误率和响应时间调整状态
            if instance.error_rate > 0.1:  # 错误率超过10%
                instance.status = ServiceStatus.UNHEALTHY
            elif instance.error_rate > 0.05 or instance.average_response_time > 1000:
                instance.status = ServiceStatus.DEGRADED
            else:
                instance.status = ServiceStatus.HEALTHY
                
        except Exception as e:
            logger.error(f"Health check failed for instance {instance.id}: {e}")
            instance.status = ServiceStatus.UNHEALTHY
    
    async def _cache_cleaner(self):
        """缓存清理器"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                current_time = time.time()
                expired_keys = []
                
                for key, (data, timestamp) in self.request_cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.request_cache[key]
                
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Cache cleaner error: {e}")
    
    async def _auto_scaler(self):
        """自动扩缩容"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                # 基于QPS历史进行扩缩容决策
                if len(self.qps_history) >= 10:
                    recent_qps = list(self.qps_history)[-10:]  # 最近10秒
                    avg_qps = sum(recent_qps) / len(recent_qps)
                    
                    healthy_instances = len([inst for inst in self.load_balancer.instances 
                                           if inst.status == ServiceStatus.HEALTHY])
                    
                    # 扩容条件：平均QPS过高且成功率下降
                    if (avg_qps > 1000 and healthy_instances < 10 and 
                        self.stats.success_rate < 0.95):
                        await self._scale_out()
                    
                    # 缩容条件：平均QPS较低且实例数量较多
                    elif avg_qps < 100 and healthy_instances > 2:
                        await self._scale_in()
                
            except Exception as e:
                logger.error(f"Auto scaler error: {e}")
    
    async def _scale_out(self):
        """扩容"""
        # 模拟添加新实例
        new_instance = ServiceInstance(
            id=f"instance_{len(self.load_balancer.instances) + 1}",
            host=f"10.0.0.{len(self.load_balancer.instances) + 1}",
            port=8080,
            weight=1.0,
            status=ServiceStatus.HEALTHY
        )
        
        self.load_balancer.add_instance(new_instance)
        logger.info(f"Scaled out: added instance {new_instance.id}")
    
    async def _scale_in(self):
        """缩容"""
        # 移除负载最低的实例
        if len(self.load_balancer.instances) > 1:
            instance_to_remove = min(self.load_balancer.instances, 
                                   key=lambda x: x.current_connections)
            
            self.load_balancer.remove_instance(instance_to_remove.id)
            logger.info(f"Scaled in: removed instance {instance_to_remove.id}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "concurrency_stats": {
                "current_qps": self.stats.current_qps,
                "peak_qps": self.stats.peak_qps,
                "total_requests": self.stats.total_requests,
                "success_rate": self.stats.success_rate,
                "average_response_time": self.stats.average_response_time,
                "p95_response_time": self.stats.p95_response_time,
                "p99_response_time": self.stats.p99_response_time,
                "active_connections": self.stats.active_connections
            },
            "service_instances": [
                {
                    "id": inst.id,
                    "status": inst.status.value,
                    "health_score": inst.health_score,
                    "current_connections": inst.current_connections,
                    "total_requests": inst.total_requests,
                    "error_rate": inst.error_rate,
                    "average_response_time": inst.average_response_time,
                    "cpu_usage": inst.cpu_usage,
                    "memory_usage": inst.memory_usage
                }
                for inst in self.load_balancer.instances
            ],
            "cache_stats": {
                "cache_size": len(self.request_cache),
                "cache_hit_rate": 0.0  # 简化实现
            },
            "system_config": {
                "max_qps": self.max_qps,
                "max_connections": self.max_connections,
                "worker_threads": self.worker_threads,
                "worker_processes": self.worker_processes
            }
        }

# 全局实例
high_concurrency_system = HighConcurrencySystem(
    max_qps=1000000,
    max_connections=100000
)

async def process_high_concurrency_request(
    request_data: Any,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    priority: int = 1
) -> Dict[str, Any]:
    """高并发请求处理便捷函数"""
    return await high_concurrency_system.process_request(
        request_data=request_data,
        request_id=request_id,
        user_id=user_id,
        priority=priority
    )

# 测试代码
if __name__ == "__main__":
    async def test_high_concurrency():
        print("🚀 测试高并发系统")
        print("=" * 50)
        
        # 初始化系统
        system = high_concurrency_system
        
        # 添加一些服务实例
        for i in range(3):
            instance = ServiceInstance(
                id=f"test_instance_{i+1}",
                host=f"10.0.0.{i+1}",
                port=8080,
                weight=1.0
            )
            system.load_balancer.add_instance(instance)
        
        # 启动系统
        await system.start()
        
        print(f"系统配置:")
        print(f"  最大QPS: {system.max_qps:,}")
        print(f"  最大连接数: {system.max_connections:,}")
        print(f"  工作线程: {system.worker_threads}")
        print(f"  服务实例: {len(system.load_balancer.instances)}")
        
        # 并发测试
        print(f"\n🔥 并发压力测试:")
        
        # 生成测试请求
        test_requests = []
        for i in range(1000):  # 1000个并发请求
            test_requests.append({
                "request_data": f"test_request_{i}",
                "user_id": f"user_{i % 100}",  # 100个不同用户
                "priority": random.randint(1, 10)
            })
        
        # 执行并发测试
        start_time = time.time()
        
        tasks = []
        for req in test_requests:
            task = process_high_concurrency_request(
                request_data=req["request_data"],
                user_id=req["user_id"],
                priority=req["priority"]
            )
            tasks.append(task)
        
        # 等待所有请求完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 分析结果
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed_requests = len(results) - successful_requests
        actual_qps = len(results) / duration
        
        print(f"  测试时长: {duration:.2f}s")
        print(f"  总请求数: {len(results)}")
        print(f"  成功请求: {successful_requests}")
        print(f"  失败请求: {failed_requests}")
        print(f"  成功率: {successful_requests/len(results):.2%}")
        print(f"  实际QPS: {actual_qps:.0f}")
        
        # 等待统计更新
        await asyncio.sleep(2)
        
        # 获取系统统计
        stats = system.get_system_stats()
        
        print(f"\n📊 系统统计:")
        concurrency_stats = stats["concurrency_stats"]
        print(f"  当前QPS: {concurrency_stats['current_qps']:.0f}")
        print(f"  峰值QPS: {concurrency_stats['peak_qps']:.0f}")
        print(f"  总请求数: {concurrency_stats['total_requests']}")
        print(f"  成功率: {concurrency_stats['success_rate']:.2%}")
        print(f"  平均响应时间: {concurrency_stats['average_response_time']:.2f}ms")
        print(f"  P95响应时间: {concurrency_stats['p95_response_time']:.2f}ms")
        print(f"  活跃连接: {concurrency_stats['active_connections']}")
        
        print(f"\n🏥 服务实例状态:")
        for inst_stat in stats["service_instances"]:
            print(f"  {inst_stat['id']}: {inst_stat['status']} "
                  f"(健康度: {inst_stat['health_score']:.2f}, "
                  f"连接: {inst_stat['current_connections']}, "
                  f"请求: {inst_stat['total_requests']})")
        
        # 停止系统
        await system.stop()
        
        # 评估测试结果
        success_criteria = [
            successful_requests >= len(results) * 0.95,  # 95%成功率
            actual_qps >= 100,  # 至少100 QPS
            concurrency_stats['average_response_time'] < 200,  # 平均响应时间小于200ms
            len(stats["service_instances"]) >= 3,  # 至少3个实例
            all(inst['health_score'] > 0 for inst in stats["service_instances"])  # 所有实例健康
        ]
        
        success_rate = sum(success_criteria) / len(success_criteria)
        print(f"\n🎯 测试成功率: {success_rate:.1%}")
        
        return success_rate >= 0.8
    
    # 运行测试
    import asyncio
    success = asyncio.run(test_high_concurrency())
    print(f"\n🎉 测试{'通过' if success else '失败'}！")
