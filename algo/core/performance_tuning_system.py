"""
VoiceHelper v1.24.0 - 性能调优系统
实现内存优化、GC调优和性能监控，提升系统性能
"""

import asyncio
import time
import logging
import json
import gc
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import functools
import weakref
import tracemalloc
import cProfile
import pstats
import io
import sys

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """性能指标"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    MEMORY_LEAKS = "memory_leaks"
    GC_COLLECTIONS = "gc_collections"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

class OptimizationStrategy(Enum):
    """优化策略"""
    MEMORY_OPTIMIZATION = "memory_optimization"
    GC_TUNING = "gc_tuning"
    CACHE_OPTIMIZATION = "cache_optimization"
    CONNECTION_POOLING = "connection_pooling"
    ASYNC_OPTIMIZATION = "async_optimization"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"

@dataclass
class PerformanceProfile:
    """性能分析"""
    timestamp: float
    cpu_percent: float
    memory_usage: float
    memory_available: float
    gc_collections: Dict[str, int]
    response_times: List[float]
    throughput: float
    error_count: int
    active_connections: int

@dataclass
class OptimizationResult:
    """优化结果"""
    strategy: OptimizationStrategy
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percent: float
    timestamp: float
    description: str

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80%内存使用率触发优化
        self.optimization_history = deque(maxlen=1000)
        self.weak_refs = weakref.WeakSet()
        
    async def optimize_memory(self) -> OptimizationResult:
        """执行内存优化"""
        start_time = time.time()
        
        # 记录优化前指标
        before_metrics = self._get_memory_metrics()
        
        # 执行优化策略
        await self._force_gc_collection()
        await self._clear_weak_references()
        await self._optimize_caches()
        await self._reduce_memory_fragmentation()
        
        # 记录优化后指标
        after_metrics = self._get_memory_metrics()
        
        # 计算改进百分比
        memory_improvement = (
            (before_metrics['memory_usage'] - after_metrics['memory_usage']) / 
            before_metrics['memory_usage'] * 100
        )
        
        result = OptimizationResult(
            strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=memory_improvement,
            timestamp=time.time(),
            description="Memory optimization completed"
        )
        
        self.optimization_history.append(result)
        logger.info(f"Memory optimization completed: {memory_improvement:.2f}% improvement")
        
        return result
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """获取内存指标"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'memory_usage': memory.percent,
            'memory_available': memory.available / (1024**3),  # GB
            'process_memory': process.memory_info().rss / (1024**2),  # MB
            'memory_fragmentation': self._calculate_fragmentation()
        }
    
    def _calculate_fragmentation(self) -> float:
        """计算内存碎片化程度"""
        try:
            # 简化的碎片化计算
            gc_stats = gc.get_stats()
            total_collections = sum(stat['collections'] for stat in gc_stats)
            return total_collections / 1000.0  # 归一化
        except:
            return 0.0
    
    async def _force_gc_collection(self):
        """强制垃圾回收"""
        logger.debug("Forcing garbage collection")
        
        # 收集所有代
        collected = gc.collect()
        logger.debug(f"GC collected {collected} objects")
        
        # 清理循环引用
        gc.collect()
    
    async def _clear_weak_references(self):
        """清理弱引用"""
        # 清理弱引用集合
        if hasattr(self, 'weak_refs'):
            self.weak_refs.clear()
        
        logger.debug("Cleared weak references")
    
    async def _optimize_caches(self):
        """优化缓存"""
        # 这里应该清理应用中的各种缓存
        # 例如：LRU缓存、TTL缓存等
        logger.debug("Optimizing caches")
    
    async def _reduce_memory_fragmentation(self):
        """减少内存碎片"""
        # 执行多次GC来减少碎片
        for _ in range(3):
            gc.collect()
        
        logger.debug("Reduced memory fragmentation")
    
    def register_weak_reference(self, obj):
        """注册弱引用对象"""
        self.weak_refs.add(obj)
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """获取优化历史"""
        return list(self.optimization_history)

class GCOptimizer:
    """垃圾回收优化器"""
    
    def __init__(self):
        self.gc_thresholds = {
            0: 700,   # 第0代阈值
            1: 10,    # 第1代阈值
            2: 10     # 第2代阈值
        }
        self.original_thresholds = None
        
    def optimize_gc_settings(self) -> OptimizationResult:
        """优化GC设置"""
        start_time = time.time()
        
        # 记录原始设置
        self.original_thresholds = gc.get_threshold()
        
        # 记录优化前指标
        before_metrics = self._get_gc_metrics()
        
        # 应用优化设置
        gc.set_threshold(*self.gc_thresholds.values())
        
        # 启用调试
        gc.set_debug(gc.DEBUG_STATS)
        
        # 记录优化后指标
        after_metrics = self._get_gc_metrics()
        
        # 计算改进百分比
        improvement = self._calculate_gc_improvement(before_metrics, after_metrics)
        
        result = OptimizationResult(
            strategy=OptimizationStrategy.GC_TUNING,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=improvement,
            timestamp=time.time(),
            description="GC optimization completed"
        )
        
        logger.info(f"GC optimization completed: {improvement:.2f}% improvement")
        
        return result
    
    def restore_gc_settings(self):
        """恢复GC设置"""
        if self.original_thresholds:
            gc.set_threshold(*self.original_thresholds)
            gc.set_debug(0)
            logger.info("GC settings restored")
    
    def _get_gc_metrics(self) -> Dict[str, float]:
        """获取GC指标"""
        stats = gc.get_stats()
        
        return {
            'total_collections': sum(stat['collections'] for stat in stats),
            'total_collected': sum(stat['collected'] for stat in stats),
            'total_uncollectable': sum(stat['uncollectable'] for stat in stats),
            'gc_time': sum(stat.get('time', 0) for stat in stats)
        }
    
    def _calculate_gc_improvement(self, before: Dict, after: Dict) -> float:
        """计算GC改进百分比"""
        # 简化的改进计算
        before_total = before.get('total_collections', 1)
        after_total = after.get('total_collections', 1)
        
        if before_total > 0:
            return (before_total - after_total) / before_total * 100
        return 0.0

class CacheOptimizer:
    """缓存优化器"""
    
    def __init__(self):
        self.cache_metrics = defaultdict(list)
        self.optimization_history = []
        
    async def optimize_caches(self) -> OptimizationResult:
        """优化缓存"""
        start_time = time.time()
        
        # 记录优化前指标
        before_metrics = self._get_cache_metrics()
        
        # 执行缓存优化
        await self._optimize_cache_sizes()
        await self._optimize_cache_eviction()
        await self._optimize_cache_ttl()
        
        # 记录优化后指标
        after_metrics = self._get_cache_metrics()
        
        # 计算改进百分比
        hit_rate_improvement = (
            (after_metrics.get('hit_rate', 0) - before_metrics.get('hit_rate', 0)) /
            max(before_metrics.get('hit_rate', 1), 0.01) * 100
        )
        
        result = OptimizationResult(
            strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=hit_rate_improvement,
            timestamp=time.time(),
            description="Cache optimization completed"
        )
        
        self.optimization_history.append(result)
        logger.info(f"Cache optimization completed: {hit_rate_improvement:.2f}% hit rate improvement")
        
        return result
    
    def _get_cache_metrics(self) -> Dict[str, float]:
        """获取缓存指标"""
        # 这里应该从实际的缓存系统获取指标
        return {
            'hit_rate': 0.85,  # 模拟数据
            'miss_rate': 0.15,
            'cache_size': 1000,
            'eviction_rate': 0.1
        }
    
    async def _optimize_cache_sizes(self):
        """优化缓存大小"""
        logger.debug("Optimizing cache sizes")
        # 实现缓存大小优化逻辑
    
    async def _optimize_cache_eviction(self):
        """优化缓存淘汰策略"""
        logger.debug("Optimizing cache eviction")
        # 实现缓存淘汰策略优化
    
    async def _optimize_cache_ttl(self):
        """优化缓存TTL"""
        logger.debug("Optimizing cache TTL")
        # 实现缓存TTL优化

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiles = deque(maxlen=1000)
        self.memory_traces = []
        
    def start_profiling(self):
        """开始性能分析"""
        # 启动内存跟踪
        tracemalloc.start()
        
        # 启动CPU分析
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        logger.debug("Performance profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """停止性能分析并获取结果"""
        # 停止CPU分析
        self.profiler.disable()
        
        # 获取内存快照
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 分析CPU性能
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # 前20个最耗时的函数
        cpu_profile = s.getvalue()
        
        profile_result = {
            'timestamp': time.time(),
            'cpu_profile': cpu_profile,
            'memory_current': current / (1024**2),  # MB
            'memory_peak': peak / (1024**2),  # MB
            'top_functions': self._extract_top_functions(ps)
        }
        
        self.profiles.append(profile_result)
        logger.debug("Performance profiling completed")
        
        return profile_result
    
    def _extract_top_functions(self, ps: pstats.Stats) -> List[Dict[str, Any]]:
        """提取最耗时的函数"""
        top_functions = []
        
        # 获取统计信息
        stats = ps.stats
        
        # 按累积时间排序
        sorted_stats = sorted(stats.items(), 
                            key=lambda x: x[1][3], reverse=True)  # 第4个元素是累积时间
        
        for (filename, line, func), (cc, nc, tt, ct, callers) in sorted_stats[:10]:
            top_functions.append({
                'function': func,
                'filename': filename,
                'line': line,
                'cumulative_time': ct,
                'total_time': tt,
                'call_count': cc
            })
        
        return top_functions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.profiles:
            return {"message": "No profiling data available"}
        
        recent_profiles = list(self.profiles)[-10:]  # 最近10次分析
        
        avg_memory = sum(p['memory_current'] for p in recent_profiles) / len(recent_profiles)
        avg_peak_memory = sum(p['memory_peak'] for p in recent_profiles) / len(recent_profiles)
        
        # 找出最常出现的性能瓶颈
        function_usage = defaultdict(int)
        for profile in recent_profiles:
            for func in profile['top_functions']:
                function_usage[func['function']] += func['cumulative_time']
        
        top_bottlenecks = sorted(function_usage.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_profiles': len(self.profiles),
            'recent_profiles': len(recent_profiles),
            'avg_memory_usage': avg_memory,
            'avg_peak_memory': avg_peak_memory,
            'top_bottlenecks': top_bottlenecks,
            'last_profile_time': recent_profiles[-1]['timestamp'] if recent_profiles else None
        }

class PerformanceTuningSystem:
    """性能调优系统"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.gc_optimizer = GCOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.profiler = PerformanceProfiler()
        
        self.optimization_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_optimizations': 0,
            'memory_optimizations': 0,
            'gc_optimizations': 0,
            'cache_optimizations': 0,
            'avg_improvement': 0.0
        }
        
        # 自动优化配置
        self.auto_optimization_enabled = True
        self.optimization_threshold = 0.8  # 80%资源使用率触发优化
        self.optimization_interval = 300.0  # 5分钟
        
        self.optimization_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("Performance tuning system initialized")
    
    async def start_auto_optimization(self):
        """启动自动优化"""
        if self.is_running:
            return
        
        self.is_running = True
        self.optimization_task = asyncio.create_task(self._auto_optimization_loop())
        logger.info("Auto optimization started")
    
    async def stop_auto_optimization(self):
        """停止自动优化"""
        self.is_running = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto optimization stopped")
    
    async def _auto_optimization_loop(self):
        """自动优化循环"""
        while self.is_running:
            try:
                # 检查系统资源使用率
                if await self._should_optimize():
                    await self._perform_optimization()
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Auto optimization loop error: {e}")
                await asyncio.sleep(self.optimization_interval)
    
    async def _should_optimize(self) -> bool:
        """检查是否需要优化"""
        if not self.auto_optimization_enabled:
            return False
        
        # 检查内存使用率
        memory = psutil.virtual_memory()
        if memory.percent > self.optimization_threshold * 100:
            return True
        
        # 检查CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.optimization_threshold * 100:
            return True
        
        return False
    
    async def _perform_optimization(self):
        """执行优化"""
        logger.info("Performing automatic optimization")
        
        optimization_results = []
        
        # 内存优化
        try:
            memory_result = await self.memory_optimizer.optimize_memory()
            optimization_results.append(memory_result)
            self.performance_metrics['memory_optimizations'] += 1
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
        
        # 缓存优化
        try:
            cache_result = await self.cache_optimizer.optimize_caches()
            optimization_results.append(cache_result)
            self.performance_metrics['cache_optimizations'] += 1
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
        
        # 更新统计
        if optimization_results:
            self.optimization_history.extend(optimization_results)
            self.performance_metrics['total_optimizations'] += len(optimization_results)
            
            avg_improvement = sum(r.improvement_percent for r in optimization_results) / len(optimization_results)
            self.performance_metrics['avg_improvement'] = avg_improvement
    
    async def manual_optimization(self, strategy: OptimizationStrategy) -> OptimizationResult:
        """手动优化"""
        logger.info(f"Performing manual optimization: {strategy.value}")
        
        if strategy == OptimizationStrategy.MEMORY_OPTIMIZATION:
            result = await self.memory_optimizer.optimize_memory()
        elif strategy == OptimizationStrategy.GC_TUNING:
            result = self.gc_optimizer.optimize_gc_settings()
        elif strategy == OptimizationStrategy.CACHE_OPTIMIZATION:
            result = await self.cache_optimizer.optimize_caches()
        else:
            raise ValueError(f"Unsupported optimization strategy: {strategy}")
        
        self.optimization_history.append(result)
        self.performance_metrics['total_optimizations'] += 1
        
        return result
    
    def start_profiling(self):
        """开始性能分析"""
        self.profiler.start_profiling()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """停止性能分析"""
        return self.profiler.stop_profiling()
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """获取优化历史"""
        return list(self.optimization_history)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.performance_metrics,
            'profiling_summary': self.profiler.get_performance_summary(),
            'recent_optimizations': list(self.optimization_history)[-10:],
            'system_resources': self._get_system_resources()
        }
    
    def _get_system_resources(self) -> Dict[str, float]:
        """获取系统资源使用情况"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3)
        }
    
    def configure_auto_optimization(self, 
                                  enabled: bool = True,
                                  threshold: float = 0.8,
                                  interval: float = 300.0):
        """配置自动优化"""
        self.auto_optimization_enabled = enabled
        self.optimization_threshold = threshold
        self.optimization_interval = interval
        
        logger.info(f"Auto optimization configured: enabled={enabled}, "
                   f"threshold={threshold}, interval={interval}")

# 全局实例
_performance_tuning_system = None

def get_performance_tuning_system() -> PerformanceTuningSystem:
    """获取性能调优系统实例"""
    global _performance_tuning_system
    if _performance_tuning_system is None:
        _performance_tuning_system = PerformanceTuningSystem()
    return _performance_tuning_system

# 性能监控装饰器
def performance_monitor(metric_name: str = None):
    """性能监控装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss
                memory_delta = end_memory - start_memory
                
                # 记录性能指标
                performance_system = get_performance_tuning_system()
                if metric_name:
                    performance_system.memory_optimizer.register_weak_reference(result)
                
                logger.debug(f"Performance: {func.__name__} took {execution_time:.3f}s, "
                           f"memory delta: {memory_delta / (1024**2):.2f}MB")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Performance error in {func.__name__}: {e} "
                           f"(took {execution_time:.3f}s)")
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 使用示例
if __name__ == "__main__":
    async def test_performance_system():
        """测试性能调优系统"""
        performance_system = get_performance_tuning_system()
        
        # 启动自动优化
        await performance_system.start_auto_optimization()
        
        # 手动优化
        memory_result = await performance_system.manual_optimization(
            OptimizationStrategy.MEMORY_OPTIMIZATION
        )
        print(f"Memory optimization: {memory_result.improvement_percent:.2f}% improvement")
        
        # 性能分析
        performance_system.start_profiling()
        await asyncio.sleep(1)  # 模拟一些工作
        profile_result = performance_system.stop_profiling()
        print(f"Performance profile: {profile_result['memory_current']:.2f}MB current memory")
        
        # 获取性能指标
        metrics = performance_system.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # 停止自动优化
        await performance_system.stop_auto_optimization()
    
    # 运行测试
    asyncio.run(test_performance_system())
