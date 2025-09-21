"""
热点数据缓存系统

识别和缓存热点数据，提升高频访问性能
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import heapq
import threading

logger = logging.getLogger(__name__)


@dataclass
class AccessPattern:
    """访问模式"""
    key: str
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    access_frequency: float = 0.0  # 访问频率 (次/秒)
    access_times: List[float] = field(default_factory=list)
    
    def record_access(self):
        """记录访问"""
        current_time = time.time()
        self.access_count += 1
        self.last_access = current_time
        self.access_times.append(current_time)
        
        # 保持最近100次访问记录
        if len(self.access_times) > 100:
            self.access_times = self.access_times[-100:]
        
        # 计算访问频率
        if len(self.access_times) > 1:
            time_span = self.access_times[-1] - self.access_times[0]
            if time_span > 0:
                self.access_frequency = len(self.access_times) / time_span
    
    def get_hotness_score(self, current_time: Optional[float] = None) -> float:
        """计算热度评分"""
        if current_time is None:
            current_time = time.time()
        
        # 时间衰减因子
        time_decay = max(0.1, 1.0 - (current_time - self.last_access) / 3600.0)  # 1小时衰减
        
        # 频率权重
        frequency_weight = min(10.0, self.access_frequency)
        
        # 总访问次数权重
        count_weight = min(5.0, self.access_count / 10.0)
        
        # 综合热度评分
        hotness = (frequency_weight * 0.5 + count_weight * 0.3) * time_decay
        return hotness


@dataclass
class HotspotEntry:
    """热点缓存条目"""
    key: str
    data: Any
    access_pattern: AccessPattern
    cached_at: float = field(default_factory=time.time)
    ttl: float = 1800.0  # 30分钟默认TTL
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.cached_at > self.ttl
    
    def access(self):
        """记录访问"""
        self.access_pattern.record_access()


class HotspotDetector:
    """热点检测器"""
    
    def __init__(
        self,
        detection_window: float = 300.0,  # 5分钟检测窗口
        min_access_count: int = 5,        # 最小访问次数
        min_frequency: float = 0.01       # 最小频率 (次/秒)
    ):
        self.detection_window = detection_window
        self.min_access_count = min_access_count
        self.min_frequency = min_frequency
        
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.hotspot_candidates: List[Tuple[float, str]] = []  # (hotness_score, key)
        
        self._lock = threading.RLock()
    
    def record_access(self, key: str):
        """记录访问"""
        with self._lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = AccessPattern(key=key)
            
            self.access_patterns[key].record_access()
    
    def detect_hotspots(self, top_k: int = 10) -> List[str]:
        """检测热点数据"""
        with self._lock:
            current_time = time.time()
            hotspots = []
            
            for key, pattern in self.access_patterns.items():
                # 检查是否满足热点条件
                if (pattern.access_count >= self.min_access_count and 
                    pattern.access_frequency >= self.min_frequency and
                    current_time - pattern.last_access < self.detection_window):
                    
                    hotness_score = pattern.get_hotness_score(current_time)
                    hotspots.append((hotness_score, key))
            
            # 排序并返回top-k
            hotspots.sort(reverse=True)
            return [key for _, key in hotspots[:top_k]]
    
    def get_access_pattern(self, key: str) -> Optional[AccessPattern]:
        """获取访问模式"""
        with self._lock:
            return self.access_patterns.get(key)
    
    def cleanup_old_patterns(self, max_age: float = 3600.0):
        """清理旧的访问模式"""
        with self._lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, pattern in self.access_patterns.items():
                if current_time - pattern.last_access > max_age:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.access_patterns[key]
            
            logger.info(f"Cleaned up {len(keys_to_remove)} old access patterns")


class HotspotCache:
    """热点数据缓存"""
    
    def __init__(
        self,
        max_size: int = 100,
        detection_interval: float = 60.0,  # 1分钟检测间隔
        auto_promote: bool = True
    ):
        self.max_size = max_size
        self.detection_interval = detection_interval
        self.auto_promote = auto_promote
        
        self.cache: Dict[str, HotspotEntry] = {}
        self.detector = HotspotDetector()
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'promotions': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # 后台检测任务
        self._detection_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动热点检测"""
        if not self._running:
            self._running = True
            self._detection_task = asyncio.create_task(self._detection_loop())
            logger.info("Hotspot cache started")
    
    async def stop(self):
        """停止热点检测"""
        self._running = False
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
        logger.info("Hotspot cache stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        self.stats['total_requests'] += 1
        
        # 记录访问
        self.detector.record_access(key)
        
        # 检查热点缓存
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                entry.access()
                self.stats['hits'] += 1
                return entry.data
            else:
                del self.cache[key]
        
        self.stats['misses'] += 1
        return None
    
    async def put(self, key: str, data: Any, ttl: Optional[float] = None):
        """存储数据到热点缓存"""
        if ttl is None:
            ttl = 1800.0  # 30分钟
        
        # 获取访问模式
        access_pattern = self.detector.get_access_pattern(key)
        if access_pattern is None:
            access_pattern = AccessPattern(key=key)
        
        entry = HotspotEntry(
            key=key,
            data=data,
            access_pattern=access_pattern,
            ttl=ttl
        )
        
        # 检查缓存容量
        if len(self.cache) >= self.max_size:
            await self._evict_least_hot()
        
        self.cache[key] = entry
        logger.debug(f"Stored hotspot data: {key}")
    
    async def promote_to_hotspot(self, key: str, data: Any):
        """提升为热点数据"""
        await self.put(key, data)
        self.stats['promotions'] += 1
        logger.info(f"Promoted to hotspot: {key}")
    
    async def _evict_least_hot(self):
        """驱逐最不热的数据"""
        if not self.cache:
            return
        
        current_time = time.time()
        least_hot_key = None
        least_hotness = float('inf')
        
        for key, entry in self.cache.items():
            hotness = entry.access_pattern.get_hotness_score(current_time)
            if hotness < least_hotness:
                least_hotness = hotness
                least_hot_key = key
        
        if least_hot_key:
            del self.cache[least_hot_key]
            self.stats['evictions'] += 1
            logger.debug(f"Evicted least hot entry: {least_hot_key}")
    
    async def _detection_loop(self):
        """热点检测循环"""
        while self._running:
            try:
                await asyncio.sleep(self.detection_interval)
                
                if self.auto_promote:
                    # 检测新的热点
                    hotspots = self.detector.detect_hotspots(top_k=20)
                    
                    for key in hotspots:
                        if key not in self.cache:
                            # 这里需要从底层存储获取数据
                            # 暂时跳过，实际使用时需要集成数据源
                            pass
                
                # 清理过期条目
                await self._cleanup_expired()
                
                # 清理旧的访问模式
                self.detector.cleanup_old_patterns()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
    
    async def _cleanup_expired(self):
        """清理过期条目"""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired hotspot entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        hit_rate = self.stats['hits'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0.0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'access_patterns_count': len(self.detector.access_patterns)
        }
    
    def get_hotspot_analysis(self) -> Dict[str, Any]:
        """获取热点分析"""
        current_time = time.time()
        hotspots = []
        
        for key, entry in self.cache.items():
            hotness = entry.access_pattern.get_hotness_score(current_time)
            hotspots.append({
                'key': key,
                'hotness_score': hotness,
                'access_count': entry.access_pattern.access_count,
                'access_frequency': entry.access_pattern.access_frequency,
                'last_access': entry.access_pattern.last_access
            })
        
        hotspots.sort(key=lambda x: x['hotness_score'], reverse=True)
        
        return {
            'top_hotspots': hotspots[:10],
            'total_hotspots': len(hotspots),
            'avg_hotness': sum(h['hotness_score'] for h in hotspots) / len(hotspots) if hotspots else 0.0
        }


class AdaptiveHotspotCache:
    """自适应热点缓存"""
    
    def __init__(
        self,
        initial_size: int = 50,
        max_size: int = 200,
        size_adjustment_interval: float = 300.0  # 5分钟
    ):
        self.initial_size = initial_size
        self.max_size = max_size
        self.size_adjustment_interval = size_adjustment_interval
        
        self.hotspot_cache = HotspotCache(max_size=initial_size)
        self.performance_history: List[float] = []
        
        self._adjustment_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动自适应缓存"""
        await self.hotspot_cache.start()
        self._adjustment_task = asyncio.create_task(self._size_adjustment_loop())
        logger.info("Adaptive hotspot cache started")
    
    async def stop(self):
        """停止自适应缓存"""
        await self.hotspot_cache.stop()
        if self._adjustment_task:
            self._adjustment_task.cancel()
            try:
                await self._adjustment_task
            except asyncio.CancelledError:
                pass
        logger.info("Adaptive hotspot cache stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取数据"""
        return await self.hotspot_cache.get(key)
    
    async def put(self, key: str, data: Any, ttl: Optional[float] = None):
        """存储数据"""
        await self.hotspot_cache.put(key, data, ttl)
    
    async def _size_adjustment_loop(self):
        """缓存大小调整循环"""
        while True:
            try:
                await asyncio.sleep(self.size_adjustment_interval)
                
                # 获取当前性能指标
                stats = self.hotspot_cache.get_stats()
                current_hit_rate = stats['hit_rate']
                
                self.performance_history.append(current_hit_rate)
                if len(self.performance_history) > 10:
                    self.performance_history = self.performance_history[-10:]
                
                # 调整缓存大小
                await self._adjust_cache_size(current_hit_rate)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Size adjustment error: {e}")
    
    async def _adjust_cache_size(self, current_hit_rate: float):
        """调整缓存大小"""
        if len(self.performance_history) < 3:
            return
        
        # 计算性能趋势
        recent_avg = sum(self.performance_history[-3:]) / 3
        older_avg = sum(self.performance_history[-6:-3]) / 3 if len(self.performance_history) >= 6 else recent_avg
        
        performance_trend = recent_avg - older_avg
        current_size = self.hotspot_cache.max_size
        
        # 调整策略
        if performance_trend > 0.05 and current_hit_rate > 0.8:
            # 性能提升且命中率高，可以适当增加缓存大小
            new_size = min(self.max_size, int(current_size * 1.2))
        elif performance_trend < -0.05 or current_hit_rate < 0.5:
            # 性能下降或命中率低，减少缓存大小
            new_size = max(self.initial_size, int(current_size * 0.8))
        else:
            new_size = current_size
        
        if new_size != current_size:
            self.hotspot_cache.max_size = new_size
            logger.info(f"Adjusted hotspot cache size: {current_size} -> {new_size}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.hotspot_cache.get_stats()
        stats['performance_history'] = self.performance_history
        stats['adaptive_max_size'] = self.max_size
        return stats


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 创建自适应热点缓存
    cache = AdaptiveHotspotCache(initial_size=20, max_size=100)
    await cache.start()
    
    try:
        # 模拟访问模式
        for i in range(100):
            key = f"data_{i % 10}"  # 创建热点数据
            
            # 获取数据
            data = await cache.get(key)
            if data is None:
                # 模拟从数据源获取
                data = f"Content for {key}"
                await cache.put(key, data)
            
            # 模拟不同的访问频率
            if i % 10 < 3:  # 前3个key访问更频繁
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.1)
        
        # 等待一段时间让检测器工作
        await asyncio.sleep(2)
        
        # 获取统计信息
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        
        # 获取热点分析
        analysis = cache.hotspot_cache.get_hotspot_analysis()
        print(f"Hotspot analysis: {analysis}")
        
    finally:
        await cache.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
