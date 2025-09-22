"""
内存优化器
实施对象池、缓存管理等内存优化策略
"""

import gc
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

T = TypeVar('T')

class ObjectPool(Generic[T]):
    """对象池 - 重用对象以减少内存分配"""
    
    def __init__(self, create_func: Callable[[], T], reset_func: Callable[[T], None] = None, max_size: int = 100):
        self.create_func = create_func
        self.reset_func = reset_func
        self.pool: List[T] = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0
    
    def get_object(self) -> T:
        """获取对象"""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
                self.reused_count += 1
                return obj
            else:
                obj = self.create_func()
                self.created_count += 1
                return obj
    
    def return_object(self, obj: T) -> None:
        """归还对象"""
        with self.lock:
            if len(self.pool) < self.max_size:
                if self.reset_func:
                    self.reset_func(obj)
                self.pool.append(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'max_size': self.max_size,
                'created_count': self.created_count,
                'reused_count': self.reused_count,
                'reuse_rate': self.reused_count / max(1, self.created_count + self.reused_count)
            }

class LRUCache:
    """LRU缓存 - 限制内存使用的缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾 (最近使用)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self.lock:
            if key in self.cache:
                # 更新现有值
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 删除最少使用的项
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'usage_rate': len(self.cache) / self.max_size
            }

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.weak_refs: List[weakref.ref] = []
        self.gc_stats = []
    
    def track_object(self, obj: Any) -> None:
        """跟踪对象"""
        self.weak_refs.append(weakref.ref(obj))
    
    def cleanup_dead_refs(self) -> int:
        """清理死亡引用"""
        alive_refs = []
        dead_count = 0
        
        for ref in self.weak_refs:
            if ref() is not None:
                alive_refs.append(ref)
            else:
                dead_count += 1
        
        self.weak_refs = alive_refs
        return dead_count
    
    def force_gc(self) -> Dict[str, int]:
        """强制垃圾回收"""
        before_counts = [len(gc.get_objects())]
        
        # 执行垃圾回收
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        after_counts = [len(gc.get_objects())]
        
        stats = {
            'objects_before': before_counts[0],
            'objects_after': after_counts[0],
            'objects_freed': before_counts[0] - after_counts[0],
            'collected_gen0': collected[0],
            'collected_gen1': collected[1],
            'collected_gen2': collected[2]
        }
        
        self.gc_stats.append({
            'timestamp': time.time(),
            'stats': stats
        })
        
        return stats
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 ** 2),
            'vms_mb': memory_info.vms / (1024 ** 2),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 ** 2),
            'gc_counts': gc.get_count(),
            'tracked_objects': len(self.weak_refs)
        }

class MemoryOptimizer:
    """内存优化器主类"""
    
    def __init__(self):
        self.object_pools: Dict[str, ObjectPool] = {}
        self.caches: Dict[str, LRUCache] = {}
        self.monitor = MemoryMonitor()
        self.optimization_enabled = True
    
    def create_object_pool(self, name: str, create_func: Callable, reset_func: Callable = None, max_size: int = 100) -> ObjectPool:
        """创建对象池"""
        pool = ObjectPool(create_func, reset_func, max_size)
        self.object_pools[name] = pool
        return pool
    
    def create_cache(self, name: str, max_size: int = 1000) -> LRUCache:
        """创建缓存"""
        cache = LRUCache(max_size)
        self.caches[name] = cache
        return cache
    
    def optimize_memory(self) -> Dict[str, Any]:
        """执行内存优化"""
        if not self.optimization_enabled:
            return {'status': 'disabled'}
        
        results = {}
        
        # 1. 清理死亡引用
        dead_refs = self.monitor.cleanup_dead_refs()
        results['dead_refs_cleaned'] = dead_refs
        
        # 2. 强制垃圾回收
        gc_stats = self.monitor.force_gc()
        results['gc_stats'] = gc_stats
        
        # 3. 获取内存信息
        memory_info = self.monitor.get_memory_info()
        results['memory_info'] = memory_info
        
        # 4. 对象池统计
        pool_stats = {}
        for name, pool in self.object_pools.items():
            pool_stats[name] = pool.get_stats()
        results['object_pools'] = pool_stats
        
        # 5. 缓存统计
        cache_stats = {}
        for name, cache in self.caches.items():
            cache_stats[name] = cache.get_stats()
        results['caches'] = cache_stats
        
        return results
    
    def get_optimization_report(self) -> str:
        """获取优化报告"""
        stats = self.optimize_memory()
        
        report = []
        report.append("=== 内存优化报告 ===")
        report.append(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 内存信息
        memory_info = stats['memory_info']
        report.append("内存使用情况:")
        report.append(f"  RSS内存: {memory_info['rss_mb']:.2f} MB")
        report.append(f"  VMS内存: {memory_info['vms_mb']:.2f} MB")
        report.append(f"  内存占用率: {memory_info['percent']:.2f}%")
        report.append(f"  可用内存: {memory_info['available_mb']:.2f} MB")
        report.append("")
        
        # 垃圾回收统计
        gc_stats = stats['gc_stats']
        report.append("垃圾回收统计:")
        report.append(f"  回收前对象数: {gc_stats['objects_before']}")
        report.append(f"  回收后对象数: {gc_stats['objects_after']}")
        report.append(f"  释放对象数: {gc_stats['objects_freed']}")
        report.append("")
        
        # 对象池统计
        if stats['object_pools']:
            report.append("对象池统计:")
            for name, pool_stat in stats['object_pools'].items():
                report.append(f"  {name}:")
                report.append(f"    池大小: {pool_stat['pool_size']}/{pool_stat['max_size']}")
                report.append(f"    重用率: {pool_stat['reuse_rate']:.2%}")
        
        # 缓存统计
        if stats['caches']:
            report.append("缓存统计:")
            for name, cache_stat in stats['caches'].items():
                report.append(f"  {name}:")
                report.append(f"    缓存大小: {cache_stat['size']}/{cache_stat['max_size']}")
                report.append(f"    命中率: {cache_stat['hit_rate']:.2%}")
        
        return "\n".join(report)

# 全局内存优化器实例
memory_optimizer = MemoryOptimizer()

# 便捷函数
def create_object_pool(name: str, create_func: Callable, reset_func: Callable = None, max_size: int = 100) -> ObjectPool:
    """创建对象池的便捷函数"""
    return memory_optimizer.create_object_pool(name, create_func, reset_func, max_size)

def create_cache(name: str, max_size: int = 1000) -> LRUCache:
    """创建缓存的便捷函数"""
    return memory_optimizer.create_cache(name, max_size)

def optimize_memory() -> Dict[str, Any]:
    """执行内存优化的便捷函数"""
    return memory_optimizer.optimize_memory()

def get_memory_report() -> str:
    """获取内存报告的便捷函数"""
    return memory_optimizer.get_optimization_report()

if __name__ == "__main__":
    # 测试内存优化器
    print("测试内存优化器...")
    
    # 创建测试对象池
    def create_test_object():
        return {'data': [0] * 1000}
    
    def reset_test_object(obj):
        obj['data'] = [0] * 1000
    
    pool = create_object_pool('test_pool', create_test_object, reset_test_object, 10)
    
    # 创建测试缓存
    cache = create_cache('test_cache', 100)
    
    # 测试对象池
    objs = []
    for i in range(20):
        obj = pool.get_object()
        objs.append(obj)
    
    for obj in objs:
        pool.return_object(obj)
    
    # 测试缓存
    for i in range(50):
        cache.put(f'key_{i}', f'value_{i}')
    
    for i in range(25):
        cache.get(f'key_{i}')
    
    # 生成报告
    print(get_memory_report())
