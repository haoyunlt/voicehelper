"""
集成缓存服务

整合语义缓存、热点缓存和预热系统
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
import json

from ..core.semantic_cache import SemanticCache
from ..core.hotspot_cache import AdaptiveHotspotCache
from ..core.cache_prewarming import SmartPrewarmer

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    # 语义缓存配置
    semantic_cache_size: int = 1000
    semantic_similarity_threshold: float = 0.85
    semantic_ttl: float = 3600.0
    
    # 热点缓存配置
    hotspot_initial_size: int = 50
    hotspot_max_size: int = 200
    hotspot_detection_interval: float = 60.0
    
    # 预热配置
    enable_prewarming: bool = True
    prewarm_workers: int = 3
    prewarm_interval: float = 300.0
    
    # 性能配置
    enable_performance_monitoring: bool = True
    stats_collection_interval: float = 30.0


class IntegratedCacheService:
    """集成缓存服务"""
    
    def __init__(
        self,
        processing_func: Callable[[str, str, Dict[str, Any]], Awaitable[Any]],
        config: Optional[CacheConfig] = None
    ):
        self.processing_func = processing_func
        self.config = config or CacheConfig()
        
        # 缓存层
        self.semantic_cache = SemanticCache(
            max_size=self.config.semantic_cache_size,
            similarity_threshold=self.config.semantic_similarity_threshold,
            default_ttl=self.config.semantic_ttl
        )
        
        self.hotspot_cache = AdaptiveHotspotCache(
            initial_size=self.config.hotspot_initial_size,
            max_size=self.config.hotspot_max_size
        )
        
        # 预热系统
        self.prewarmer: Optional[SmartPrewarmer] = None
        if self.config.enable_prewarming:
            self.prewarmer = SmartPrewarmer(
                processing_func=self._process_with_caching,
                cache_put_func=self._put_to_all_caches,
                cache_get_func=self._get_from_semantic_cache,
                max_concurrent_tasks=self.config.prewarm_workers
            )
        
        # 统计和监控
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'hotspot_hits': 0,
            'semantic_hits': 0,
            'processing_time_saved': 0.0,
            'avg_response_time': 0.0
        }
        
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动缓存服务"""
        if not self._running:
            self._running = True
            
            # 启动热点缓存
            await self.hotspot_cache.start()
            
            # 启动预热系统
            if self.prewarmer:
                await self.prewarmer.start()
            
            # 启动性能监控
            if self.config.enable_performance_monitoring:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Integrated cache service started")
    
    async def stop(self):
        """停止缓存服务"""
        if self._running:
            self._running = False
            
            # 停止热点缓存
            await self.hotspot_cache.stop()
            
            # 停止预热系统
            if self.prewarmer:
                await self.prewarmer.stop()
            
            # 停止监控
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Integrated cache service stopped")
    
    async def get_response(
        self,
        content: str,
        model: str,
        parameters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Any:
        """获取响应 (带缓存)"""
        if parameters is None:
            parameters = {}
        
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        if use_cache:
            # 1. 尝试热点缓存 (最快)
            cache_key = self._generate_cache_key(content, model, parameters)
            hotspot_response = await self.hotspot_cache.get(cache_key)
            
            if hotspot_response is not None:
                self.stats['cache_hits'] += 1
                self.stats['hotspot_hits'] += 1
                
                # 记录查询用于预热
                if self.prewarmer:
                    await self.prewarmer.record_query_and_response(content, True)
                
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                
                logger.debug(f"Hotspot cache hit: {cache_key}")
                return hotspot_response
            
            # 2. 尝试语义缓存
            semantic_response = await self.semantic_cache.get(content, model, parameters)
            
            if semantic_response is not None:
                self.stats['cache_hits'] += 1
                self.stats['semantic_hits'] += 1
                
                # 提升到热点缓存
                await self.hotspot_cache.hotspot_cache.promote_to_hotspot(cache_key, semantic_response)
                
                # 记录查询
                if self.prewarmer:
                    await self.prewarmer.record_query_and_response(content, True)
                
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                
                logger.debug(f"Semantic cache hit: {cache_key}")
                return semantic_response
        
        # 3. 缓存未命中，调用处理函数
        self.stats['cache_misses'] += 1
        
        # 记录处理开始时间
        processing_start = time.time()
        
        try:
            response = await self.processing_func(content, model, parameters)
            
            processing_time = time.time() - processing_start
            
            # 存储到缓存
            if use_cache:
                await self._put_to_all_caches(content, model, response, parameters)
            
            # 记录查询
            if self.prewarmer:
                await self.prewarmer.record_query_and_response(content, False)
            
            total_response_time = time.time() - start_time
            self._update_response_time(total_response_time)
            
            logger.debug(f"Cache miss, processed in {processing_time:.3f}s: {content[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise
    
    async def _process_with_caching(
        self,
        content: str,
        model: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """带缓存的处理 (用于预热)"""
        return await self.get_response(content, model, parameters, use_cache=False)
    
    async def _put_to_all_caches(
        self,
        content: str,
        model: str,
        response: Any,
        parameters: Dict[str, Any]
    ):
        """存储到所有缓存层"""
        # 存储到语义缓存
        await self.semantic_cache.put(content, model, response, parameters)
        
        # 根据访问模式决定是否存储到热点缓存
        cache_key = self._generate_cache_key(content, model, parameters)
        access_pattern = self.hotspot_cache.hotspot_cache.detector.get_access_pattern(cache_key)
        
        if access_pattern and access_pattern.access_count >= 2:
            # 多次访问的数据存储到热点缓存
            await self.hotspot_cache.hotspot_cache.promote_to_hotspot(cache_key, response)
    
    async def _get_from_semantic_cache(
        self,
        content: str,
        model: str,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """从语义缓存获取 (用于预热)"""
        return await self.semantic_cache.get(content, model, parameters)
    
    def _generate_cache_key(self, content: str, model: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{content}:{model}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _update_response_time(self, response_time: float):
        """更新平均响应时间"""
        if self.stats['total_requests'] == 1:
            self.stats['avg_response_time'] = response_time
        else:
            # 指数移动平均
            alpha = 0.1
            self.stats['avg_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.stats['avg_response_time']
            )
    
    async def _monitoring_loop(self):
        """监控循环"""
        logger.info("Cache service monitoring started")
        
        while self._running:
            try:
                await asyncio.sleep(self.config.stats_collection_interval)
                
                # 收集统计信息
                await self._collect_performance_stats()
                
                # 执行自适应优化
                await self._adaptive_optimization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
        
        logger.info("Cache service monitoring stopped")
    
    async def _collect_performance_stats(self):
        """收集性能统计"""
        # 计算缓存命中率
        hit_rate = (
            self.stats['cache_hits'] / self.stats['total_requests'] 
            if self.stats['total_requests'] > 0 else 0.0
        )
        
        # 计算时间节省
        if self.stats['cache_hits'] > 0:
            # 假设缓存命中节省80%的处理时间
            estimated_processing_time = self.stats['avg_response_time'] / 0.2  # 假设缓存响应时间是处理时间的20%
            self.stats['processing_time_saved'] = (
                self.stats['cache_hits'] * estimated_processing_time * 0.8
            )
        
        # 记录性能指标
        if self.stats['total_requests'] % 100 == 0:  # 每100个请求记录一次
            logger.info(
                f"Cache performance: {hit_rate:.2%} hit rate, "
                f"{self.stats['avg_response_time']:.3f}s avg response time, "
                f"{self.stats['processing_time_saved']:.1f}s time saved"
            )
    
    async def _adaptive_optimization(self):
        """自适应优化"""
        # 获取各缓存层的统计信息
        semantic_stats = self.semantic_cache.get_stats()
        hotspot_stats = self.hotspot_cache.get_stats()
        
        # 动态调整语义缓存相似度阈值
        if semantic_stats['hit_rate'] < 0.3:
            # 命中率低，降低相似度阈值
            self.semantic_cache.similarity_threshold = max(0.75, self.semantic_cache.similarity_threshold - 0.05)
        elif semantic_stats['hit_rate'] > 0.8:
            # 命中率高，提高相似度阈值
            self.semantic_cache.similarity_threshold = min(0.95, self.semantic_cache.similarity_threshold + 0.02)
        
        # 执行自适应预热
        if self.prewarmer:
            await self.prewarmer.adaptive_prewarm()
    
    async def invalidate_cache(self, pattern: Optional[str] = None):
        """失效缓存"""
        self.semantic_cache.invalidate(pattern)
        
        if pattern:
            # 热点缓存的简单失效 (实际实现可能需要更复杂的逻辑)
            keys_to_remove = []
            for key in self.hotspot_cache.hotspot_cache.cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self.hotspot_cache.hotspot_cache.cache:
                    del self.hotspot_cache.hotspot_cache.cache[key]
        
        logger.info(f"Cache invalidated with pattern: {pattern}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        # 基础统计
        hit_rate = (
            self.stats['cache_hits'] / self.stats['total_requests'] 
            if self.stats['total_requests'] > 0 else 0.0
        )
        
        miss_rate = 1.0 - hit_rate
        
        # 性能提升计算
        if self.stats['cache_hits'] > 0 and self.stats['avg_response_time'] > 0:
            # 估算无缓存时的响应时间
            estimated_no_cache_time = self.stats['avg_response_time'] / (1 - hit_rate * 0.8)
            performance_improvement = (
                (estimated_no_cache_time - self.stats['avg_response_time']) / 
                estimated_no_cache_time
            )
        else:
            performance_improvement = 0.0
        
        comprehensive_stats = {
            # 基础统计
            'service_stats': {
                **self.stats,
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'performance_improvement': performance_improvement
            },
            
            # 各缓存层统计
            'semantic_cache_stats': self.semantic_cache.get_stats(),
            'hotspot_cache_stats': self.hotspot_cache.get_stats(),
            
            # 配置信息
            'config': {
                'semantic_cache_size': self.config.semantic_cache_size,
                'semantic_similarity_threshold': self.semantic_cache.similarity_threshold,
                'hotspot_max_size': self.config.hotspot_max_size,
                'prewarming_enabled': self.config.enable_prewarming
            }
        }
        
        # 预热统计
        if self.prewarmer:
            comprehensive_stats['prewarmer_stats'] = self.prewarmer.get_stats()
        
        return comprehensive_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试缓存功能
            test_content = "Health check test query"
            test_model = "gpt-3.5-turbo"
            test_params = {"temperature": 0.7}
            
            start_time = time.time()
            
            # 测试存储
            await self.semantic_cache.put(
                test_content, 
                test_model, 
                "Health check response", 
                test_params
            )
            
            # 测试获取
            response = await self.semantic_cache.get(test_content, test_model, test_params)
            
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'cache_functional': response is not None,
                'response_time': response_time,
                'running': self._running,
                'hotspot_cache_running': self.hotspot_cache.hotspot_cache._running if hasattr(self.hotspot_cache.hotspot_cache, '_running') else True,
                'prewarmer_running': self.prewarmer is not None
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'running': self._running
            }


# 使用示例和性能测试
async def example_usage():
    """使用示例"""
    
    # 模拟处理函数
    async def mock_processing_func(content: str, model: str, parameters: Dict[str, Any]) -> str:
        # 模拟处理延迟
        await asyncio.sleep(0.8)
        return f"AI response to: {content}"
    
    # 创建缓存配置
    config = CacheConfig(
        semantic_cache_size=500,
        semantic_similarity_threshold=0.85,
        hotspot_initial_size=20,
        hotspot_max_size=100,
        enable_prewarming=True,
        prewarm_workers=2
    )
    
    # 创建缓存服务
    cache_service = IntegratedCacheService(
        processing_func=mock_processing_func,
        config=config
    )
    
    await cache_service.start()
    
    try:
        print("🚀 Testing integrated cache service...")
        
        # 测试查询
        test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain deep learning",
            "What is AI?",  # 应该匹配第一个查询
            "Tell me about machine learning",  # 应该匹配第二个查询
            "What is artificial intelligence?",  # 精确匹配
        ]
        
        total_start_time = time.time()
        
        for i, query in enumerate(test_queries):
            print(f"\n--- Query {i+1}: {query} ---")
            
            start_time = time.time()
            response = await cache_service.get_response(
                content=query,
                model="gpt-3.5-turbo",
                parameters={"temperature": 0.7}
            )
            response_time = time.time() - start_time
            
            print(f"Response time: {response_time:.3f}s")
            print(f"Response: {response[:100]}...")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal test time: {total_time:.3f}s")
        
        # 等待预热任务执行
        print("\n⏳ Waiting for prewarming tasks...")
        await asyncio.sleep(3)
        
        # 获取统计信息
        stats = cache_service.get_comprehensive_stats()
        print(f"\n📊 Cache Service Statistics:")
        print(f"Hit Rate: {stats['service_stats']['hit_rate']:.2%}")
        print(f"Performance Improvement: {stats['service_stats']['performance_improvement']:.2%}")
        print(f"Average Response Time: {stats['service_stats']['avg_response_time']:.3f}s")
        print(f"Hotspot Hits: {stats['service_stats']['hotspot_hits']}")
        print(f"Semantic Hits: {stats['service_stats']['semantic_hits']}")
        
        # 健康检查
        health = await cache_service.health_check()
        print(f"\n🏥 Health Check: {health['status']}")
        
        print(f"\n✅ Cache service test completed!")
        
    finally:
        await cache_service.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
