"""
语义缓存系统 - 统一版本

基于语义相似度的智能缓存，提升40-60%响应速度
现已整合到unified_utils.py中，此文件保留向后兼容性
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time

# 导入统一工具类
from .unified_utils import (
    UnifiedCacheManager, 
    UnifiedCacheEntry,
    get_cache_manager,
    get_content_normalizer,
    get_similarity_calculator
)

logger = logging.getLogger(__name__)


# 向后兼容的别名
@dataclass
class CacheEntry:
    """缓存条目 - 向后兼容"""
    key: str
    content: str
    response: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 3600.0
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计 - 向后兼容"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


# 向后兼容的类
class ContentNormalizer:
    """内容标准化器 - 向后兼容，使用统一实现"""
    
    def __init__(self):
        self._normalizer = get_content_normalizer()
    
    def normalize(self, content: str) -> str:
        return self._normalizer.normalize(content)
    
    def extract_keywords(self, content: str) -> List[str]:
        return self._normalizer.extract_keywords(content)


class SimpleSimilarityCalculator:
    """相似度计算器 - 向后兼容，使用统一实现"""
    
    def __init__(self):
        self._calculator = get_similarity_calculator()
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        return self._calculator.calculate_similarity(content1, content2, method="hybrid")


class SemanticCache:
    """语义缓存系统 - 向后兼容，使用统一缓存管理器"""
    
    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        default_ttl: float = 3600.0
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        
        # 使用统一缓存管理器
        self._cache_manager = get_cache_manager(
            max_size=max_size,
            similarity_threshold=similarity_threshold,
            default_ttl=default_ttl
        )
        
        # 向后兼容的统计
        self.stats = CacheStats()
    
    async def get(
        self, 
        content: str, 
        model: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """获取缓存响应"""
        result = await self._cache_manager.get(content, model, parameters)
        
        # 更新向后兼容的统计
        if result is not None:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        self.stats.total_requests += 1
        
        return result
    
    async def put(
        self, 
        content: str, 
        model: str, 
        response: Any, 
        parameters: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ):
        """存储缓存响应"""
        await self._cache_manager.put(content, response, model, parameters, ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        # 合并统一缓存管理器的统计和向后兼容统计
        unified_stats = self._cache_manager.get_stats()
        
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'evictions': unified_stats.get('evictions', 0),
            'total_requests': self.stats.total_requests,
            'hit_rate': self.stats.hit_rate,
            'cache_size': unified_stats.get('cache_size', 0)
        }
    
    async def start(self):
        """启动缓存"""
        await self._cache_manager.start()
    
    async def stop(self):
        """停止缓存"""
        await self._cache_manager.stop()


# 便捷函数
async def get_semantic_cache_response(content: str, model: str = "", parameters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """获取语义缓存响应"""
    cache_manager = get_cache_manager()
    return await cache_manager.get(content, model, parameters)


async def cache_semantic_response(content: str, response: Any, model: str = "", parameters: Optional[Dict[str, Any]] = None):
    """缓存语义响应"""
    cache_manager = get_cache_manager()
    await cache_manager.put(content, response, model, parameters)
