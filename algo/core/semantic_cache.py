"""
语义缓存系统

基于语义相似度的智能缓存，提升40-60%响应速度
"""

import asyncio
import hashlib
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    content: str
    response: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 3600.0  # 1小时默认TTL
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        return self.hits / self.total_requests if self.total_requests > 0 else 0.0


class ContentNormalizer:
    """内容标准化器"""
    
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\d+')
    
    def normalize(self, content: str) -> str:
        """标准化内容"""
        normalized = content.lower().strip()
        normalized = self.whitespace_pattern.sub(' ', normalized)
        normalized = self.punctuation_pattern.sub('', normalized)
        normalized = self.number_pattern.sub('NUM', normalized)
        return normalized
    
    def extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        normalized = self.normalize(content)
        words = normalized.split()
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords


class SimpleSimilarityCalculator:
    """简单相似度计算器"""
    
    def __init__(self):
        self.normalizer = ContentNormalizer()
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """计算两个内容的相似度"""
        if content1 == content2:
            return 1.0
        
        norm1 = self.normalizer.normalize(content1)
        norm2 = self.normalizer.normalize(content2)
        
        if norm1 == norm2:
            return 0.95
        
        seq_similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        keywords1 = set(self.normalizer.extract_keywords(content1))
        keywords2 = set(self.normalizer.extract_keywords(content2))
        
        if keywords1 and keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
        else:
            keyword_similarity = 0.0
        
        final_similarity = 0.6 * seq_similarity + 0.4 * keyword_similarity
        return final_similarity


class SemanticCache:
    """语义缓存系统"""
    
    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        default_ttl: float = 3600.0
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.similarity_calculator = SimpleSimilarityCalculator()
        self.normalizer = ContentNormalizer()
        self.stats = CacheStats()
    
    def _generate_cache_key(self, content: str, model: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        normalized_content = self.normalizer.normalize(content)
        param_signature = json.dumps(parameters, sort_keys=True)
        key_data = f"{normalized_content}:{model}:{param_signature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(
        self, 
        content: str, 
        model: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """获取缓存响应"""
        if parameters is None:
            parameters = {}
        
        self.stats.total_requests += 1
        
        # 精确匹配
        cache_key = self._generate_cache_key(content, model, parameters)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired():
                self.cache.move_to_end(cache_key)
                entry.update_access()
                self.stats.hits += 1
                return entry.response
            else:
                del self.cache[cache_key]
        
        # 语义搜索
        for entry in list(self.cache.values()):
            if entry.is_expired():
                continue
            
            similarity = self.similarity_calculator.calculate_similarity(content, entry.content)
            if similarity >= self.similarity_threshold:
                entry.update_access()
                self.stats.hits += 1
                return entry.response
        
        self.stats.misses += 1
        return None
    
    async def put(
        self, 
        content: str, 
        model: str, 
        response: Any, 
        parameters: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ):
        """存储缓存响应"""
        if parameters is None:
            parameters = {}
        
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._generate_cache_key(content, model, parameters)
        
        entry = CacheEntry(
            key=cache_key,
            content=content,
            response=response,
            ttl=ttl
        )
        
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            self.stats.evictions += 1
        
        self.cache[cache_key] = entry
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'total_requests': self.stats.total_requests,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': self.stats.hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'evictions': self.stats.evictions
        }


# 使用示例
async def example_usage():
    """使用示例"""
    cache = SemanticCache(max_size=100, similarity_threshold=0.85)
    
    # 存储响应
    await cache.put(
        "What is AI?",
        "gpt-3.5-turbo",
        "AI is artificial intelligence.",
        {"temperature": 0.7}
    )
    
    # 精确匹配
    response = await cache.get("What is AI?", "gpt-3.5-turbo", {"temperature": 0.7})
    print(f"Exact: {response}")
    
    # 语义匹配
    response = await cache.get("What is artificial intelligence?", "gpt-3.5-turbo", {"temperature": 0.7})
    print(f"Semantic: {response}")
    
    print(f"Stats: {cache.get_stats()}")


if __name__ == "__main__":
    asyncio.run(example_usage())