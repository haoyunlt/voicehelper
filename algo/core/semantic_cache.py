"""
语义缓存系统 - v1.2.0
用于缓存相似查询的结果，减少LLM调用和响应延迟
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import redis
from loguru import logger

from core.embeddings import EmbeddingService
from core.config import config


@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    query_hash: str
    embedding: List[float]
    response: str
    references: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: float
    last_hit_at: float
    hit_count: int
    ttl: int  # 秒
    similarity_threshold: float = 0.95


class SemanticCacheManager:
    """语义缓存管理器"""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        default_ttl: int = 86400  # 24小时
    ):
        self.redis = redis_client
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        
        # 缓存键前缀
        self.cache_prefix = "semantic_cache"
        self.embedding_prefix = "embedding_cache"
        self.index_prefix = "cache_index"
        
        # 内存缓存（LRU）
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        
        # 初始化统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0,
            "avg_similarity": 0.0,
            "cache_size": 0
        }
        
        logger.info(f"语义缓存初始化完成，相似度阈值: {similarity_threshold}")
    
    def _generate_hash(self, text: str) -> str:
        """生成文本哈希"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询文本"""
        # 移除多余空格
        query = " ".join(query.split())
        # 转换为小写
        query = query.lower()
        # 移除标点符号（保留必要的）
        import re
        query = re.sub(r'[^\w\s\u4e00-\u9fff？?。.，,！!]', '', query)
        return query.strip()
    
    async def get(
        self,
        query: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """
        获取缓存的响应
        
        Returns:
            (response, references, similarity_score) 或 None
        """
        self.stats["total_queries"] += 1
        
        # 标准化查询
        normalized_query = self._normalize_query(query)
        query_hash = self._generate_hash(f"{tenant_id}:{normalized_query}")
        
        # 1. 检查精确匹配（内存缓存）
        if query_hash in self.memory_cache:
            entry = self.memory_cache[query_hash]
            if self._is_valid(entry):
                self._update_hit_stats(entry, 1.0)
                logger.info(f"内存缓存命中 (精确匹配): {query_hash}")
                return entry.response, entry.references, 1.0
        
        # 2. 检查Redis精确匹配
        redis_key = f"{self.cache_prefix}:{tenant_id}:{query_hash}"
        cached_data = self.redis.get(redis_key)
        
        if cached_data:
            try:
                entry_dict = json.loads(cached_data)
                entry = CacheEntry(**entry_dict)
                
                if self._is_valid(entry):
                    self._update_hit_stats(entry, 1.0)
                    self._add_to_memory_cache(query_hash, entry)
                    logger.info(f"Redis缓存命中 (精确匹配): {query_hash}")
                    return entry.response, entry.references, 1.0
            except Exception as e:
                logger.error(f"解析缓存数据失败: {e}")
        
        # 3. 语义相似度搜索
        try:
            # 获取查询向量
            query_embedding = await self.embedding_service.embed_text(normalized_query)
            
            # 搜索相似缓存
            similar_entry = await self._search_similar(
                query_embedding,
                tenant_id,
                context
            )
            
            if similar_entry:
                response, references, similarity = similar_entry
                self._update_hit_stats(None, similarity)
                logger.info(f"语义缓存命中，相似度: {similarity:.3f}")
                return response, references, similarity
                
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
        
        # 缓存未命中
        self.stats["misses"] += 1
        return None
    
    async def set(
        self,
        query: str,
        response: str,
        references: List[Dict[str, Any]],
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存"""
        try:
            # 标准化查询
            normalized_query = self._normalize_query(query)
            query_hash = self._generate_hash(f"{tenant_id}:{normalized_query}")
            
            # 获取查询向量
            query_embedding = await self.embedding_service.embed_text(normalized_query)
            
            # 创建缓存条目
            entry = CacheEntry(
                query=normalized_query,
                query_hash=query_hash,
                embedding=query_embedding.tolist(),
                response=response,
                references=references,
                metadata=metadata or {},
                created_at=time.time(),
                last_hit_at=time.time(),
                hit_count=0,
                ttl=ttl or self.default_ttl
            )
            
            # 存储到Redis
            redis_key = f"{self.cache_prefix}:{tenant_id}:{query_hash}"
            self.redis.setex(
                redis_key,
                entry.ttl,
                json.dumps(asdict(entry))
            )
            
            # 存储向量索引（用于相似度搜索）
            embedding_key = f"{self.embedding_prefix}:{tenant_id}:{query_hash}"
            self.redis.setex(
                embedding_key,
                entry.ttl,
                json.dumps(entry.embedding)
            )
            
            # 更新索引
            index_key = f"{self.index_prefix}:{tenant_id}"
            self.redis.sadd(index_key, query_hash)
            self.redis.expire(index_key, entry.ttl)
            
            # 添加到内存缓存
            self._add_to_memory_cache(query_hash, entry)
            
            # 检查缓存大小
            self._check_cache_size(tenant_id)
            
            logger.info(f"缓存已设置: {query_hash}")
            return True
            
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    async def _search_similar(
        self,
        query_embedding: np.ndarray,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """搜索相似的缓存条目"""
        try:
            # 获取租户的所有缓存键
            index_key = f"{self.index_prefix}:{tenant_id}"
            cache_keys = self.redis.smembers(index_key)
            
            if not cache_keys:
                return None
            
            best_match = None
            best_similarity = 0.0
            
            # 批量获取向量
            for cache_key in cache_keys:
                cache_key = cache_key.decode() if isinstance(cache_key, bytes) else cache_key
                
                # 获取缓存的向量
                embedding_key = f"{self.embedding_prefix}:{tenant_id}:{cache_key}"
                embedding_data = self.redis.get(embedding_key)
                
                if not embedding_data:
                    continue
                
                cached_embedding = np.array(json.loads(embedding_data))
                
                # 计算余弦相似度
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    cached_embedding.reshape(1, -1)
                )[0][0]
                
                # 检查是否满足阈值
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    # 获取完整缓存数据
                    redis_key = f"{self.cache_prefix}:{tenant_id}:{cache_key}"
                    cached_data = self.redis.get(redis_key)
                    
                    if cached_data:
                        entry_dict = json.loads(cached_data)
                        entry = CacheEntry(**entry_dict)
                        
                        if self._is_valid(entry):
                            best_match = (entry.response, entry.references, similarity)
                            best_similarity = similarity
            
            return best_match
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return None
    
    def _is_valid(self, entry: CacheEntry) -> bool:
        """检查缓存条目是否有效"""
        # 检查TTL
        if time.time() - entry.created_at > entry.ttl:
            return False
        
        # 检查其他业务规则
        # 例如：某些类型的响应不应该被缓存太久
        if entry.metadata.get("volatile", False):
            if time.time() - entry.created_at > 300:  # 5分钟
                return False
        
        return True
    
    def _update_hit_stats(self, entry: Optional[CacheEntry], similarity: float):
        """更新命中统计"""
        self.stats["hits"] += 1
        
        # 更新平均相似度
        total = self.stats["hits"]
        self.stats["avg_similarity"] = (
            (self.stats["avg_similarity"] * (total - 1) + similarity) / total
        )
        
        if entry:
            entry.hit_count += 1
            entry.last_hit_at = time.time()
    
    def _add_to_memory_cache(self, key: str, entry: CacheEntry):
        """添加到内存缓存（LRU）"""
        # 如果已存在，先移除
        if key in self.memory_cache:
            self.access_order.remove(key)
        
        # 添加到最前面
        self.access_order.insert(0, key)
        self.memory_cache[key] = entry
        
        # 限制内存缓存大小
        if len(self.memory_cache) > 100:  # 内存中最多保留100个
            oldest_key = self.access_order.pop()
            del self.memory_cache[oldest_key]
    
    def _check_cache_size(self, tenant_id: str):
        """检查并限制缓存大小"""
        try:
            index_key = f"{self.index_prefix}:{tenant_id}"
            cache_size = self.redis.scard(index_key)
            
            if cache_size > self.max_cache_size:
                # 删除最老的缓存
                cache_keys = list(self.redis.smembers(index_key))
                
                # 获取所有缓存的创建时间
                cache_times = []
                for cache_key in cache_keys[:100]:  # 批量处理前100个
                    cache_key = cache_key.decode() if isinstance(cache_key, bytes) else cache_key
                    redis_key = f"{self.cache_prefix}:{tenant_id}:{cache_key}"
                    cached_data = self.redis.get(redis_key)
                    
                    if cached_data:
                        entry_dict = json.loads(cached_data)
                        cache_times.append((cache_key, entry_dict.get("created_at", 0)))
                
                # 按创建时间排序
                cache_times.sort(key=lambda x: x[1])
                
                # 删除最老的10%
                to_delete = cache_times[:len(cache_times) // 10]
                for cache_key, _ in to_delete:
                    self._delete_cache(tenant_id, cache_key)
                
                logger.info(f"清理了 {len(to_delete)} 个过期缓存")
                
        except Exception as e:
            logger.error(f"检查缓存大小失败: {e}")
    
    def _delete_cache(self, tenant_id: str, cache_key: str):
        """删除缓存"""
        try:
            # 删除缓存数据
            redis_key = f"{self.cache_prefix}:{tenant_id}:{cache_key}"
            self.redis.delete(redis_key)
            
            # 删除向量数据
            embedding_key = f"{self.embedding_prefix}:{tenant_id}:{cache_key}"
            self.redis.delete(embedding_key)
            
            # 从索引中移除
            index_key = f"{self.index_prefix}:{tenant_id}"
            self.redis.srem(index_key, cache_key)
            
            # 从内存缓存中移除
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                    
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
    
    def invalidate(self, tenant_id: str, pattern: Optional[str] = None):
        """使缓存失效"""
        try:
            if pattern:
                # 按模式使失效
                index_key = f"{self.index_prefix}:{tenant_id}"
                cache_keys = self.redis.smembers(index_key)
                
                for cache_key in cache_keys:
                    cache_key = cache_key.decode() if isinstance(cache_key, bytes) else cache_key
                    redis_key = f"{self.cache_prefix}:{tenant_id}:{cache_key}"
                    cached_data = self.redis.get(redis_key)
                    
                    if cached_data:
                        entry_dict = json.loads(cached_data)
                        if pattern in entry_dict.get("query", ""):
                            self._delete_cache(tenant_id, cache_key)
            else:
                # 使所有缓存失效
                index_key = f"{self.index_prefix}:{tenant_id}"
                cache_keys = self.redis.smembers(index_key)
                
                for cache_key in cache_keys:
                    cache_key = cache_key.decode() if isinstance(cache_key, bytes) else cache_key
                    self._delete_cache(tenant_id, cache_key)
                
                # 清空索引
                self.redis.delete(index_key)
                
            logger.info(f"缓存已失效: tenant={tenant_id}, pattern={pattern}")
            
        except Exception as e:
            logger.error(f"使缓存失效失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        hit_rate = (
            self.stats["hits"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0 else 0
        )
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "similarity_threshold": self.similarity_threshold
        }
    
    def warm_up(self, tenant_id: str, queries: List[Tuple[str, str, List[Dict[str, Any]]]]):
        """预热缓存"""
        logger.info(f"开始预热缓存: {len(queries)} 个查询")
        
        for query, response, references in queries:
            try:
                self.set(
                    query=query,
                    response=response,
                    references=references,
                    tenant_id=tenant_id,
                    metadata={"warmed_up": True}
                )
            except Exception as e:
                logger.error(f"预热缓存失败: {e}")
        
        logger.info("缓存预热完成")
