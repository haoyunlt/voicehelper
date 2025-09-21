"""
统一工具类 - 合并重复功能
整合相似度计算、内容标准化、请求去重、缓存管理等功能
"""

import asyncio
import hashlib
import json
import time
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque
from difflib import SequenceMatcher
import heapq

logger = logging.getLogger(__name__)


# ==================== 内容标准化 ====================

class UnifiedContentNormalizer:
    """统一内容标准化器"""
    
    def __init__(self):
        # 停用词列表
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be'
        }
    
    def normalize(self, content: str) -> str:
        """标准化内容"""
        if not content:
            return ""
        
        # 1. 转换为小写
        content = content.lower()
        
        # 2. 移除多余空格和换行
        content = re.sub(r'\s+', ' ', content.strip())
        
        # 3. 移除标点符号（保留中文字符）
        content = re.sub(r'[^\w\s\u4e00-\u9fff]', '', content)
        
        # 4. 移除停用词
        words = content.split()
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """提取关键词"""
        normalized = self.normalize(content)
        words = normalized.split()
        
        # 简单的词频统计
        word_freq = {}
        for word in words:
            if len(word) > 1:  # 过滤单字符
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]


# ==================== 相似度计算 ====================

class UnifiedSimilarityCalculator:
    """统一相似度计算器"""
    
    def __init__(self):
        self.normalizer = UnifiedContentNormalizer()
        self.similarity_cache = {}  # 缓存相似度计算结果
    
    def calculate_similarity(self, content1: str, content2: str, method: str = "hybrid") -> float:
        """
        计算两个内容的相似度
        
        Args:
            content1: 内容1
            content2: 内容2
            method: 计算方法 ("sequence", "keyword", "hybrid", "hash")
        
        Returns:
            相似度分数 (0.0 - 1.0)
        """
        if content1 == content2:
            return 1.0
        
        # 检查缓存
        cache_key = (content1, content2) if content1 < content2 else (content2, content1)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        similarity = 0.0
        
        if method == "hash":
            similarity = self._hash_similarity(content1, content2)
        elif method == "sequence":
            similarity = self._sequence_similarity(content1, content2)
        elif method == "keyword":
            similarity = self._keyword_similarity(content1, content2)
        elif method == "hybrid":
            similarity = self._hybrid_similarity(content1, content2)
        
        # 缓存结果
        self.similarity_cache[cache_key] = similarity
        
        # 限制缓存大小
        if len(self.similarity_cache) > 1000:
            oldest_key = next(iter(self.similarity_cache))
            del self.similarity_cache[oldest_key]
        
        return similarity
    
    def _hash_similarity(self, content1: str, content2: str) -> float:
        """基于哈希的精确匹配"""
        norm1 = self.normalizer.normalize(content1)
        norm2 = self.normalizer.normalize(content2)
        return 1.0 if norm1 == norm2 else 0.0
    
    def _sequence_similarity(self, content1: str, content2: str) -> float:
        """基于序列的相似度"""
        norm1 = self.normalizer.normalize(content1)
        norm2 = self.normalizer.normalize(content2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _keyword_similarity(self, content1: str, content2: str) -> float:
        """基于关键词的相似度"""
        keywords1 = set(self.normalizer.extract_keywords(content1))
        keywords2 = set(self.normalizer.extract_keywords(content2))
        
        if not keywords1 and not keywords2:
            return 1.0
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        return len(intersection) / len(union)
    
    def _hybrid_similarity(self, content1: str, content2: str) -> float:
        """混合相似度计算"""
        seq_sim = self._sequence_similarity(content1, content2)
        keyword_sim = self._keyword_similarity(content1, content2)
        
        # 加权组合
        return 0.6 * seq_sim + 0.4 * keyword_sim


# ==================== 请求去重和合并 ====================

@dataclass
class UnifiedRequest:
    """统一请求结构"""
    id: str
    content: str
    model: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None
    
    # 计算字段
    normalized_content: str = ""
    content_hash: str = ""
    similarity_group: Optional[str] = None
    
    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()
        
        normalizer = UnifiedContentNormalizer()
        self.normalized_content = normalizer.normalize(self.content)
        
        # 计算哈希
        content_str = f"{self.normalized_content}:{self.model}:{sorted(self.parameters.items())}"
        self.content_hash = hashlib.md5(content_str.encode()).hexdigest()


class UnifiedRequestProcessor:
    """统一请求处理器 - 合并去重和相似请求合并功能"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        merge_window: float = 5.0,
        max_group_size: int = 10
    ):
        self.similarity_threshold = similarity_threshold
        self.merge_window = merge_window
        self.max_group_size = max_group_size
        
        self.similarity_calculator = UnifiedSimilarityCalculator()
        
        # 缓存和分组
        self.response_cache: Dict[str, Any] = {}
        self.merge_groups: Dict[str, List[UnifiedRequest]] = {}
        self.hash_to_group: Dict[str, str] = {}
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'exact_duplicates': 0,
            'similarity_merges': 0,
            'cache_hits': 0,
            'unique_requests': 0
        }
    
    def process_requests(self, requests: List[UnifiedRequest]) -> Tuple[List[UnifiedRequest], Dict[str, List[str]]]:
        """
        处理请求：去重、合并、缓存
        
        Returns:
            (unique_requests, merge_mapping)
        """
        self.stats['total_requests'] += len(requests)
        
        unique_requests = []
        merge_mapping = defaultdict(list)
        
        # 清理过期分组
        self._cleanup_expired_groups()
        
        for request in requests:
            processed = False
            
            # 1. 检查响应缓存
            if request.content_hash in self.response_cache:
                if request.future and not request.future.done():
                    request.future.set_result(self.response_cache[request.content_hash])
                self.stats['cache_hits'] += 1
                processed = True
                continue
            
            # 2. 检查精确重复
            if request.content_hash in self.hash_to_group:
                group_id = self.hash_to_group[request.content_hash]
                if group_id in self.merge_groups:
                    self.merge_groups[group_id].append(request)
                    merge_mapping[group_id].append(request.id)
                    self.stats['exact_duplicates'] += 1
                    processed = True
            
            # 3. 检查相似请求
            if not processed:
                for group_id, group_requests in self.merge_groups.items():
                    if len(group_requests) >= self.max_group_size:
                        continue
                    
                    representative = group_requests[0]
                    if self._can_merge(request, representative):
                        group_requests.append(request)
                        merge_mapping[group_id].append(request.id)
                        self.stats['similarity_merges'] += 1
                        processed = True
                        break
            
            # 4. 创建新分组
            if not processed:
                group_id = f"group_{int(time.time() * 1000)}_{len(self.merge_groups)}"
                self.merge_groups[group_id] = [request]
                self.hash_to_group[request.content_hash] = group_id
                unique_requests.append(request)
                self.stats['unique_requests'] += 1
        
        return unique_requests, dict(merge_mapping)
    
    def _can_merge(self, request1: UnifiedRequest, request2: UnifiedRequest) -> bool:
        """判断两个请求是否可以合并"""
        # 检查模型
        if request1.model != request2.model:
            return False
        
        # 检查关键参数
        critical_params = ['temperature', 'max_tokens', 'top_p']
        for param in critical_params:
            if request1.parameters.get(param) != request2.parameters.get(param):
                return False
        
        # 检查相似度
        similarity = self.similarity_calculator.calculate_similarity(
            request1.content, request2.content, method="hybrid"
        )
        return similarity >= self.similarity_threshold
    
    def cache_response(self, request: UnifiedRequest, response: Any):
        """缓存响应"""
        self.response_cache[request.content_hash] = response
        
        # 限制缓存大小
        if len(self.response_cache) > 1000:
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
    
    def _cleanup_expired_groups(self):
        """清理过期的合并分组"""
        current_time = time.time()
        expired_groups = []
        
        for group_id, requests in self.merge_groups.items():
            if requests and current_time - requests[0].timestamp > self.merge_window:
                expired_groups.append(group_id)
        
        for group_id in expired_groups:
            requests = self.merge_groups.pop(group_id, [])
            for request in requests:
                if request.content_hash in self.hash_to_group:
                    del self.hash_to_group[request.content_hash]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        total = self.stats['total_requests']
        if total > 0:
            self.stats['deduplication_rate'] = (
                self.stats['exact_duplicates'] + self.stats['similarity_merges']
            ) / total
            self.stats['cache_hit_rate'] = self.stats['cache_hits'] / total
        
        return self.stats.copy()


# ==================== 统一缓存管理 ====================

@dataclass
class UnifiedCacheEntry:
    """统一缓存条目"""
    key: str
    content: str
    response: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: float = 3600.0
    priority: float = 1.0  # 缓存优先级
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        self.accessed_at = time.time()
        self.access_count += 1
        # 更新优先级（访问频率 + 时间衰减）
        time_factor = 1.0 / (1.0 + (time.time() - self.created_at) / 3600.0)
        self.priority = self.access_count * time_factor


class UnifiedCacheManager:
    """统一缓存管理器 - 整合语义缓存、热点缓存、预热缓存"""
    
    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        default_ttl: float = 3600.0,
        enable_prewarming: bool = True
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.enable_prewarming = enable_prewarming
        
        # 核心组件
        self.similarity_calculator = UnifiedSimilarityCalculator()
        self.normalizer = UnifiedContentNormalizer()
        
        # 缓存存储
        self.cache: OrderedDict[str, UnifiedCacheEntry] = OrderedDict()
        self.hotspot_patterns: Dict[str, int] = defaultdict(int)
        self.prewarming_queue: List[Tuple[float, str, Dict[str, Any]]] = []  # (priority, content, params)
        
        # 统计
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'prewarming_hits': 0,
            'total_requests': 0
        }
        
        # 后台任务
        self._prewarming_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动缓存管理器"""
        if not self._running and self.enable_prewarming:
            self._running = True
            self._prewarming_task = asyncio.create_task(self._prewarming_loop())
            logger.info("Unified cache manager started")
    
    async def stop(self):
        """停止缓存管理器"""
        self._running = False
        if self._prewarming_task:
            self._prewarming_task.cancel()
            try:
                await self._prewarming_task
            except asyncio.CancelledError:
                pass
        logger.info("Unified cache manager stopped")
    
    async def get(
        self,
        content: str,
        model: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """获取缓存响应"""
        if parameters is None:
            parameters = {}
        
        self.stats['total_requests'] += 1
        
        # 记录访问模式
        pattern_key = self.normalizer.normalize(content)
        self.hotspot_patterns[pattern_key] += 1
        
        # 生成缓存键
        cache_key = self._generate_cache_key(content, model, parameters)
        
        # 1. 精确匹配
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired():
                self.cache.move_to_end(cache_key)
                entry.update_access()
                self.stats['hits'] += 1
                return entry.response
            else:
                del self.cache[cache_key]
        
        # 2. 语义相似匹配
        for entry in list(self.cache.values()):
            if entry.is_expired():
                continue
            
            similarity = self.similarity_calculator.calculate_similarity(
                content, entry.content, method="hybrid"
            )
            if similarity >= self.similarity_threshold:
                entry.update_access()
                self.stats['hits'] += 1
                return entry.response
        
        self.stats['misses'] += 1
        return None
    
    async def put(
        self,
        content: str,
        response: Any,
        model: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ):
        """存储缓存响应"""
        if parameters is None:
            parameters = {}
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._generate_cache_key(content, model, parameters)
        
        entry = UnifiedCacheEntry(
            key=cache_key,
            content=content,
            response=response,
            ttl=ttl
        )
        
        # 容量管理
        if len(self.cache) >= self.max_size:
            self._evict_least_valuable()
        
        self.cache[cache_key] = entry
        
        # 触发预热
        if self.enable_prewarming:
            await self._schedule_prewarming(content, parameters)
    
    def _generate_cache_key(self, content: str, model: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        normalized_content = self.normalizer.normalize(content)
        param_signature = json.dumps(parameters, sort_keys=True)
        key_data = f"{normalized_content}:{model}:{param_signature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _evict_least_valuable(self):
        """驱逐最低价值的缓存条目"""
        if not self.cache:
            return
        
        # 找到优先级最低的条目
        min_priority = float('inf')
        min_key = None
        
        for key, entry in self.cache.items():
            if entry.priority < min_priority:
                min_priority = entry.priority
                min_key = key
        
        if min_key:
            del self.cache[min_key]
            self.stats['evictions'] += 1
    
    async def _schedule_prewarming(self, content: str, parameters: Dict[str, Any]):
        """调度预热任务"""
        # 基于访问模式计算预热优先级
        pattern_key = self.normalizer.normalize(content)
        access_count = self.hotspot_patterns.get(pattern_key, 0)
        priority = access_count * 1.0  # 简单的优先级计算
        
        heapq.heappush(self.prewarming_queue, (-priority, content, parameters))
        
        # 限制队列大小
        if len(self.prewarming_queue) > 100:
            heapq.heappop(self.prewarming_queue)
    
    async def _prewarming_loop(self):
        """预热循环"""
        while self._running:
            try:
                if self.prewarming_queue:
                    # 处理高优先级的预热任务
                    neg_priority, content, parameters = heapq.heappop(self.prewarming_queue)
                    
                    # 检查是否已经缓存
                    cache_key = self._generate_cache_key(content, "", parameters)
                    if cache_key not in self.cache:
                        # 这里可以调用实际的预热逻辑
                        logger.debug(f"Prewarming cache for: {content[:50]}...")
                
                await asyncio.sleep(1.0)  # 预热间隔
                
            except Exception as e:
                logger.error(f"Prewarming error: {e}")
                await asyncio.sleep(5.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.stats['total_requests']
        if total > 0:
            self.stats['hit_rate'] = self.stats['hits'] / total
            self.stats['miss_rate'] = self.stats['misses'] / total
        
        self.stats['cache_size'] = len(self.cache)
        self.stats['hotspot_patterns'] = len(self.hotspot_patterns)
        
        return self.stats.copy()


# ==================== 导出的统一接口 ====================

# 全局实例
_content_normalizer = None
_similarity_calculator = None
_request_processor = None
_cache_manager = None


def get_content_normalizer() -> UnifiedContentNormalizer:
    """获取内容标准化器单例"""
    global _content_normalizer
    if _content_normalizer is None:
        _content_normalizer = UnifiedContentNormalizer()
    return _content_normalizer


def get_similarity_calculator() -> UnifiedSimilarityCalculator:
    """获取相似度计算器单例"""
    global _similarity_calculator
    if _similarity_calculator is None:
        _similarity_calculator = UnifiedSimilarityCalculator()
    return _similarity_calculator


def get_request_processor(**kwargs) -> UnifiedRequestProcessor:
    """获取请求处理器单例"""
    global _request_processor
    if _request_processor is None:
        _request_processor = UnifiedRequestProcessor(**kwargs)
    return _request_processor


def get_cache_manager(**kwargs) -> UnifiedCacheManager:
    """获取缓存管理器单例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = UnifiedCacheManager(**kwargs)
    return _cache_manager


# 便捷函数
def normalize_content(content: str) -> str:
    """标准化内容"""
    return get_content_normalizer().normalize(content)


def calculate_similarity(content1: str, content2: str, method: str = "hybrid") -> float:
    """计算相似度"""
    return get_similarity_calculator().calculate_similarity(content1, content2, method)


async def process_requests(requests: List[UnifiedRequest]) -> Tuple[List[UnifiedRequest], Dict[str, List[str]]]:
    """处理请求"""
    return get_request_processor().process_requests(requests)


async def get_cached_response(content: str, model: str = "", parameters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """获取缓存响应"""
    return await get_cache_manager().get(content, model, parameters)


async def cache_response(content: str, response: Any, model: str = "", parameters: Optional[Dict[str, Any]] = None):
    """缓存响应"""
    await get_cache_manager().put(content, response, model, parameters)
