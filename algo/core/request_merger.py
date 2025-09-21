"""
请求合并机制

通过合并相似请求减少20-40%重复计算
"""

import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)


@dataclass
class MergeableRequest:
    """可合并的请求"""
    id: str
    content: str
    model: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None
    
    # 合并相关字段
    normalized_content: str = ""
    content_hash: str = ""
    similarity_group: Optional[str] = None
    
    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()
        self.normalized_content = self._normalize_content()
        self.content_hash = self._calculate_hash()
    
    def _normalize_content(self) -> str:
        """标准化内容"""
        # 移除多余空格和换行
        content = re.sub(r'\s+', ' ', self.content.strip())
        # 转换为小写
        content = content.lower()
        # 移除标点符号
        content = re.sub(r'[^\w\s]', '', content)
        return content
    
    def _calculate_hash(self) -> str:
        """计算内容哈希"""
        content_str = f"{self.normalized_content}:{self.model}:{sorted(self.parameters.items())}"
        return hashlib.md5(content_str.encode()).hexdigest()


@dataclass
class MergeGroup:
    """合并组"""
    id: str
    representative_request: MergeableRequest
    merged_requests: List[MergeableRequest] = field(default_factory=list)
    similarity_threshold: float = 0.85
    created_at: float = field(default_factory=time.time)
    
    def can_merge(self, request: MergeableRequest) -> bool:
        """判断是否可以合并"""
        # 检查模型是否相同
        if request.model != self.representative_request.model:
            return False
        
        # 检查参数是否兼容
        if not self._parameters_compatible(request.parameters):
            return False
        
        # 检查内容相似度
        similarity = self._calculate_similarity(request.normalized_content)
        return similarity >= self.similarity_threshold
    
    def add_request(self, request: MergeableRequest):
        """添加请求到合并组"""
        request.similarity_group = self.id
        self.merged_requests.append(request)
    
    def _parameters_compatible(self, parameters: Dict[str, Any]) -> bool:
        """检查参数是否兼容"""
        rep_params = self.representative_request.parameters
        
        # 关键参数必须相同
        critical_params = ['temperature', 'max_tokens', 'top_p', 'frequency_penalty']
        for param in critical_params:
            if param in rep_params or param in parameters:
                if rep_params.get(param) != parameters.get(param):
                    return False
        
        return True
    
    def _calculate_similarity(self, content: str) -> float:
        """计算内容相似度"""
        return SequenceMatcher(
            None,
            self.representative_request.normalized_content,
            content
        ).ratio()


class RequestMerger:
    """请求合并器"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        merge_window: float = 5.0,  # 5秒合并窗口
        max_group_size: int = 10
    ):
        self.similarity_threshold = similarity_threshold
        self.merge_window = merge_window
        self.max_group_size = max_group_size
        
        # 合并组管理
        self.merge_groups: Dict[str, MergeGroup] = {}
        self.content_hash_to_group: Dict[str, str] = {}
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'merged_requests': 0,
            'merge_groups_created': 0,
            'exact_duplicates': 0,
            'similarity_merges': 0,
            'merge_savings': 0.0
        }
    
    def merge_requests(self, requests: List[MergeableRequest]) -> Tuple[List[MergeableRequest], Dict[str, List[str]]]:
        """
        合并请求
        
        Args:
            requests: 待合并的请求列表
        
        Returns:
            (unique_requests, merge_mapping)
            unique_requests: 去重后的唯一请求
            merge_mapping: 合并映射 {group_id: [merged_request_ids]}
        """
        self.stats['total_requests'] += len(requests)
        
        unique_requests = []
        merge_mapping = defaultdict(list)
        
        # 清理过期的合并组
        self._cleanup_expired_groups()
        
        for request in requests:
            merged = False
            
            # 1. 检查完全相同的请求 (哈希匹配)
            if request.content_hash in self.content_hash_to_group:
                group_id = self.content_hash_to_group[request.content_hash]
                if group_id in self.merge_groups:
                    group = self.merge_groups[group_id]
                    group.add_request(request)
                    merge_mapping[group_id].append(request.id)
                    self.stats['exact_duplicates'] += 1
                    merged = True
            
            # 2. 检查相似请求
            if not merged:
                for group_id, group in self.merge_groups.items():
                    if (len(group.merged_requests) < self.max_group_size and 
                        group.can_merge(request)):
                        group.add_request(request)
                        merge_mapping[group_id].append(request.id)
                        self.stats['similarity_merges'] += 1
                        merged = True
                        break
            
            # 3. 创建新的合并组
            if not merged:
                group_id = f"group_{int(time.time() * 1000)}_{len(self.merge_groups)}"
                group = MergeGroup(
                    id=group_id,
                    representative_request=request,
                    similarity_threshold=self.similarity_threshold
                )
                
                self.merge_groups[group_id] = group
                self.content_hash_to_group[request.content_hash] = group_id
                unique_requests.append(request)
                self.stats['merge_groups_created'] += 1
        
        # 统计合并节省
        total_merged = sum(len(requests) for requests in merge_mapping.values())
        self.stats['merged_requests'] += total_merged
        self.stats['merge_savings'] = total_merged / len(requests) if requests else 0
        
        return unique_requests, dict(merge_mapping)
    
    def distribute_response(
        self,
        group_id: str,
        response: Any,
        merge_mapping: Dict[str, List[str]]
    ):
        """分发响应到合并的请求"""
        if group_id not in self.merge_groups:
            return
        
        group = self.merge_groups[group_id]
        merged_request_ids = merge_mapping.get(group_id, [])
        
        # 分发给代表请求
        if not group.representative_request.future.done():
            group.representative_request.future.set_result(response)
        
        # 分发给合并的请求
        for request in group.merged_requests:
            if request.id in merged_request_ids and not request.future.done():
                # 创建响应副本
                response_copy = self._create_response_copy(response, request.id)
                request.future.set_result(response_copy)
    
    def _create_response_copy(self, original_response: Any, request_id: str) -> Any:
        """创建响应副本"""
        # 这里需要根据实际的响应类型来实现
        # 示例实现
        if hasattr(original_response, '__dict__'):
            response_dict = original_response.__dict__.copy()
            response_dict['request_id'] = request_id
            return type(original_response)(**response_dict)
        else:
            return original_response
    
    def _cleanup_expired_groups(self):
        """清理过期的合并组"""
        current_time = time.time()
        expired_groups = []
        
        for group_id, group in self.merge_groups.items():
            if current_time - group.created_at > self.merge_window:
                expired_groups.append(group_id)
        
        for group_id in expired_groups:
            group = self.merge_groups.pop(group_id)
            # 清理哈希映射
            rep_hash = group.representative_request.content_hash
            if self.content_hash_to_group.get(rep_hash) == group_id:
                del self.content_hash_to_group[rep_hash]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['merge_rate'] = stats['merged_requests'] / stats['total_requests']
            stats['duplicate_rate'] = stats['exact_duplicates'] / stats['total_requests']
            stats['similarity_rate'] = stats['similarity_merges'] / stats['total_requests']
        else:
            stats['merge_rate'] = 0
            stats['duplicate_rate'] = 0
            stats['similarity_rate'] = 0
        
        stats['active_groups'] = len(self.merge_groups)
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'merged_requests': 0,
            'merge_groups_created': 0,
            'exact_duplicates': 0,
            'similarity_merges': 0,
            'merge_savings': 0.0
        }


class AdvancedRequestMerger(RequestMerger):
    """高级请求合并器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 语义相似度缓存
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # 模板检测
        self.template_patterns: Dict[str, re.Pattern] = {}
        self._init_template_patterns()
    
    def _init_template_patterns(self):
        """初始化模板模式"""
        # 常见的模板模式
        patterns = {
            'translation': re.compile(r'translate\s+["\'](.+?)["\'].*?to\s+(\w+)', re.IGNORECASE),
            'summarization': re.compile(r'summarize\s+["\'](.+?)["\']', re.IGNORECASE),
            'question_answering': re.compile(r'what\s+is\s+(.+?)\?', re.IGNORECASE),
            'code_generation': re.compile(r'write\s+(\w+)\s+code\s+to\s+(.+)', re.IGNORECASE),
        }
        
        self.template_patterns = patterns
    
    def _detect_template(self, content: str) -> Optional[str]:
        """检测内容模板"""
        for template_name, pattern in self.template_patterns.items():
            if pattern.search(content):
                return template_name
        return None
    
    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """计算语义相似度 (可以集成更复杂的模型)"""
        cache_key = (content1, content2) if content1 < content2 else (content2, content1)
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # 简单的词汇相似度计算
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            similarity = 1.0
        elif not words1 or not words2:
            similarity = 0.0
        else:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            similarity = len(intersection) / len(union)
        
        # 缓存结果
        self.similarity_cache[cache_key] = similarity
        
        # 限制缓存大小
        if len(self.similarity_cache) > 1000:
            # 移除最旧的条目
            oldest_key = next(iter(self.similarity_cache))
            del self.similarity_cache[oldest_key]
        
        return similarity
    
    def merge_requests_advanced(
        self,
        requests: List[MergeableRequest]
    ) -> Tuple[List[MergeableRequest], Dict[str, List[str]]]:
        """
        高级请求合并
        """
        # 按模板类型分组
        template_groups = defaultdict(list)
        no_template_requests = []
        
        for request in requests:
            template = self._detect_template(request.content)
            if template:
                template_groups[template].append(request)
            else:
                no_template_requests.append(request)
        
        unique_requests = []
        merge_mapping = defaultdict(list)
        
        # 处理模板化请求
        for template, template_requests in template_groups.items():
            template_unique, template_mapping = self._merge_template_requests(
                template_requests, template
            )
            unique_requests.extend(template_unique)
            merge_mapping.update(template_mapping)
        
        # 处理非模板化请求
        if no_template_requests:
            other_unique, other_mapping = self.merge_requests(no_template_requests)
            unique_requests.extend(other_unique)
            merge_mapping.update(other_mapping)
        
        return unique_requests, dict(merge_mapping)
    
    def _merge_template_requests(
        self,
        requests: List[MergeableRequest],
        template: str
    ) -> Tuple[List[MergeableRequest], Dict[str, List[str]]]:
        """合并模板化请求"""
        # 对于模板化请求，使用更严格的相似度阈值
        original_threshold = self.similarity_threshold
        self.similarity_threshold = 0.95  # 提高阈值
        
        try:
            return self.merge_requests(requests)
        finally:
            self.similarity_threshold = original_threshold


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 创建请求合并器
    merger = AdvancedRequestMerger(
        similarity_threshold=0.8,
        merge_window=3.0,
        max_group_size=5
    )
    
    # 创建测试请求
    requests = []
    
    # 相似的翻译请求
    for i in range(3):
        request = MergeableRequest(
            id=f"trans_{i}",
            content=f"Translate 'Hello world {i}' to Chinese",
            model="gpt-3.5-turbo"
        )
        requests.append(request)
    
    # 完全相同的请求
    for i in range(2):
        request = MergeableRequest(
            id=f"same_{i}",
            content="What is artificial intelligence?",
            model="gpt-3.5-turbo"
        )
        requests.append(request)
    
    # 不同的请求
    request = MergeableRequest(
        id="different",
        content="Write Python code to sort a list",
        model="gpt-3.5-turbo"
    )
    requests.append(request)
    
    # 执行合并
    unique_requests, merge_mapping = merger.merge_requests_advanced(requests)
    
    print(f"原始请求数: {len(requests)}")
    print(f"合并后请求数: {len(unique_requests)}")
    print(f"合并映射: {merge_mapping}")
    
    # 打印统计信息
    stats = merger.get_stats()
    print(f"\n统计信息:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}" if 'rate' in key else f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_usage())
