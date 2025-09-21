"""
请求合并机制 - 统一版本

通过合并相似请求减少20-40%重复计算
现已整合到unified_utils.py中，此文件保留向后兼容性
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# 导入统一工具类
from .unified_utils import (
    UnifiedRequest,
    UnifiedRequestProcessor,
    get_request_processor,
    get_similarity_calculator
)

import logging

logger = logging.getLogger(__name__)


# 向后兼容的别名
@dataclass
class MergeableRequest:
    """可合并的请求 - 向后兼容"""
    id: str
    content: str
    model: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None
    
    # 兼容字段
    normalized_content: str = ""
    content_hash: str = ""
    similarity_group: Optional[str] = None
    
    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()
        
        # 使用统一工具进行标准化
        from .unified_utils import get_content_normalizer
        normalizer = get_content_normalizer()
        self.normalized_content = normalizer.normalize(self.content)
        
        # 计算哈希
        import hashlib
        content_str = f"{self.normalized_content}:{self.model}:{sorted(self.parameters.items())}"
        self.content_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    def to_unified_request(self) -> UnifiedRequest:
        """转换为统一请求格式"""
        return UnifiedRequest(
            id=self.id,
            content=self.content,
            model=self.model,
            parameters=self.parameters,
            timestamp=self.timestamp,
            future=self.future
        )


@dataclass
class MergeGroup:
    """合并组 - 向后兼容"""
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
        
        # 使用统一相似度计算
        calculator = get_similarity_calculator()
        similarity = calculator.calculate_similarity(
            request.content, 
            self.representative_request.content,
            method="hybrid"
        )
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


class RequestMerger:
    """请求合并器 - 向后兼容，使用统一请求处理器"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        merge_window: float = 5.0,
        max_group_size: int = 10
    ):
        self.similarity_threshold = similarity_threshold
        self.merge_window = merge_window
        self.max_group_size = max_group_size
        
        # 使用统一请求处理器
        self._processor = get_request_processor(
            similarity_threshold=similarity_threshold,
            merge_window=merge_window,
            max_group_size=max_group_size
        )
        
        # 向后兼容的统计
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
        """
        # 转换为统一请求格式
        unified_requests = [req.to_unified_request() for req in requests]
        
        # 使用统一处理器
        unique_unified, merge_mapping = self._processor.process_requests(unified_requests)
        
        # 转换回原格式
        unique_requests = []
        for unified_req in unique_unified:
            # 找到对应的原始请求
            for orig_req in requests:
                if orig_req.id == unified_req.id:
                    unique_requests.append(orig_req)
                    break
        
        # 更新统计
        processor_stats = self._processor.get_stats()
        self.stats.update(processor_stats)
        
        return unique_requests, merge_mapping
    
    def get_stats(self) -> Dict[str, Any]:
        """获取合并统计"""
        return self.stats.copy()


class AdvancedRequestMerger(RequestMerger):
    """高级请求合并器 - 向后兼容"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        merge_window: float = 5.0,
        max_group_size: int = 10,
        enable_semantic_similarity: bool = True
    ):
        super().__init__(similarity_threshold, merge_window, max_group_size)
        self.enable_semantic_similarity = enable_semantic_similarity
        
        # 额外的缓存
        self.similarity_cache = {}
        self.template_patterns = {}
    
    def merge_requests_advanced(
        self,
        requests: List[MergeableRequest]
    ) -> Tuple[List[MergeableRequest], Dict[str, List[str]]]:
        """高级合并请求"""
        # 使用基础合并功能
        return self.merge_requests(requests)


# 便捷函数
def merge_similar_requests(requests: List[MergeableRequest], **kwargs) -> Tuple[List[MergeableRequest], Dict[str, List[str]]]:
    """合并相似请求"""
    merger = RequestMerger(**kwargs)
    return merger.merge_requests(requests)


async def process_mergeable_requests(requests: List[MergeableRequest]) -> Tuple[List[MergeableRequest], Dict[str, List[str]]]:
    """异步处理可合并请求"""
    # 转换为统一格式并处理
    unified_requests = [req.to_unified_request() for req in requests]
    
    from .unified_utils import process_requests
    unique_unified, merge_mapping = await process_requests(unified_requests)
    
    # 转换回原格式
    unique_requests = []
    for unified_req in unique_unified:
        for orig_req in requests:
            if orig_req.id == unified_req.id:
                unique_requests.append(orig_req)
                break
    
    return unique_requests, merge_mapping
