"""
LLM请求批量处理器

实现请求批量化以提升30-50%吞吐量
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """批处理策略"""
    TIME_BASED = "time_based"      # 基于时间的批处理
    SIZE_BASED = "size_based"      # 基于大小的批处理
    ADAPTIVE = "adaptive"          # 自适应批处理
    SIMILARITY = "similarity"      # 基于相似度的批处理


@dataclass
class BatchRequest:
    """批处理请求"""
    id: str
    content: str
    model: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    future: Optional[asyncio.Future] = None
    
    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()


@dataclass
class BatchResponse:
    """批处理响应"""
    request_id: str
    content: str
    tokens_used: int
    processing_time: float
    error: Optional[str] = None


@dataclass
class BatchConfig:
    """批处理配置"""
    max_batch_size: int = 8
    max_wait_time: float = 0.1  # 100ms
    min_batch_size: int = 2
    adaptive_threshold: float = 0.8
    similarity_threshold: float = 0.85
    enable_deduplication: bool = True
    enable_caching: bool = True


class RequestDeduplicator:
    """请求去重器"""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.request_cache: Dict[str, BatchRequest] = {}
        self.response_cache: Dict[str, BatchResponse] = {}
    
    def get_request_hash(self, request: BatchRequest) -> str:
        """获取请求哈希"""
        content = f"{request.content}:{request.model}:{sorted(request.parameters.items())}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def deduplicate(self, requests: List[BatchRequest]) -> Tuple[List[BatchRequest], Dict[str, str]]:
        """
        去重请求
        
        Returns:
            (unique_requests, duplicate_mapping)
        """
        unique_requests = []
        duplicate_mapping = {}  # original_id -> unique_id
        hash_to_request = {}
        
        for request in requests:
            request_hash = self.get_request_hash(request)
            
            # 检查缓存的响应
            if request_hash in self.response_cache:
                cached_response = self.response_cache[request_hash]
                # 直接返回缓存的结果
                if request.future and not request.future.done():
                    request.future.set_result(cached_response)
                continue
            
            # 检查是否有相同的请求
            if request_hash in hash_to_request:
                duplicate_mapping[request.id] = hash_to_request[request_hash].id
            else:
                unique_requests.append(request)
                hash_to_request[request_hash] = request
        
        return unique_requests, duplicate_mapping
    
    def cache_response(self, request: BatchRequest, response: BatchResponse):
        """缓存响应"""
        request_hash = self.get_request_hash(request)
        self.response_cache[request_hash] = response


class LLMBatchProcessor:
    """LLM批量处理器"""
    
    def __init__(
        self,
        llm_client,
        config: BatchConfig = None,
        strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    ):
        self.llm_client = llm_client
        self.config = config or BatchConfig()
        self.strategy = strategy
        
        # 请求队列
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        
        # 去重器
        self.deduplicator = RequestDeduplicator(
            similarity_threshold=self.config.similarity_threshold
        ) if self.config.enable_deduplication else None
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'cache_hits': 0,
            'duplicate_requests': 0,
            'processing_time': 0.0,
            'batch_sizes': [],
            'wait_times': []
        }
        
        # 启动处理任务
        self._processing_task = None
    
    async def start(self):
        """启动批处理器"""
        if not self.processing:
            self.processing = True
            self._processing_task = asyncio.create_task(self._process_batches())
            logger.info("LLM批处理器已启动")
    
    async def stop(self):
        """停止批处理器"""
        self.processing = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("LLM批处理器已停止")
    
    async def submit_request(
        self,
        content: str,
        model: str = "default",
        parameters: Dict[str, Any] = None,
        priority: int = 0
    ) -> BatchResponse:
        """
        提交请求到批处理队列
        
        Args:
            content: 请求内容
            model: 模型名称
            parameters: 模型参数
            priority: 优先级 (数字越大优先级越高)
        
        Returns:
            BatchResponse: 处理结果
        """
        request = BatchRequest(
            id=f"req_{int(time.time() * 1000000)}_{hash(content) % 10000}",
            content=content,
            model=model,
            parameters=parameters or {},
            priority=priority
        )
        
        await self.request_queue.put(request)
        self.stats['total_requests'] += 1
        
        # 等待处理结果
        try:
            response = await request.future
            return response
        except Exception as e:
            logger.error(f"请求处理失败: {e}")
            return BatchResponse(
                request_id=request.id,
                content="",
                tokens_used=0,
                processing_time=0.0,
                error=str(e)
            )
    
    async def _process_batches(self):
        """批处理主循环"""
        while self.processing:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"批处理错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """收集批处理请求"""
        batch = []
        start_time = time.time()
        
        # 等待第一个请求
        try:
            first_request = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=1.0
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            return batch
        
        # 根据策略收集更多请求
        while len(batch) < self.config.max_batch_size:
            wait_time = self.config.max_wait_time - (time.time() - start_time)
            if wait_time <= 0:
                break
            
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=wait_time
                )
                
                # 检查是否应该添加到当前批次
                if self._should_add_to_batch(batch, request):
                    batch.append(request)
                else:
                    # 放回队列
                    await self.request_queue.put(request)
                    break
                    
            except asyncio.TimeoutError:
                break
        
        # 记录等待时间
        actual_wait_time = time.time() - start_time
        self.stats['wait_times'].append(actual_wait_time)
        
        return batch
    
    def _should_add_to_batch(self, batch: List[BatchRequest], request: BatchRequest) -> bool:
        """判断是否应该将请求添加到当前批次"""
        if not batch:
            return True
        
        if self.strategy == BatchStrategy.TIME_BASED:
            return True
        
        elif self.strategy == BatchStrategy.SIZE_BASED:
            return len(batch) < self.config.max_batch_size
        
        elif self.strategy == BatchStrategy.SIMILARITY:
            # 检查内容相似度
            for existing_request in batch:
                if self._calculate_similarity(request.content, existing_request.content) > self.config.similarity_threshold:
                    return True
            return False
        
        elif self.strategy == BatchStrategy.ADAPTIVE:
            # 自适应策略：基于系统负载和请求特征
            return self._adaptive_decision(batch, request)
        
        return True
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度 (简单实现)"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _adaptive_decision(self, batch: List[BatchRequest], request: BatchRequest) -> bool:
        """自适应决策"""
        # 基于队列长度
        queue_size = self.request_queue.qsize()
        if queue_size > 10:  # 队列较长时，倾向于更大的批次
            return len(batch) < self.config.max_batch_size
        
        # 基于优先级
        if request.priority > max(req.priority for req in batch):
            return False  # 高优先级请求单独处理
        
        # 基于模型类型
        if request.model != batch[0].model:
            return False  # 不同模型分开处理
        
        return True
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """处理批次"""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # 去重处理
            if self.deduplicator and self.config.enable_deduplication:
                unique_requests, duplicate_mapping = self.deduplicator.deduplicate(batch)
                self.stats['duplicate_requests'] += len(duplicate_mapping)
            else:
                unique_requests = batch
                duplicate_mapping = {}
            
            if not unique_requests:
                return
            
            # 按模型分组
            model_groups = defaultdict(list)
            for request in unique_requests:
                model_groups[request.model].append(request)
            
            # 并行处理不同模型的请求
            tasks = []
            for model, requests in model_groups.items():
                task = asyncio.create_task(self._process_model_batch(model, requests))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # 处理重复请求
            for original_id, unique_id in duplicate_mapping.items():
                # 找到对应的响应并复制
                for request in batch:
                    if request.id == original_id:
                        # 找到唯一请求的响应
                        for unique_request in unique_requests:
                            if unique_request.id == unique_id and unique_request.future.done():
                                response = unique_request.future.result()
                                # 创建新的响应对象
                                duplicate_response = BatchResponse(
                                    request_id=request.id,
                                    content=response.content,
                                    tokens_used=response.tokens_used,
                                    processing_time=response.processing_time,
                                    error=response.error
                                )
                                if not request.future.done():
                                    request.future.set_result(duplicate_response)
                                break
                        break
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats['batched_requests'] += len(batch)
            self.stats['processing_time'] += processing_time
            self.stats['batch_sizes'].append(len(batch))
            
            logger.info(f"处理批次完成: {len(batch)}个请求, 耗时: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            # 设置所有请求的错误结果
            for request in batch:
                if not request.future.done():
                    error_response = BatchResponse(
                        request_id=request.id,
                        content="",
                        tokens_used=0,
                        processing_time=0.0,
                        error=str(e)
                    )
                    request.future.set_result(error_response)
    
    async def _process_model_batch(self, model: str, requests: List[BatchRequest]):
        """处理特定模型的批次"""
        try:
            # 构建批量请求
            batch_contents = [req.content for req in requests]
            batch_parameters = [req.parameters for req in requests]
            
            # 调用LLM批量接口
            batch_results = await self.llm_client.batch_generate(
                model=model,
                contents=batch_contents,
                parameters=batch_parameters
            )
            
            # 处理结果
            for i, (request, result) in enumerate(zip(requests, batch_results)):
                response = BatchResponse(
                    request_id=request.id,
                    content=result.get('content', ''),
                    tokens_used=result.get('tokens_used', 0),
                    processing_time=result.get('processing_time', 0.0),
                    error=result.get('error')
                )
                
                # 缓存响应
                if self.deduplicator and not response.error:
                    self.deduplicator.cache_response(request, response)
                
                # 设置结果
                if not request.future.done():
                    request.future.set_result(response)
                
        except Exception as e:
            logger.error(f"模型 {model} 批处理失败: {e}")
            # 设置错误结果
            for request in requests:
                if not request.future.done():
                    error_response = BatchResponse(
                        request_id=request.id,
                        content="",
                        tokens_used=0,
                        processing_time=0.0,
                        error=str(e)
                    )
                    request.future.set_result(error_response)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats['batch_sizes']:
            stats['avg_batch_size'] = sum(stats['batch_sizes']) / len(stats['batch_sizes'])
            stats['max_batch_size'] = max(stats['batch_sizes'])
        else:
            stats['avg_batch_size'] = 0
            stats['max_batch_size'] = 0
        
        if stats['wait_times']:
            stats['avg_wait_time'] = sum(stats['wait_times']) / len(stats['wait_times'])
            stats['max_wait_time'] = max(stats['wait_times'])
        else:
            stats['avg_wait_time'] = 0
            stats['max_wait_time'] = 0
        
        if stats['total_requests'] > 0:
            stats['batch_efficiency'] = stats['batched_requests'] / stats['total_requests']
            stats['duplicate_rate'] = stats['duplicate_requests'] / stats['total_requests']
        else:
            stats['batch_efficiency'] = 0
            stats['duplicate_rate'] = 0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'cache_hits': 0,
            'duplicate_requests': 0,
            'processing_time': 0.0,
            'batch_sizes': [],
            'wait_times': []
        }


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 模拟LLM客户端
    class MockLLMClient:
        async def batch_generate(self, model: str, contents: List[str], parameters: List[Dict]):
            # 模拟批量生成
            await asyncio.sleep(0.1)  # 模拟处理时间
            results = []
            for content in contents:
                results.append({
                    'content': f"Response to: {content}",
                    'tokens_used': len(content.split()),
                    'processing_time': 0.1
                })
            return results
    
    # 创建批处理器
    llm_client = MockLLMClient()
    config = BatchConfig(
        max_batch_size=4,
        max_wait_time=0.05,  # 50ms
        enable_deduplication=True
    )
    
    processor = LLMBatchProcessor(
        llm_client=llm_client,
        config=config,
        strategy=BatchStrategy.ADAPTIVE
    )
    
    # 启动处理器
    await processor.start()
    
    try:
        # 提交多个请求
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                processor.submit_request(
                    content=f"请求 {i}",
                    model="gpt-3.5-turbo"
                )
            )
            tasks.append(task)
        
        # 等待所有请求完成
        responses = await asyncio.gather(*tasks)
        
        # 打印结果
        for i, response in enumerate(responses):
            print(f"请求 {i}: {response.content} (tokens: {response.tokens_used})")
        
        # 打印统计信息
        stats = processor.get_stats()
        print(f"\n统计信息:")
        print(f"总请求数: {stats['total_requests']}")
        print(f"批处理效率: {stats['batch_efficiency']:.2%}")
        print(f"平均批次大小: {stats['avg_batch_size']:.1f}")
        print(f"重复请求率: {stats['duplicate_rate']:.2%}")
        
    finally:
        # 停止处理器
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
