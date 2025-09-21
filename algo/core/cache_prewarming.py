"""
缓存预热系统

智能预热缓存，提升首次访问性能
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import random

logger = logging.getLogger(__name__)


@dataclass
class PrewarmTask:
    """预热任务"""
    key: str
    content: str
    model: str
    parameters: Dict[str, Any]
    priority: int = 0  # 0=normal, 1=high, -1=low
    estimated_time: float = 1.0  # 预估处理时间
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """用于优先级队列排序"""
        return self.priority > other.priority  # 高优先级在前


@dataclass
class PrewarmPattern:
    """预热模式"""
    pattern_id: str
    queries: List[str]
    model: str
    parameters: Dict[str, Any]
    schedule: str  # "daily", "hourly", "on_demand"
    last_execution: float = 0.0
    success_rate: float = 1.0


class QueryPredictor:
    """查询预测器"""
    
    def __init__(self):
        self.query_history: List[str] = []
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.time_patterns: Dict[int, List[str]] = defaultdict(list)  # hour -> queries
        
    def record_query(self, query: str):
        """记录查询"""
        self.query_history.append(query)
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
        
        # 更新查询模式
        normalized_query = self._normalize_query(query)
        self.query_patterns[normalized_query] += 1
        
        # 更新时间模式
        current_hour = time.localtime().tm_hour
        self.time_patterns[current_hour].append(normalized_query)
        
        # 保持时间模式数据量
        if len(self.time_patterns[current_hour]) > 100:
            self.time_patterns[current_hour] = self.time_patterns[current_hour][-100:]
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询"""
        # 简单的标准化：小写、去除多余空格
        return ' '.join(query.lower().split())
    
    def predict_next_queries(self, count: int = 10) -> List[str]:
        """预测下一批可能的查询"""
        current_hour = time.localtime().tm_hour
        
        # 基于时间模式预测
        time_based_queries = []
        if current_hour in self.time_patterns:
            # 获取当前时间段的热门查询
            hour_queries = self.time_patterns[current_hour]
            query_counts = defaultdict(int)
            for query in hour_queries:
                query_counts[query] += 1
            
            # 按频率排序
            sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
            time_based_queries = [query for query, _ in sorted_queries[:count//2]]
        
        # 基于历史频率预测
        frequency_based_queries = []
        sorted_patterns = sorted(self.query_patterns.items(), key=lambda x: x[1], reverse=True)
        frequency_based_queries = [query for query, _ in sorted_patterns[:count//2]]
        
        # 合并预测结果
        predicted_queries = list(set(time_based_queries + frequency_based_queries))
        return predicted_queries[:count]
    
    def get_query_popularity(self, query: str) -> float:
        """获取查询热度"""
        normalized = self._normalize_query(query)
        total_queries = sum(self.query_patterns.values())
        if total_queries == 0:
            return 0.0
        
        return self.query_patterns[normalized] / total_queries


class CachePrewarmer:
    """缓存预热器"""
    
    def __init__(
        self,
        processing_func: Callable[[str, str, Dict[str, Any]], Awaitable[Any]],
        cache_put_func: Callable[[str, str, Any, Dict[str, Any]], Awaitable[None]],
        max_concurrent_tasks: int = 5,
        prewarm_interval: float = 300.0  # 5分钟
    ):
        self.processing_func = processing_func
        self.cache_put_func = cache_put_func
        self.max_concurrent_tasks = max_concurrent_tasks
        self.prewarm_interval = prewarm_interval
        
        self.query_predictor = QueryPredictor()
        self.prewarm_patterns: List[PrewarmPattern] = []
        self.task_queue = asyncio.PriorityQueue()
        
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_prewarm_time': 0.0,
            'cache_hits_from_prewarm': 0
        }
        
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._scheduler_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动预热器"""
        if not self._running:
            self._running = True
            
            # 启动工作线程
            for i in range(self.max_concurrent_tasks):
                task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self._worker_tasks.append(task)
            
            # 启动调度器
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            logger.info(f"Cache prewarmer started with {self.max_concurrent_tasks} workers")
    
    async def stop(self):
        """停止预热器"""
        self._running = False
        
        # 停止工作线程
        for task in self._worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        # 停止调度器
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache prewarmer stopped")
    
    def record_query(self, query: str):
        """记录用户查询"""
        self.query_predictor.record_query(query)
    
    def add_prewarm_pattern(self, pattern: PrewarmPattern):
        """添加预热模式"""
        self.prewarm_patterns.append(pattern)
        logger.info(f"Added prewarm pattern: {pattern.pattern_id}")
    
    async def schedule_prewarm_task(self, task: PrewarmTask):
        """调度预热任务"""
        await self.task_queue.put((task.priority, time.time(), task))
        logger.debug(f"Scheduled prewarm task: {task.key}")
    
    async def prewarm_predicted_queries(self, count: int = 10):
        """预热预测的查询"""
        predicted_queries = self.query_predictor.predict_next_queries(count)
        
        for query in predicted_queries:
            # 计算优先级
            popularity = self.query_predictor.get_query_popularity(query)
            priority = 1 if popularity > 0.1 else 0
            
            task = PrewarmTask(
                key=f"predicted_{hash(query)}",
                content=query,
                model="gpt-3.5-turbo",
                parameters={"temperature": 0.7},
                priority=priority,
                estimated_time=1.0
            )
            
            await self.schedule_prewarm_task(task)
        
        logger.info(f"Scheduled {len(predicted_queries)} predicted queries for prewarming")
    
    async def _worker_loop(self, worker_id: str):
        """工作线程循环"""
        logger.info(f"Prewarmer worker {worker_id} started")
        
        while self._running:
            try:
                # 获取任务 (带超时)
                try:
                    _, _, task = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 执行预热任务
                await self._execute_prewarm_task(task, worker_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Prewarmer worker {worker_id} stopped")
    
    async def _execute_prewarm_task(self, task: PrewarmTask, worker_id: str):
        """执行预热任务"""
        start_time = time.time()
        
        try:
            logger.debug(f"Worker {worker_id} executing prewarm task: {task.key}")
            
            # 调用处理函数
            response = await self.processing_func(
                task.content, 
                task.model, 
                task.parameters
            )
            
            # 存储到缓存
            await self.cache_put_func(
                task.content,
                task.model,
                response,
                task.parameters
            )
            
            # 更新统计
            execution_time = time.time() - start_time
            self.stats['tasks_completed'] += 1
            self.stats['total_prewarm_time'] += execution_time
            
            logger.debug(f"Prewarm task completed: {task.key} in {execution_time:.3f}s")
            
        except Exception as e:
            self.stats['tasks_failed'] += 1
            logger.error(f"Prewarm task failed: {task.key}, error: {e}")
    
    async def _scheduler_loop(self):
        """调度器循环"""
        logger.info("Prewarmer scheduler started")
        
        while self._running:
            try:
                await asyncio.sleep(self.prewarm_interval)
                
                # 执行预定义的预热模式
                await self._execute_scheduled_patterns()
                
                # 预热预测的查询
                await self.prewarm_predicted_queries()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
        
        logger.info("Prewarmer scheduler stopped")
    
    async def _execute_scheduled_patterns(self):
        """执行预定的预热模式"""
        current_time = time.time()
        
        for pattern in self.prewarm_patterns:
            should_execute = False
            
            if pattern.schedule == "hourly":
                should_execute = current_time - pattern.last_execution >= 3600
            elif pattern.schedule == "daily":
                should_execute = current_time - pattern.last_execution >= 86400
            elif pattern.schedule == "on_demand":
                # 基于需求执行，这里简化为随机执行
                should_execute = random.random() < 0.1
            
            if should_execute:
                await self._execute_pattern(pattern)
                pattern.last_execution = current_time
    
    async def _execute_pattern(self, pattern: PrewarmPattern):
        """执行预热模式"""
        logger.info(f"Executing prewarm pattern: {pattern.pattern_id}")
        
        for query in pattern.queries:
            task = PrewarmTask(
                key=f"pattern_{pattern.pattern_id}_{hash(query)}",
                content=query,
                model=pattern.model,
                parameters=pattern.parameters,
                priority=1,  # 预定义模式高优先级
                estimated_time=1.0
            )
            
            await self.schedule_prewarm_task(task)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tasks = self.stats['tasks_completed'] + self.stats['tasks_failed']
        success_rate = self.stats['tasks_completed'] / total_tasks if total_tasks > 0 else 0.0
        
        avg_prewarm_time = (
            self.stats['total_prewarm_time'] / self.stats['tasks_completed'] 
            if self.stats['tasks_completed'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'avg_prewarm_time': avg_prewarm_time,
            'queue_size': self.task_queue.qsize(),
            'active_workers': len(self._worker_tasks),
            'prewarm_patterns_count': len(self.prewarm_patterns),
            'query_patterns_count': len(self.query_predictor.query_patterns)
        }


class SmartPrewarmer:
    """智能预热器"""
    
    def __init__(
        self,
        processing_func: Callable[[str, str, Dict[str, Any]], Awaitable[Any]],
        cache_put_func: Callable[[str, str, Any, Dict[str, Any]], Awaitable[None]],
        cache_get_func: Callable[[str, str, Dict[str, Any]], Awaitable[Optional[Any]]],
        max_concurrent_tasks: int = 3
    ):
        self.cache_get_func = cache_get_func
        
        self.prewarmer = CachePrewarmer(
            processing_func=processing_func,
            cache_put_func=cache_put_func,
            max_concurrent_tasks=max_concurrent_tasks,
            prewarm_interval=180.0  # 3分钟
        )
        
        # 智能预热配置
        self.prewarm_effectiveness: Dict[str, float] = {}  # pattern_id -> effectiveness
        self.cache_hit_tracking: Dict[str, bool] = {}  # key -> was_prewarmed
    
    async def start(self):
        """启动智能预热器"""
        await self.prewarmer.start()
        
        # 添加默认预热模式
        await self._setup_default_patterns()
        
        logger.info("Smart prewarmer started")
    
    async def stop(self):
        """停止智能预热器"""
        await self.prewarmer.stop()
        logger.info("Smart prewarmer stopped")
    
    async def _setup_default_patterns(self):
        """设置默认预热模式"""
        # 常见问题预热
        common_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain deep learning",
            "What are neural networks?",
            "What is natural language processing?"
        ]
        
        common_pattern = PrewarmPattern(
            pattern_id="common_ai_questions",
            queries=common_queries,
            model="gpt-3.5-turbo",
            parameters={"temperature": 0.7, "max_tokens": 200},
            schedule="hourly"
        )
        
        self.prewarmer.add_prewarm_pattern(common_pattern)
        
        # 技术文档预热
        tech_queries = [
            "How to install Python?",
            "What is Docker?",
            "Explain REST API",
            "What is microservices architecture?",
            "How to use Git?"
        ]
        
        tech_pattern = PrewarmPattern(
            pattern_id="tech_documentation",
            queries=tech_queries,
            model="gpt-3.5-turbo",
            parameters={"temperature": 0.5, "max_tokens": 300},
            schedule="daily"
        )
        
        self.prewarmer.add_prewarm_pattern(tech_pattern)
    
    async def record_query_and_response(self, query: str, was_cache_hit: bool):
        """记录查询和响应"""
        self.prewarmer.record_query(query)
        
        # 跟踪预热效果
        query_key = f"query_{hash(query)}"
        if query_key in self.cache_hit_tracking:
            # 这个查询被预热过
            if was_cache_hit:
                # 预热成功
                self._update_prewarm_effectiveness(query, 1.0)
            else:
                # 预热失效 (可能过期了)
                self._update_prewarm_effectiveness(query, 0.5)
    
    def _update_prewarm_effectiveness(self, query: str, effectiveness: float):
        """更新预热效果"""
        # 找到相关的预热模式
        for pattern in self.prewarmer.prewarm_patterns:
            if any(q in query.lower() or query.lower() in q.lower() for q in pattern.queries):
                if pattern.pattern_id not in self.prewarm_effectiveness:
                    self.prewarm_effectiveness[pattern.pattern_id] = effectiveness
                else:
                    # 指数移动平均
                    current = self.prewarm_effectiveness[pattern.pattern_id]
                    self.prewarm_effectiveness[pattern.pattern_id] = 0.8 * current + 0.2 * effectiveness
    
    async def adaptive_prewarm(self):
        """自适应预热"""
        # 基于效果调整预热策略
        for pattern in self.prewarmer.prewarm_patterns:
            effectiveness = self.prewarm_effectiveness.get(pattern.pattern_id, 0.5)
            
            if effectiveness > 0.8:
                # 效果好，增加预热频率
                if pattern.schedule == "daily":
                    pattern.schedule = "hourly"
            elif effectiveness < 0.3:
                # 效果差，减少预热频率
                if pattern.schedule == "hourly":
                    pattern.schedule = "daily"
        
        # 预热预测的查询
        await self.prewarmer.prewarm_predicted_queries(count=15)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.prewarmer.get_stats()
        stats['prewarm_effectiveness'] = self.prewarm_effectiveness
        return stats


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 模拟处理函数
    async def mock_processing_func(content: str, model: str, parameters: Dict[str, Any]) -> str:
        await asyncio.sleep(0.5)  # 模拟处理时间
        return f"Response to: {content}"
    
    # 模拟缓存存储函数
    cache_storage = {}
    
    async def mock_cache_put_func(content: str, model: str, response: Any, parameters: Dict[str, Any]):
        key = f"{content}:{model}:{json.dumps(parameters, sort_keys=True)}"
        cache_storage[key] = response
        print(f"Cached: {key}")
    
    async def mock_cache_get_func(content: str, model: str, parameters: Dict[str, Any]) -> Optional[Any]:
        key = f"{content}:{model}:{json.dumps(parameters, sort_keys=True)}"
        return cache_storage.get(key)
    
    # 创建智能预热器
    prewarmer = SmartPrewarmer(
        processing_func=mock_processing_func,
        cache_put_func=mock_cache_put_func,
        cache_get_func=mock_cache_get_func,
        max_concurrent_tasks=2
    )
    
    await prewarmer.start()
    
    try:
        # 模拟用户查询
        user_queries = [
            "What is AI?",
            "How does ML work?",
            "Explain deep learning",
            "What are neural networks?",
            "Tell me about Python"
        ]
        
        for query in user_queries:
            # 检查缓存
            cached_response = await mock_cache_get_func(query, "gpt-3.5-turbo", {"temperature": 0.7})
            was_cache_hit = cached_response is not None
            
            # 记录查询
            await prewarmer.record_query_and_response(query, was_cache_hit)
            
            print(f"Query: {query}, Cache hit: {was_cache_hit}")
            
            await asyncio.sleep(0.1)
        
        # 等待预热任务执行
        await asyncio.sleep(5)
        
        # 执行自适应预热
        await prewarmer.adaptive_prewarm()
        
        # 等待更多预热任务
        await asyncio.sleep(3)
        
        # 获取统计信息
        stats = prewarmer.get_stats()
        print(f"\nPrewarmer stats: {json.dumps(stats, indent=2)}")
        
        print(f"\nCached items: {len(cache_storage)}")
        for key in list(cache_storage.keys())[:5]:
            print(f"  {key}")
        
    finally:
        await prewarmer.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
