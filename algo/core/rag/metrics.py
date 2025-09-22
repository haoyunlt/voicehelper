"""
RAG 指标监控
监控检索延迟、加载时间、命中率等关键指标
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from loguru import logger


@dataclass
class RetrievalMetrics:
    """检索指标"""
    query: str
    retrieval_time_ms: float
    results_count: int
    top_score: float
    avg_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    retriever_type: str = "unknown"
    cache_hit: bool = False


@dataclass
class IndexMetrics:
    """索引指标"""
    load_time_ms: float
    index_size_mb: float
    vector_count: int
    dimension: int
    index_type: str
    timestamp: datetime = field(default_factory=datetime.now)


class RAGMetricsCollector:
    """RAG指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.retrieval_metrics: deque = deque(maxlen=max_history)
        self.index_metrics: List[IndexMetrics] = []
        self.query_cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        
        # 实时统计
        self._current_window_metrics = defaultdict(list)
        self._window_size = timedelta(minutes=5)
    
    def record_retrieval(
        self,
        query: str,
        retrieval_time_ms: float,
        results: List[Dict[str, Any]],
        retriever_type: str = "unknown",
        cache_hit: bool = False
    ):
        """记录检索指标"""
        scores = [r.get("score", 0.0) for r in results]
        
        metrics = RetrievalMetrics(
            query=query,
            retrieval_time_ms=retrieval_time_ms,
            results_count=len(results),
            top_score=max(scores) if scores else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            retriever_type=retriever_type,
            cache_hit=cache_hit
        )
        
        self.retrieval_metrics.append(metrics)
        
        # 更新缓存统计
        self.query_cache_stats["total_queries"] += 1
        if cache_hit:
            self.query_cache_stats["hits"] += 1
        else:
            self.query_cache_stats["misses"] += 1
        
        # 记录到当前窗口
        now = datetime.now()
        self._current_window_metrics[now.replace(second=0, microsecond=0)].append(metrics)
        
        # 清理过期数据
        self._cleanup_window_metrics()
        
        logger.debug(f"记录检索指标: 查询={query[:30]}..., 耗时={retrieval_time_ms:.2f}ms, 结果数={len(results)}")
    
    def record_index_load(
        self,
        load_time_ms: float,
        index_size_mb: float,
        vector_count: int,
        dimension: int,
        index_type: str
    ):
        """记录索引加载指标"""
        metrics = IndexMetrics(
            load_time_ms=load_time_ms,
            index_size_mb=index_size_mb,
            vector_count=vector_count,
            dimension=dimension,
            index_type=index_type
        )
        
        self.index_metrics.append(metrics)
        logger.info(f"记录索引加载指标: 耗时={load_time_ms:.2f}ms, 向量数={vector_count}")
    
    def get_retrieval_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """获取检索统计"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.retrieval_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "avg_results_count": 0.0,
                "avg_score": 0.0,
                "cache_hit_rate": 0.0
            }
        
        latencies = [m.retrieval_time_ms for m in recent_metrics]
        result_counts = [m.results_count for m in recent_metrics]
        scores = [m.avg_score for m in recent_metrics if m.avg_score > 0]
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        
        # 计算P95延迟
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0.0
        
        return {
            "total_queries": len(recent_metrics),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": p95_latency,
            "avg_results_count": sum(result_counts) / len(result_counts),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "cache_hit_rate": cache_hits / len(recent_metrics) if recent_metrics else 0.0,
            "window_minutes": window_minutes
        }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        if not self.index_metrics:
            return {}
        
        latest = self.index_metrics[-1]
        return {
            "latest_load_time_ms": latest.load_time_ms,
            "index_size_mb": latest.index_size_mb,
            "vector_count": latest.vector_count,
            "dimension": latest.dimension,
            "index_type": latest.index_type,
            "last_loaded": latest.timestamp.isoformat()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.query_cache_stats["total_queries"]
        hits = self.query_cache_stats["hits"]
        
        return {
            "total_queries": total,
            "cache_hits": hits,
            "cache_misses": self.query_cache_stats["misses"],
            "hit_rate": hits / total if total > 0 else 0.0
        }
    
    def _cleanup_window_metrics(self):
        """清理窗口指标"""
        cutoff_time = datetime.now() - self._window_size
        keys_to_remove = [
            k for k in self._current_window_metrics.keys() 
            if k < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self._current_window_metrics[key]
    
    def export_metrics(self) -> Dict[str, Any]:
        """导出所有指标"""
        return {
            "retrieval_stats": self.get_retrieval_stats(),
            "index_stats": self.get_index_stats(),
            "cache_stats": self.get_cache_stats(),
            "timestamp": datetime.now().isoformat()
        }


class MetricsTimer:
    """指标计时器上下文管理器"""
    
    def __init__(self, collector: RAGMetricsCollector, operation: str):
        self.collector = collector
        self.operation = operation
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        logger.debug(f"{self.operation} 耗时: {duration_ms:.2f}ms")
        return False
    
    @property
    def duration_ms(self) -> float:
        """获取持续时间（毫秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


# 全局指标收集器实例
_global_metrics_collector = RAGMetricsCollector()


def get_metrics_collector() -> RAGMetricsCollector:
    """获取全局指标收集器"""
    return _global_metrics_collector


def record_retrieval_metrics(
    query: str,
    retrieval_time_ms: float,
    results: List[Dict[str, Any]],
    retriever_type: str = "unknown",
    cache_hit: bool = False
):
    """记录检索指标的便捷函数"""
    _global_metrics_collector.record_retrieval(
        query=query,
        retrieval_time_ms=retrieval_time_ms,
        results=results,
        retriever_type=retriever_type,
        cache_hit=cache_hit
    )


def record_index_load_metrics(
    load_time_ms: float,
    index_size_mb: float,
    vector_count: int,
    dimension: int,
    index_type: str
):
    """记录索引加载指标的便捷函数"""
    _global_metrics_collector.record_index_load(
        load_time_ms=load_time_ms,
        index_size_mb=index_size_mb,
        vector_count=vector_count,
        dimension=dimension,
        index_type=index_type
    )
