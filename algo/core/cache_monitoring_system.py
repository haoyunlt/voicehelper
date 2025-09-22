"""
VoiceHelper v1.20.1 - 缓存监控系统
完善缓存命中率统计和监控指标
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheHit:
    """缓存命中记录"""
    cache_key: str
    hit_time: float
    response_time: float
    cache_size: int
    user_id: str
    request_type: str

@dataclass
class CacheMiss:
    """缓存未命中记录"""
    cache_key: str
    miss_time: float
    response_time: float
    cache_size: int
    user_id: str
    request_type: str
    reason: str  # "not_found", "expired", "invalid"

@dataclass
class CacheMetrics:
    """缓存指标"""
    total_requests: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_hit_response_time: float = 0.0
    avg_miss_response_time: float = 0.0
    cache_size: int = 0
    memory_usage: float = 0.0
    eviction_count: int = 0

class CacheMonitor:
    """缓存监控器"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.hit_history = deque(maxlen=window_size)
        self.miss_history = deque(maxlen=window_size)
        
        # 实时指标
        self.metrics = CacheMetrics()
        
        # 历史数据
        self.hourly_metrics = defaultdict(list)
        self.daily_metrics = defaultdict(list)
        
        # 用户统计
        self.user_stats = defaultdict(lambda: {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        })
        
        # 请求类型统计
        self.type_stats = defaultdict(lambda: {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        })
    
    def record_hit(self, cache_key: str, response_time: float, cache_size: int, 
                   user_id: str, request_type: str):
        """记录缓存命中"""
        hit = CacheHit(
            cache_key=cache_key,
            hit_time=time.time(),
            response_time=response_time,
            cache_size=cache_size,
            user_id=user_id,
            request_type=request_type
        )
        
        self.hit_history.append(hit)
        self._update_metrics()
        self._update_user_stats(user_id, "hit")
        self._update_type_stats(request_type, "hit")
        
        logger.debug(f"Cache hit: {cache_key} (user: {user_id}, type: {request_type})")
    
    def record_miss(self, cache_key: str, response_time: float, cache_size: int,
                    user_id: str, request_type: str, reason: str):
        """记录缓存未命中"""
        miss = CacheMiss(
            cache_key=cache_key,
            miss_time=time.time(),
            response_time=response_time,
            cache_size=cache_size,
            user_id=user_id,
            request_type=request_type,
            reason=reason
        )
        
        self.miss_history.append(miss)
        self._update_metrics()
        self._update_user_stats(user_id, "miss")
        self._update_type_stats(request_type, "miss")
        
        logger.debug(f"Cache miss: {cache_key} (user: {user_id}, type: {request_type}, reason: {reason})")
    
    def _update_metrics(self):
        """更新指标"""
        total_requests = len(self.hit_history) + len(self.miss_history)
        total_hits = len(self.hit_history)
        total_misses = len(self.miss_history)
        
        self.metrics.total_requests = total_requests
        self.metrics.total_hits = total_hits
        self.metrics.total_misses = total_misses
        
        if total_requests > 0:
            self.metrics.hit_rate = total_hits / total_requests
            self.metrics.miss_rate = total_misses / total_requests
        
        # 计算平均响应时间
        if total_hits > 0:
            hit_times = [hit.response_time for hit in self.hit_history]
            self.metrics.avg_hit_response_time = sum(hit_times) / len(hit_times)
        
        if total_misses > 0:
            miss_times = [miss.response_time for miss in self.miss_history]
            self.metrics.avg_miss_response_time = sum(miss_times) / len(miss_times)
    
    def _update_user_stats(self, user_id: str, event_type: str):
        """更新用户统计"""
        if event_type == "hit":
            self.user_stats[user_id]["hits"] += 1
        else:
            self.user_stats[user_id]["misses"] += 1
        
        self.user_stats[user_id]["total_requests"] += 1
    
    def _update_type_stats(self, request_type: str, event_type: str):
        """更新请求类型统计"""
        if event_type == "hit":
            self.type_stats[request_type]["hits"] += 1
        else:
            self.type_stats[request_type]["misses"] += 1
        
        self.type_stats[request_type]["total_requests"] += 1
    
    def get_metrics(self) -> CacheMetrics:
        """获取当前指标"""
        return self.metrics
    
    def get_hourly_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取小时级指标"""
        now = datetime.now()
        hourly_data = []
        
        for i in range(hours):
            hour_start = now - timedelta(hours=i+1)
            hour_end = now - timedelta(hours=i)
            
            # 过滤该小时的数据
            hour_hits = [
                hit for hit in self.hit_history
                if hour_start.timestamp() <= hit.hit_time <= hour_end.timestamp()
            ]
            hour_misses = [
                miss for miss in self.miss_history
                if hour_start.timestamp() <= miss.miss_time <= hour_end.timestamp()
            ]
            
            total_requests = len(hour_hits) + len(hour_misses)
            hit_rate = len(hour_hits) / total_requests if total_requests > 0 else 0
            
            hourly_data.append({
                "hour": hour_start.strftime("%Y-%m-%d %H:00"),
                "total_requests": total_requests,
                "hits": len(hour_hits),
                "misses": len(hour_misses),
                "hit_rate": hit_rate,
                "avg_response_time": self._calculate_avg_response_time(hour_hits, hour_misses)
            })
        
        return hourly_data
    
    def get_user_metrics(self, user_id: str) -> Dict[str, Any]:
        """获取用户指标"""
        if user_id not in self.user_stats:
            return {
                "user_id": user_id,
                "total_requests": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0
            }
        
        stats = self.user_stats[user_id]
        hit_rate = stats["hits"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
        
        return {
            "user_id": user_id,
            "total_requests": stats["total_requests"],
            "hits": stats["hits"],
            "misses": stats["misses"],
            "hit_rate": hit_rate
        }
    
    def get_type_metrics(self, request_type: str) -> Dict[str, Any]:
        """获取请求类型指标"""
        if request_type not in self.type_stats:
            return {
                "request_type": request_type,
                "total_requests": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0
            }
        
        stats = self.type_stats[request_type]
        hit_rate = stats["hits"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
        
        return {
            "request_type": request_type,
            "total_requests": stats["total_requests"],
            "hits": stats["hits"],
            "misses": stats["misses"],
            "hit_rate": hit_rate
        }
    
    def _calculate_avg_response_time(self, hits: List[CacheHit], misses: List[CacheMiss]) -> float:
        """计算平均响应时间"""
        all_times = [hit.response_time for hit in hits] + [miss.response_time for miss in misses]
        return sum(all_times) / len(all_times) if all_times else 0.0
    
    def get_top_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取活跃用户排行"""
        user_list = []
        for user_id, stats in self.user_stats.items():
            if stats["total_requests"] > 0:
                hit_rate = stats["hits"] / stats["total_requests"]
                user_list.append({
                    "user_id": user_id,
                    "total_requests": stats["total_requests"],
                    "hit_rate": hit_rate
                })
        
        return sorted(user_list, key=lambda x: x["total_requests"], reverse=True)[:limit]
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """获取缓存效率报告"""
        return {
            "overall_metrics": {
                "total_requests": self.metrics.total_requests,
                "hit_rate": self.metrics.hit_rate,
                "miss_rate": self.metrics.miss_rate,
                "avg_hit_response_time": self.metrics.avg_hit_response_time,
                "avg_miss_response_time": self.metrics.avg_miss_response_time
            },
            "top_users": self.get_top_users(5),
            "type_breakdown": {
                request_type: self.get_type_metrics(request_type)
                for request_type in self.type_stats.keys()
            },
            "hourly_trend": self.get_hourly_metrics(24)
        }

class EnhancedCachePredictor:
    """增强的缓存预测器"""
    
    def __init__(self, monitor: CacheMonitor):
        self.monitor = monitor
        self.prediction_accuracy = 0.0
        self.prediction_history = deque(maxlen=1000)
        
    async def predict_and_cache(self, user_id: str, context: str, request_type: str = "text_generation"):
        """预测用户需求并预缓存"""
        try:
            # 分析用户历史模式
            user_metrics = self.monitor.get_user_metrics(user_id)
            
            # 基于上下文的预测
            predicted_queries = await self._predict_next_queries(context, user_metrics)
            
            # 预缓存热门响应
            cache_hits = 0
            for query in predicted_queries:
                cache_key = f"{user_id}:{request_type}:{hash(query)}"
                
                # 检查是否已缓存
                if await self._check_cache_exists(cache_key):
                    cache_hits += 1
                    self.monitor.record_hit(
                        cache_key=cache_key,
                        response_time=0.001,  # 缓存命中响应时间
                        cache_size=len(query),
                        user_id=user_id,
                        request_type=request_type
                    )
                else:
                    # 预缓存
                    await self._pre_cache_response(cache_key, query)
            
            # 记录预测准确率
            prediction_accuracy = cache_hits / len(predicted_queries) if predicted_queries else 0
            self.prediction_history.append(prediction_accuracy)
            self.prediction_accuracy = sum(self.prediction_history) / len(self.prediction_history)
            
            logger.info(f"Cache prediction for user {user_id}: {cache_hits}/{len(predicted_queries)} hits")
            
        except Exception as e:
            logger.error(f"Cache prediction error: {e}")
    
    async def _predict_next_queries(self, context: str, user_metrics: Dict[str, Any]) -> List[str]:
        """预测下一个查询"""
        # 简化的预测逻辑
        predictions = []
        
        # 基于上下文的预测
        if "天气" in context:
            predictions.extend(["今天天气怎么样", "明天会下雨吗", "温度如何"])
        
        if "时间" in context:
            predictions.extend(["现在几点了", "今天星期几", "明天是什么日子"])
        
        if "帮助" in context:
            predictions.extend(["如何使用", "有什么功能", "常见问题"])
        
        # 基于用户历史的预测
        if user_metrics["total_requests"] > 10:
            # 活跃用户，增加更多预测
            predictions.extend(["继续", "更多信息", "详细说明"])
        
        return predictions[:5]  # 最多5个预测
    
    async def _check_cache_exists(self, cache_key: str) -> bool:
        """检查缓存是否存在"""
        # 模拟缓存检查
        return hash(cache_key) % 3 == 0  # 30%的缓存命中率
    
    async def _pre_cache_response(self, cache_key: str, query: str):
        """预缓存响应"""
        # 模拟预缓存
        await asyncio.sleep(0.001)
        logger.debug(f"Pre-cached: {cache_key}")

# 全局监控实例
cache_monitor = CacheMonitor()
enhanced_cache_predictor = EnhancedCachePredictor(cache_monitor)

async def get_cache_metrics() -> Dict[str, Any]:
    """获取缓存指标"""
    return cache_monitor.get_cache_efficiency_report()

async def record_cache_hit(cache_key: str, response_time: float, cache_size: int, 
                          user_id: str, request_type: str):
    """记录缓存命中"""
    cache_monitor.record_hit(cache_key, response_time, cache_size, user_id, request_type)

async def record_cache_miss(cache_key: str, response_time: float, cache_size: int,
                           user_id: str, request_type: str, reason: str):
    """记录缓存未命中"""
    cache_monitor.record_miss(cache_key, response_time, cache_size, user_id, request_type, reason)

if __name__ == "__main__":
    # 测试代码
    async def test_cache_monitoring():
        # 模拟一些缓存操作
        await record_cache_hit("user1:text:123", 0.01, 100, "user1", "text_generation")
        await record_cache_miss("user1:text:456", 0.05, 100, "user1", "text_generation", "not_found")
        await record_cache_hit("user2:voice:789", 0.02, 200, "user2", "voice_synthesis")
        
        # 获取指标
        metrics = await get_cache_metrics()
        print("缓存指标:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    asyncio.run(test_cache_monitoring())
