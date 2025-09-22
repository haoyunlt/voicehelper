"""
用户行为和对话质量分析系统
提供实时分析、趋势预测和智能洞察
基于GitHub开源项目的最佳实践
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import sqlite3
import aioredis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class EventType(Enum):
    USER_MESSAGE = "user_message"
    BOT_RESPONSE = "bot_response"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    EMOTION_DETECTED = "emotion_detected"
    INTENT_RECOGNIZED = "intent_recognized"
    ERROR_OCCURRED = "error_occurred"
    FEEDBACK_RECEIVED = "feedback_received"
    MODEL_SWITCHED = "model_switched"

class MetricType(Enum):
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    CONVERSATION_LENGTH = "conversation_length"
    INTENT_ACCURACY = "intent_accuracy"
    EMOTION_DISTRIBUTION = "emotion_distribution"
    ERROR_RATE = "error_rate"
    ENGAGEMENT_SCORE = "engagement_score"
    RETENTION_RATE = "retention_rate"

@dataclass
class AnalyticsEvent:
    event_id: str
    event_type: EventType
    user_id: str
    session_id: str
    tenant_id: str
    timestamp: float
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ConversationMetrics:
    session_id: str
    user_id: str
    tenant_id: str
    start_time: float
    end_time: float
    message_count: int
    avg_response_time: float
    user_satisfaction: Optional[float]
    emotions_detected: List[str]
    intents_recognized: List[str]
    errors_count: int
    engagement_score: float
    completion_rate: float

@dataclass
class UserBehaviorProfile:
    user_id: str
    tenant_id: str
    total_sessions: int
    total_messages: int
    avg_session_duration: float
    preferred_topics: List[str]
    emotion_patterns: Dict[str, float]
    activity_patterns: Dict[str, float]  # 时间段活跃度
    satisfaction_trend: List[float]
    last_active: float
    churn_risk: float

@dataclass
class QualityMetrics:
    period_start: float
    period_end: float
    total_conversations: int
    avg_response_time: float
    avg_satisfaction: float
    intent_accuracy: float
    error_rate: float
    completion_rate: float
    engagement_score: float
    top_issues: List[Dict[str, Any]]

class AnalyticsCollector:
    """分析数据收集器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        self.event_buffer = deque(maxlen=10000)
        self.buffer_lock = asyncio.Lock()
        
    async def initialize(self):
        """初始化收集器"""
        self.redis_client = aioredis.from_url(self.redis_url)
        
        # 启动批量处理任务
        asyncio.create_task(self._batch_process_events())
        
        logger.info("Analytics collector initialized")
    
    async def track_event(self, event: AnalyticsEvent):
        """跟踪事件"""
        async with self.buffer_lock:
            self.event_buffer.append(event)
        
        # 实时事件也推送到Redis
        await self._push_to_redis(event)
    
    async def track_user_message(self, user_id: str, session_id: str, tenant_id: str, 
                               message: str, metadata: Dict[str, Any] = None):
        """跟踪用户消息"""
        event = AnalyticsEvent(
            event_id=f"msg_{int(time.time() * 1000)}_{user_id}",
            event_type=EventType.USER_MESSAGE,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            timestamp=time.time(),
            data={
                "message": message,
                "message_length": len(message),
                "word_count": len(message.split())
            },
            metadata=metadata or {}
        )
        
        await self.track_event(event)
    
    async def track_bot_response(self, user_id: str, session_id: str, tenant_id: str,
                               response: str, response_time: float, model_id: str,
                               metadata: Dict[str, Any] = None):
        """跟踪机器人响应"""
        event = AnalyticsEvent(
            event_id=f"resp_{int(time.time() * 1000)}_{user_id}",
            event_type=EventType.BOT_RESPONSE,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            timestamp=time.time(),
            data={
                "response": response,
                "response_time": response_time,
                "model_id": model_id,
                "response_length": len(response),
                "word_count": len(response.split())
            },
            metadata=metadata or {}
        )
        
        await self.track_event(event)
    
    async def track_emotion(self, user_id: str, session_id: str, tenant_id: str,
                          emotion: str, confidence: float, metadata: Dict[str, Any] = None):
        """跟踪情感检测"""
        event = AnalyticsEvent(
            event_id=f"emotion_{int(time.time() * 1000)}_{user_id}",
            event_type=EventType.EMOTION_DETECTED,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            timestamp=time.time(),
            data={
                "emotion": emotion,
                "confidence": confidence
            },
            metadata=metadata or {}
        )
        
        await self.track_event(event)
    
    async def track_feedback(self, user_id: str, session_id: str, tenant_id: str,
                           rating: float, feedback_text: str = "", metadata: Dict[str, Any] = None):
        """跟踪用户反馈"""
        event = AnalyticsEvent(
            event_id=f"feedback_{int(time.time() * 1000)}_{user_id}",
            event_type=EventType.FEEDBACK_RECEIVED,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            timestamp=time.time(),
            data={
                "rating": rating,
                "feedback_text": feedback_text
            },
            metadata=metadata or {}
        )
        
        await self.track_event(event)
    
    async def _push_to_redis(self, event: AnalyticsEvent):
        """推送事件到Redis"""
        if not self.redis_client:
            return
        
        try:
            # 推送到实时事件流
            await self.redis_client.xadd(
                f"analytics:events:{event.tenant_id}",
                asdict(event)
            )
            
            # 更新实时指标
            await self._update_realtime_metrics(event)
            
        except Exception as e:
            logger.error(f"Failed to push event to Redis: {e}")
    
    async def _update_realtime_metrics(self, event: AnalyticsEvent):
        """更新实时指标"""
        if not self.redis_client:
            return
        
        key_prefix = f"metrics:{event.tenant_id}"
        
        try:
            # 更新计数器
            await self.redis_client.hincrby(f"{key_prefix}:counters", event.event_type.value, 1)
            
            # 更新响应时间
            if event.event_type == EventType.BOT_RESPONSE:
                response_time = event.data.get("response_time", 0)
                await self.redis_client.lpush(f"{key_prefix}:response_times", response_time)
                await self.redis_client.ltrim(f"{key_prefix}:response_times", 0, 999)  # 保留最近1000个
            
            # 更新用户活跃度
            await self.redis_client.sadd(f"{key_prefix}:active_users", event.user_id)
            await self.redis_client.expire(f"{key_prefix}:active_users", 3600)  # 1小时过期
            
        except Exception as e:
            logger.error(f"Failed to update realtime metrics: {e}")
    
    async def _batch_process_events(self):
        """批量处理事件"""
        while True:
            try:
                if len(self.event_buffer) >= 100:  # 批量大小
                    async with self.buffer_lock:
                        events = list(self.event_buffer)
                        self.event_buffer.clear()
                    
                    # 处理事件批次
                    await self._process_event_batch(events)
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(5)
    
    async def _process_event_batch(self, events: List[AnalyticsEvent]):
        """处理事件批次"""
        # 这里可以实现批量写入数据库、计算聚合指标等
        logger.info(f"Processing batch of {len(events)} events")

class ConversationAnalyzer:
    """对话分析器"""
    
    def __init__(self):
        self.session_data = defaultdict(list)
        
    async def analyze_conversation(self, events: List[AnalyticsEvent]) -> ConversationMetrics:
        """分析对话"""
        if not events:
            return None
        
        # 按会话分组
        session_events = defaultdict(list)
        for event in events:
            session_events[event.session_id].append(event)
        
        results = []
        for session_id, session_events_list in session_events.items():
            metrics = await self._analyze_session(session_id, session_events_list)
            if metrics:
                results.append(metrics)
        
        return results
    
    async def _analyze_session(self, session_id: str, events: List[AnalyticsEvent]) -> ConversationMetrics:
        """分析单个会话"""
        if not events:
            return None
        
        # 排序事件
        events.sort(key=lambda x: x.timestamp)
        
        user_id = events[0].user_id
        tenant_id = events[0].tenant_id
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        
        # 统计消息数量
        user_messages = [e for e in events if e.event_type == EventType.USER_MESSAGE]
        bot_responses = [e for e in events if e.event_type == EventType.BOT_RESPONSE]
        message_count = len(user_messages) + len(bot_responses)
        
        # 计算平均响应时间
        response_times = [e.data.get("response_time", 0) for e in bot_responses]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # 获取用户满意度
        feedback_events = [e for e in events if e.event_type == EventType.FEEDBACK_RECEIVED]
        user_satisfaction = None
        if feedback_events:
            ratings = [e.data.get("rating", 0) for e in feedback_events]
            user_satisfaction = np.mean(ratings)
        
        # 检测到的情感
        emotion_events = [e for e in events if e.event_type == EventType.EMOTION_DETECTED]
        emotions_detected = [e.data.get("emotion") for e in emotion_events]
        
        # 识别的意图
        intent_events = [e for e in events if e.event_type == EventType.INTENT_RECOGNIZED]
        intents_recognized = [e.data.get("intent") for e in intent_events]
        
        # 错误计数
        error_events = [e for e in events if e.event_type == EventType.ERROR_OCCURRED]
        errors_count = len(error_events)
        
        # 计算参与度分数
        engagement_score = self._calculate_engagement_score(events)
        
        # 计算完成率
        completion_rate = self._calculate_completion_rate(events)
        
        return ConversationMetrics(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            message_count=message_count,
            avg_response_time=avg_response_time,
            user_satisfaction=user_satisfaction,
            emotions_detected=emotions_detected,
            intents_recognized=intents_recognized,
            errors_count=errors_count,
            engagement_score=engagement_score,
            completion_rate=completion_rate
        )
    
    def _calculate_engagement_score(self, events: List[AnalyticsEvent]) -> float:
        """计算参与度分数"""
        if not events:
            return 0.0
        
        score = 0.0
        
        # 消息数量权重
        user_messages = len([e for e in events if e.event_type == EventType.USER_MESSAGE])
        score += min(user_messages * 0.1, 1.0)
        
        # 会话时长权重
        duration = events[-1].timestamp - events[0].timestamp
        score += min(duration / 3600, 1.0)  # 最多1小时
        
        # 情感多样性权重
        emotions = set([e.data.get("emotion") for e in events if e.event_type == EventType.EMOTION_DETECTED])
        score += len(emotions) * 0.1
        
        # 反馈权重
        feedback_events = [e for e in events if e.event_type == EventType.FEEDBACK_RECEIVED]
        if feedback_events:
            avg_rating = np.mean([e.data.get("rating", 0) for e in feedback_events])
            score += avg_rating / 5.0
        
        return min(score, 5.0)  # 最大5分
    
    def _calculate_completion_rate(self, events: List[AnalyticsEvent]) -> float:
        """计算完成率"""
        # 简化实现：基于会话是否正常结束
        session_end_events = [e for e in events if e.event_type == EventType.SESSION_END]
        return 1.0 if session_end_events else 0.5

class UserBehaviorAnalyzer:
    """用户行为分析器"""
    
    def __init__(self):
        self.user_profiles = {}
        
    async def analyze_user_behavior(self, user_id: str, events: List[AnalyticsEvent]) -> UserBehaviorProfile:
        """分析用户行为"""
        if not events:
            return None
        
        # 按用户过滤事件
        user_events = [e for e in events if e.user_id == user_id]
        if not user_events:
            return None
        
        tenant_id = user_events[0].tenant_id
        
        # 统计会话数
        sessions = set([e.session_id for e in user_events])
        total_sessions = len(sessions)
        
        # 统计消息数
        user_messages = [e for e in user_events if e.event_type == EventType.USER_MESSAGE]
        total_messages = len(user_messages)
        
        # 计算平均会话时长
        session_durations = []
        for session_id in sessions:
            session_events = [e for e in user_events if e.session_id == session_id]
            if len(session_events) > 1:
                duration = session_events[-1].timestamp - session_events[0].timestamp
                session_durations.append(duration)
        
        avg_session_duration = np.mean(session_durations) if session_durations else 0
        
        # 分析偏好主题（基于意图）
        intent_events = [e for e in user_events if e.event_type == EventType.INTENT_RECOGNIZED]
        intent_counts = defaultdict(int)
        for event in intent_events:
            intent = event.data.get("intent")
            if intent:
                intent_counts[intent] += 1
        
        preferred_topics = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        preferred_topics = [topic for topic, count in preferred_topics]
        
        # 分析情感模式
        emotion_events = [e for e in user_events if e.event_type == EventType.EMOTION_DETECTED]
        emotion_counts = defaultdict(int)
        for event in emotion_events:
            emotion = event.data.get("emotion")
            if emotion:
                emotion_counts[emotion] += 1
        
        total_emotions = sum(emotion_counts.values())
        emotion_patterns = {}
        if total_emotions > 0:
            emotion_patterns = {emotion: count/total_emotions for emotion, count in emotion_counts.items()}
        
        # 分析活跃时间模式
        activity_patterns = self._analyze_activity_patterns(user_events)
        
        # 分析满意度趋势
        satisfaction_trend = self._analyze_satisfaction_trend(user_events)
        
        # 计算流失风险
        churn_risk = self._calculate_churn_risk(user_events)
        
        # 最后活跃时间
        last_active = max([e.timestamp for e in user_events])
        
        return UserBehaviorProfile(
            user_id=user_id,
            tenant_id=tenant_id,
            total_sessions=total_sessions,
            total_messages=total_messages,
            avg_session_duration=avg_session_duration,
            preferred_topics=preferred_topics,
            emotion_patterns=emotion_patterns,
            activity_patterns=activity_patterns,
            satisfaction_trend=satisfaction_trend,
            last_active=last_active,
            churn_risk=churn_risk
        )
    
    def _analyze_activity_patterns(self, events: List[AnalyticsEvent]) -> Dict[str, float]:
        """分析活跃时间模式"""
        hour_counts = defaultdict(int)
        
        for event in events:
            hour = datetime.fromtimestamp(event.timestamp).hour
            hour_counts[f"hour_{hour}"] += 1
        
        total_events = len(events)
        if total_events == 0:
            return {}
        
        return {hour: count/total_events for hour, count in hour_counts.items()}
    
    def _analyze_satisfaction_trend(self, events: List[AnalyticsEvent]) -> List[float]:
        """分析满意度趋势"""
        feedback_events = [e for e in events if e.event_type == EventType.FEEDBACK_RECEIVED]
        
        # 按时间排序
        feedback_events.sort(key=lambda x: x.timestamp)
        
        # 计算滑动平均
        ratings = [e.data.get("rating", 0) for e in feedback_events]
        if len(ratings) < 3:
            return ratings
        
        # 简单移动平均
        trend = []
        window_size = 3
        for i in range(len(ratings) - window_size + 1):
            avg = np.mean(ratings[i:i + window_size])
            trend.append(avg)
        
        return trend
    
    def _calculate_churn_risk(self, events: List[AnalyticsEvent]) -> float:
        """计算流失风险"""
        if not events:
            return 1.0
        
        # 最后活跃时间
        last_active = max([e.timestamp for e in events])
        days_since_last_active = (time.time() - last_active) / (24 * 3600)
        
        # 基于时间的风险
        time_risk = min(days_since_last_active / 30, 1.0)  # 30天未活跃为高风险
        
        # 基于满意度的风险
        feedback_events = [e for e in events if e.event_type == EventType.FEEDBACK_RECEIVED]
        satisfaction_risk = 0.0
        if feedback_events:
            recent_ratings = [e.data.get("rating", 0) for e in feedback_events[-5:]]  # 最近5次反馈
            avg_rating = np.mean(recent_ratings)
            satisfaction_risk = max(0, (3 - avg_rating) / 3)  # 3分以下为风险
        
        # 基于活跃度的风险
        recent_events = [e for e in events if e.timestamp > time.time() - 7*24*3600]  # 最近7天
        activity_risk = max(0, (10 - len(recent_events)) / 10)  # 少于10个事件为风险
        
        # 综合风险评分
        total_risk = (time_risk * 0.4 + satisfaction_risk * 0.3 + activity_risk * 0.3)
        
        return min(total_risk, 1.0)

class QualityAnalyzer:
    """质量分析器"""
    
    async def analyze_quality(self, events: List[AnalyticsEvent], 
                            period_start: float, period_end: float) -> QualityMetrics:
        """分析对话质量"""
        period_events = [e for e in events if period_start <= e.timestamp <= period_end]
        
        if not period_events:
            return QualityMetrics(
                period_start=period_start,
                period_end=period_end,
                total_conversations=0,
                avg_response_time=0,
                avg_satisfaction=0,
                intent_accuracy=0,
                error_rate=0,
                completion_rate=0,
                engagement_score=0,
                top_issues=[]
            )
        
        # 统计对话数量
        sessions = set([e.session_id for e in period_events])
        total_conversations = len(sessions)
        
        # 计算平均响应时间
        response_events = [e for e in period_events if e.event_type == EventType.BOT_RESPONSE]
        response_times = [e.data.get("response_time", 0) for e in response_events]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # 计算平均满意度
        feedback_events = [e for e in period_events if e.event_type == EventType.FEEDBACK_RECEIVED]
        ratings = [e.data.get("rating", 0) for e in feedback_events]
        avg_satisfaction = np.mean(ratings) if ratings else 0
        
        # 计算意图准确率（简化实现）
        intent_events = [e for e in period_events if e.event_type == EventType.INTENT_RECOGNIZED]
        intent_accuracy = 0.85  # 模拟值，实际需要根据意图识别结果计算
        
        # 计算错误率
        error_events = [e for e in period_events if e.event_type == EventType.ERROR_OCCURRED]
        error_rate = len(error_events) / len(period_events) if period_events else 0
        
        # 计算完成率
        session_end_events = [e for e in period_events if e.event_type == EventType.SESSION_END]
        completion_rate = len(session_end_events) / total_conversations if total_conversations > 0 else 0
        
        # 计算参与度分数
        engagement_scores = []
        for session_id in sessions:
            session_events = [e for e in period_events if e.session_id == session_id]
            score = self._calculate_session_engagement(session_events)
            engagement_scores.append(score)
        
        avg_engagement_score = np.mean(engagement_scores) if engagement_scores else 0
        
        # 识别主要问题
        top_issues = await self._identify_top_issues(period_events)
        
        return QualityMetrics(
            period_start=period_start,
            period_end=period_end,
            total_conversations=total_conversations,
            avg_response_time=avg_response_time,
            avg_satisfaction=avg_satisfaction,
            intent_accuracy=intent_accuracy,
            error_rate=error_rate,
            completion_rate=completion_rate,
            engagement_score=avg_engagement_score,
            top_issues=top_issues
        )
    
    def _calculate_session_engagement(self, events: List[AnalyticsEvent]) -> float:
        """计算会话参与度"""
        if not events:
            return 0.0
        
        # 消息交互次数
        user_messages = len([e for e in events if e.event_type == EventType.USER_MESSAGE])
        bot_responses = len([e for e in events if e.event_type == EventType.BOT_RESPONSE])
        
        # 会话时长
        duration = events[-1].timestamp - events[0].timestamp if len(events) > 1 else 0
        
        # 简单的参与度计算
        engagement = (user_messages + bot_responses) * 0.1 + min(duration / 600, 1.0)  # 10分钟为满分
        
        return min(engagement, 5.0)
    
    async def _identify_top_issues(self, events: List[AnalyticsEvent]) -> List[Dict[str, Any]]:
        """识别主要问题"""
        issues = []
        
        # 高错误率问题
        error_events = [e for e in events if e.event_type == EventType.ERROR_OCCURRED]
        if len(error_events) > len(events) * 0.05:  # 错误率超过5%
            issues.append({
                "type": "high_error_rate",
                "description": "系统错误率过高",
                "severity": "high",
                "count": len(error_events),
                "percentage": len(error_events) / len(events) * 100
            })
        
        # 响应时间过长问题
        response_events = [e for e in events if e.event_type == EventType.BOT_RESPONSE]
        slow_responses = [e for e in response_events if e.data.get("response_time", 0) > 5.0]
        if len(slow_responses) > len(response_events) * 0.1:  # 超过10%的响应时间过长
            issues.append({
                "type": "slow_response",
                "description": "响应时间过长",
                "severity": "medium",
                "count": len(slow_responses),
                "percentage": len(slow_responses) / len(response_events) * 100 if response_events else 0
            })
        
        # 低满意度问题
        feedback_events = [e for e in events if e.event_type == EventType.FEEDBACK_RECEIVED]
        low_ratings = [e for e in feedback_events if e.data.get("rating", 0) < 3]
        if len(low_ratings) > len(feedback_events) * 0.3:  # 超过30%的低评分
            issues.append({
                "type": "low_satisfaction",
                "description": "用户满意度较低",
                "severity": "high",
                "count": len(low_ratings),
                "percentage": len(low_ratings) / len(feedback_events) * 100 if feedback_events else 0
            })
        
        return issues

class AnalyticsDashboard:
    """分析仪表板"""
    
    def __init__(self, collector: AnalyticsCollector):
        self.collector = collector
        self.conversation_analyzer = ConversationAnalyzer()
        self.user_analyzer = UserBehaviorAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
    
    async def get_realtime_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """获取实时指标"""
        if not self.collector.redis_client:
            return {}
        
        try:
            key_prefix = f"metrics:{tenant_id}"
            
            # 获取计数器
            counters = await self.collector.redis_client.hgetall(f"{key_prefix}:counters")
            
            # 获取响应时间
            response_times = await self.collector.redis_client.lrange(f"{key_prefix}:response_times", 0, -1)
            response_times = [float(rt) for rt in response_times]
            
            # 获取活跃用户数
            active_users_count = await self.collector.redis_client.scard(f"{key_prefix}:active_users")
            
            return {
                "counters": {k.decode(): int(v.decode()) for k, v in counters.items()},
                "avg_response_time": np.mean(response_times) if response_times else 0,
                "active_users": active_users_count,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get realtime metrics: {e}")
            return {}
    
    async def generate_report(self, tenant_id: str, start_time: float, end_time: float) -> Dict[str, Any]:
        """生成分析报告"""
        # 这里需要从数据库获取事件数据
        # 简化实现，返回模拟数据
        
        report = {
            "tenant_id": tenant_id,
            "period": {
                "start": start_time,
                "end": end_time
            },
            "summary": {
                "total_conversations": 150,
                "total_users": 45,
                "avg_response_time": 1.2,
                "avg_satisfaction": 4.2,
                "completion_rate": 0.85
            },
            "trends": {
                "daily_conversations": [20, 25, 30, 28, 35, 40, 45],
                "satisfaction_trend": [4.0, 4.1, 4.3, 4.2, 4.4, 4.1, 4.2]
            },
            "top_issues": [
                {
                    "type": "slow_response",
                    "description": "部分响应时间超过5秒",
                    "count": 12,
                    "severity": "medium"
                }
            ],
            "user_insights": {
                "new_users": 8,
                "returning_users": 37,
                "churn_risk_users": 5
            }
        }
        
        return report

# 使用示例
async def create_analytics_system():
    """创建分析系统"""
    collector = AnalyticsCollector()
    await collector.initialize()
    
    dashboard = AnalyticsDashboard(collector)
    
    return collector, dashboard

if __name__ == "__main__":
    # 测试代码
    async def test_analytics():
        collector, dashboard = await create_analytics_system()
        
        # 模拟事件
        await collector.track_user_message(
            user_id="user_123",
            session_id="session_456",
            tenant_id="tenant_789",
            message="Hello, how are you?"
        )
        
        await collector.track_bot_response(
            user_id="user_123",
            session_id="session_456",
            tenant_id="tenant_789",
            response="I'm doing well, thank you!",
            response_time=1.5,
            model_id="gpt-4"
        )
        
        await collector.track_emotion(
            user_id="user_123",
            session_id="session_456",
            tenant_id="tenant_789",
            emotion="happy",
            confidence=0.85
        )
        
        # 获取实时指标
        metrics = await dashboard.get_realtime_metrics("tenant_789")
        print("Realtime metrics:", json.dumps(metrics, indent=2))
        
        # 生成报告
        report = await dashboard.generate_report(
            tenant_id="tenant_789",
            start_time=time.time() - 7*24*3600,  # 7天前
            end_time=time.time()
        )
        print("Analytics report:", json.dumps(report, indent=2))
    
    asyncio.run(test_analytics())
