"""
Prometheus指标收集
"""
import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
import structlog

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from app.config import settings

logger = structlog.get_logger()


class VoiceMetrics:
    """语音服务指标收集器"""
    
    def __init__(self):
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.metrics_data = defaultdict(list)
        self.session_metrics = {}
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            logger.warning("Prometheus client not available, using in-memory metrics")
    
    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        # 会话指标
        self.session_counter = Counter(
            'voice_sessions_total',
            'Total number of voice sessions',
            ['status'],
            registry=self.registry
        )
        
        # 延迟指标
        self.latency_histogram = Histogram(
            'voice_latency_seconds',
            'Voice processing latency',
            ['phase'],
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        # 错误指标
        self.error_counter = Counter(
            'voice_errors_total',
            'Total number of voice processing errors',
            ['error_type', 'service'],
            registry=self.registry
        )
        
        # 活跃会话
        self.active_sessions_gauge = Gauge(
            'voice_active_sessions',
            'Number of active voice sessions',
            registry=self.registry
        )
        
        # TTS取消指标
        self.tts_cancel_counter = Counter(
            'voice_tts_cancellations_total',
            'Total number of TTS cancellations',
            ['reason'],
            registry=self.registry
        )
        
        # Barge-in指标
        self.barge_in_counter = Counter(
            'voice_barge_in_total',
            'Total number of barge-in events',
            ['trigger_type'],
            registry=self.registry
        )
        
        # 音频质量指标
        self.audio_quality_histogram = Histogram(
            'voice_audio_quality',
            'Audio quality metrics',
            ['metric_type'],
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
            registry=self.registry
        )
        
        # 成本指标
        self.cost_counter = Counter(
            'voice_cost_tokens_total',
            'Total tokens consumed',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # 缓存指标
        self.cache_counter = Counter(
            'voice_cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
    
    def start_metrics_server(self, port: Optional[int] = None):
        """启动Prometheus指标服务器"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Cannot start metrics server: Prometheus client not available")
            return
        
        port = port or settings.PROMETHEUS_METRICS_PORT
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    async def record_session_event(self, event_type: str, session_id: str, user_id: str):
        """记录会话事件"""
        timestamp = time.time()
        
        if PROMETHEUS_AVAILABLE:
            self.session_counter.labels(status=event_type).inc()
        
        # 内存指标
        self.metrics_data['session_events'].append({
            'timestamp': timestamp,
            'event_type': event_type,
            'session_id': session_id,
            'user_id': user_id
        })
        
        # 更新活跃会话计数
        if event_type == 'session_created':
            self.session_metrics[session_id] = {
                'created_at': timestamp,
                'user_id': user_id,
                'events': []
            }
            if PROMETHEUS_AVAILABLE:
                self.active_sessions_gauge.inc()
        
        elif event_type == 'session_closed':
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]
            if PROMETHEUS_AVAILABLE:
                self.active_sessions_gauge.dec()
    
    async def record_latency(
        self, 
        phase: str, 
        duration_ms: float, 
        session_id: Optional[str] = None
    ):
        """记录延迟指标"""
        duration_seconds = duration_ms / 1000.0
        
        if PROMETHEUS_AVAILABLE:
            self.latency_histogram.labels(phase=phase).observe(duration_seconds)
        
        # 内存指标
        self.metrics_data['latency'].append({
            'timestamp': time.time(),
            'phase': phase,
            'duration_ms': duration_ms,
            'session_id': session_id
        })
        
        logger.debug(
            "Latency recorded",
            phase=phase,
            duration_ms=duration_ms,
            session_id=session_id
        )
    
    async def record_error(
        self, 
        error_type: str, 
        service: str, 
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """记录错误指标"""
        if PROMETHEUS_AVAILABLE:
            self.error_counter.labels(error_type=error_type, service=service).inc()
        
        # 内存指标
        self.metrics_data['errors'].append({
            'timestamp': time.time(),
            'error_type': error_type,
            'service': service,
            'session_id': session_id,
            'details': details or {}
        })
        
        logger.warning(
            "Error recorded",
            error_type=error_type,
            service=service,
            session_id=session_id,
            details=details
        )
    
    async def record_tts_cancellation(
        self, 
        reason: str, 
        session_id: str,
        response_time_ms: Optional[float] = None
    ):
        """记录TTS取消事件"""
        if PROMETHEUS_AVAILABLE:
            self.tts_cancel_counter.labels(reason=reason).inc()
        
        # 内存指标
        self.metrics_data['tts_cancellations'].append({
            'timestamp': time.time(),
            'reason': reason,
            'session_id': session_id,
            'response_time_ms': response_time_ms
        })
        
        logger.info(
            "TTS cancellation recorded",
            reason=reason,
            session_id=session_id,
            response_time_ms=response_time_ms
        )
    
    async def record_barge_in(
        self, 
        trigger_type: str, 
        session_id: str,
        detection_latency_ms: Optional[float] = None
    ):
        """记录Barge-in事件"""
        if PROMETHEUS_AVAILABLE:
            self.barge_in_counter.labels(trigger_type=trigger_type).inc()
        
        # 内存指标
        self.metrics_data['barge_in'].append({
            'timestamp': time.time(),
            'trigger_type': trigger_type,
            'session_id': session_id,
            'detection_latency_ms': detection_latency_ms
        })
        
        logger.info(
            "Barge-in recorded",
            trigger_type=trigger_type,
            session_id=session_id,
            detection_latency_ms=detection_latency_ms
        )
    
    async def record_audio_quality(
        self, 
        metric_type: str, 
        value: float, 
        session_id: Optional[str] = None
    ):
        """记录音频质量指标"""
        if PROMETHEUS_AVAILABLE:
            self.audio_quality_histogram.labels(metric_type=metric_type).observe(value)
        
        # 内存指标
        self.metrics_data['audio_quality'].append({
            'timestamp': time.time(),
            'metric_type': metric_type,
            'value': value,
            'session_id': session_id
        })
    
    async def record_cost(
        self, 
        provider: str, 
        model: str, 
        tokens: int,
        cost_usd: Optional[float] = None
    ):
        """记录成本指标"""
        if PROMETHEUS_AVAILABLE:
            self.cost_counter.labels(provider=provider, model=model).inc(tokens)
        
        # 内存指标
        self.metrics_data['costs'].append({
            'timestamp': time.time(),
            'provider': provider,
            'model': model,
            'tokens': tokens,
            'cost_usd': cost_usd
        })
    
    async def record_cache_operation(
        self, 
        operation: str, 
        result: str,
        cache_key: Optional[str] = None
    ):
        """记录缓存操作"""
        if PROMETHEUS_AVAILABLE:
            self.cache_counter.labels(operation=operation, result=result).inc()
        
        # 内存指标
        self.metrics_data['cache_operations'].append({
            'timestamp': time.time(),
            'operation': operation,
            'result': result,
            'cache_key': cache_key
        })
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话指标摘要"""
        if session_id not in self.session_metrics:
            return None
        
        session_data = self.session_metrics[session_id]
        current_time = time.time()
        
        # 收集相关指标
        session_events = [
            event for event in self.metrics_data['session_events']
            if event['session_id'] == session_id
        ]
        
        latency_events = [
            event for event in self.metrics_data['latency']
            if event.get('session_id') == session_id
        ]
        
        error_events = [
            event for event in self.metrics_data['errors']
            if event.get('session_id') == session_id
        ]
        
        return {
            'session_id': session_id,
            'user_id': session_data['user_id'],
            'duration_seconds': current_time - session_data['created_at'],
            'total_events': len(session_events),
            'total_errors': len(error_events),
            'latency_phases': {
                event['phase']: event['duration_ms']
                for event in latency_events
            },
            'avg_latency_ms': (
                sum(event['duration_ms'] for event in latency_events) / len(latency_events)
                if latency_events else 0
            )
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统级指标摘要"""
        current_time = time.time()
        
        # 最近1小时的数据
        hour_ago = current_time - 3600
        
        recent_sessions = [
            event for event in self.metrics_data['session_events']
            if event['timestamp'] > hour_ago
        ]
        
        recent_errors = [
            event for event in self.metrics_data['errors']
            if event['timestamp'] > hour_ago
        ]
        
        recent_latency = [
            event for event in self.metrics_data['latency']
            if event['timestamp'] > hour_ago
        ]
        
        return {
            'timestamp': current_time,
            'active_sessions': len(self.session_metrics),
            'recent_hour': {
                'total_sessions': len(recent_sessions),
                'total_errors': len(recent_errors),
                'avg_latency_ms': (
                    sum(event['duration_ms'] for event in recent_latency) / len(recent_latency)
                    if recent_latency else 0
                ),
                'error_rate': len(recent_errors) / max(len(recent_sessions), 1)
            }
        }


# 全局指标收集器实例
def setup_metrics():
    """初始化指标收集"""
    global metrics_collector
    metrics_collector = VoiceMetrics()
    
    # 启动Prometheus服务器
    if PROMETHEUS_AVAILABLE:
        metrics_collector.start_metrics_server()
    
    logger.info("Metrics collection initialized")


# 全局指标收集器
metrics_collector = None


# 便捷函数
async def record_session_event(event_type: str, session_id: str, user_id: str):
    """记录会话事件的便捷函数"""
    if metrics_collector:
        await metrics_collector.record_session_event(event_type, session_id, user_id)


async def record_latency(phase: str, duration_ms: float, session_id: Optional[str] = None):
    """记录延迟的便捷函数"""
    if metrics_collector:
        await metrics_collector.record_latency(phase, duration_ms, session_id)


async def record_error(error_type: str, service: str, session_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    """记录错误的便捷函数"""
    if metrics_collector:
        await metrics_collector.record_error(error_type, service, session_id, details)
