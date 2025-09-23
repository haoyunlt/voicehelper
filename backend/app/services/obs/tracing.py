"""
OpenTelemetry分布式追踪
"""
import functools
from contextlib import contextmanager
from typing import Optional, Dict, Any
import structlog

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

from app.config import settings

logger = structlog.get_logger()

# 全局tracer
tracer = None


def setup_tracing():
    """初始化OpenTelemetry追踪"""
    global tracer
    
    try:
        # 创建资源
        resource = Resource.create({
            "service.name": "voicehelper-backend",
            "service.version": "2.0.0",
            "deployment.environment": "development" if settings.DEBUG else "production"
        })
        
        # 创建TracerProvider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # 配置导出器
        if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=settings.DEBUG
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        # 获取tracer
        tracer = trace.get_tracer(__name__)
        
        # 自动化仪表
        RequestsInstrumentor().instrument()
        AioHttpClientInstrumentor().instrument()
        
        logger.info("OpenTelemetry tracing initialized")
        
    except Exception as e:
        logger.error("Failed to initialize tracing", error=str(e))


@contextmanager
def span_ctx(
    name: str, 
    attributes: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
):
    """创建span上下文管理器"""
    if not tracer:
        # 如果tracer未初始化，使用空上下文
        class NoOpSpan:
            def set_attribute(self, key, value): pass
            def set_status(self, status): pass
            def record_exception(self, exception): pass
        
        yield NoOpSpan()
        return
    
    with tracer.start_as_current_span(name) as span:
        try:
            # 设置基础属性
            if session_id:
                span.set_attribute("session_id", session_id)
            
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            yield span
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def trace_async(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """异步函数追踪装饰器"""
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with span_ctx(span_name, attributes) as span:
                # 尝试从参数中提取session_id
                session_id = kwargs.get('session_id')
                if not session_id and args:
                    # 检查第一个参数是否有session_id属性
                    first_arg = args[0]
                    if hasattr(first_arg, 'session_id'):
                        session_id = first_arg.session_id
                
                if session_id:
                    span.set_attribute("session_id", session_id)
                
                # 记录函数参数（排除敏感信息）
                safe_kwargs = {
                    k: v for k, v in kwargs.items() 
                    if k not in ['password', 'token', 'api_key', 'secret']
                }
                span.set_attribute("function.args", str(safe_kwargs))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    return decorator


def trace_sync(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """同步函数追踪装饰器"""
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with span_ctx(span_name, attributes) as span:
                # 记录函数参数（排除敏感信息）
                safe_kwargs = {
                    k: v for k, v in kwargs.items() 
                    if k not in ['password', 'token', 'api_key', 'secret']
                }
                span.set_attribute("function.args", str(safe_kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    return decorator


class VoiceLatencyTracker:
    """语音链路延迟追踪器"""
    
    def __init__(self):
        self.active_sessions = {}
    
    def start_session(self, session_id: str):
        """开始会话追踪"""
        import time
        self.active_sessions[session_id] = {
            "start_time": time.time(),
            "phases": {},
            "events": []
        }
    
    def record_phase_start(self, session_id: str, phase: str):
        """记录阶段开始"""
        if session_id not in self.active_sessions:
            return
        
        import time
        self.active_sessions[session_id]["phases"][phase] = {
            "start_time": time.time()
        }
    
    def record_phase_end(self, session_id: str, phase: str, metadata: Optional[Dict] = None):
        """记录阶段结束"""
        if session_id not in self.active_sessions:
            return
        
        import time
        current_time = time.time()
        
        if phase in self.active_sessions[session_id]["phases"]:
            phase_info = self.active_sessions[session_id]["phases"][phase]
            phase_info["end_time"] = current_time
            phase_info["duration_ms"] = (current_time - phase_info["start_time"]) * 1000
            
            if metadata:
                phase_info["metadata"] = metadata
            
            # 记录到span
            with span_ctx(f"voice.phase.{phase}") as span:
                span.set_attribute("session_id", session_id)
                span.set_attribute("phase", phase)
                span.set_attribute("duration_ms", phase_info["duration_ms"])
                
                if metadata:
                    for key, value in metadata.items():
                        span.set_attribute(f"phase.{key}", str(value))
    
    def record_event(self, session_id: str, event_type: str, metadata: Optional[Dict] = None):
        """记录事件"""
        if session_id not in self.active_sessions:
            return
        
        import time
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "metadata": metadata or {}
        }
        
        self.active_sessions[session_id]["events"].append(event)
        
        # 记录到span
        with span_ctx(f"voice.event.{event_type}") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("event_type", event_type)
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"event.{key}", str(value))
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict]:
        """获取会话指标"""
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        import time
        current_time = time.time()
        
        metrics = {
            "session_id": session_id,
            "total_duration_ms": (current_time - session_data["start_time"]) * 1000,
            "phases": {},
            "events": session_data["events"]
        }
        
        # 计算各阶段指标
        for phase, phase_info in session_data["phases"].items():
            if "duration_ms" in phase_info:
                metrics["phases"][phase] = {
                    "duration_ms": phase_info["duration_ms"],
                    "metadata": phase_info.get("metadata", {})
                }
        
        # 计算端到端延迟
        if "capture" in metrics["phases"] and "tts_end" in metrics["phases"]:
            e2e_latency = (
                metrics["phases"]["tts_end"]["duration_ms"] - 
                metrics["phases"]["capture"]["duration_ms"]
            )
            metrics["e2e_latency_ms"] = e2e_latency
        
        return metrics
    
    def end_session(self, session_id: str):
        """结束会话追踪"""
        if session_id in self.active_sessions:
            # 记录最终指标
            final_metrics = self.get_session_metrics(session_id)
            
            with span_ctx("voice.session.complete") as span:
                span.set_attribute("session_id", session_id)
                if final_metrics:
                    span.set_attribute("total_duration_ms", final_metrics["total_duration_ms"])
                    if "e2e_latency_ms" in final_metrics:
                        span.set_attribute("e2e_latency_ms", final_metrics["e2e_latency_ms"])
            
            del self.active_sessions[session_id]
            return final_metrics
        
        return None


# 全局延迟追踪器实例
latency_tracker = VoiceLatencyTracker()
