"""
事件发射器 - 算法服务事件化核心模块
功能: LangGraph节点事件发射 + 异步事件总线 + 事件持久化
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Callable, Optional, AsyncIterator
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    """事件类型枚举"""
    # Agent事件
    AGENT_PLAN = "agent_plan"
    AGENT_STEP = "agent_step"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    
    # 工具调用事件
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    
    # TTS事件
    TTS_START = "tts_start"
    TTS_CHUNK = "tts_chunk"
    TTS_END = "tts_end"
    TTS_CANCELLED = "tts_cancelled"
    TTS_ERROR = "tts_error"
    
    # ASR事件
    ASR_START = "asr_start"
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    ASR_ERROR = "asr_error"
    
    # VAD事件
    VAD_SPEECH_START = "vad_speech_start"
    VAD_SPEECH_END = "vad_speech_end"
    VAD_SILENCE = "vad_silence"
    
    # LLM事件
    LLM_START = "llm_start"
    LLM_DELTA = "llm_delta"
    LLM_COMPLETE = "llm_complete"
    LLM_ERROR = "llm_error"
    
    # 控制事件
    CANCEL = "cancel"
    INTERRUPT = "interrupt"
    RESUME = "resume"
    
    # 系统事件
    SUMMARY = "summary"
    METRICS = "metrics"
    HEALTH_CHECK = "health_check"

@dataclass
class EventMeta:
    """事件元数据"""
    session_id: str
    trace_id: str
    timestamp: int
    sequence_num: int
    source: str = "algo_service"
    version: str = "1.0"

@dataclass
class Event:
    """事件数据结构"""
    type: EventType
    data: Dict[str, Any]
    meta: EventMeta
    error: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type.value,
            "data": self.data,
            "meta": asdict(self.meta),
            "error": self.error
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

class EventEmitter:
    """事件发射器"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.global_subscribers: List[Callable] = []
        self.event_history: List[Event] = []
        self.max_history = 1000
        self.sequence_counter = 0
        self._lock = asyncio.Lock()
        
    async def emit(self, event_type: EventType, data: Dict[str, Any], 
                   session_id: str, trace_id: Optional[str] = None) -> Event:
        """发射事件"""
        if trace_id is None:
            trace_id = self.generate_trace_id()
            
        async with self._lock:
            self.sequence_counter += 1
            sequence_num = self.sequence_counter
        
        # 创建事件
        event = Event(
            type=event_type,
            data=data,
            meta=EventMeta(
                session_id=session_id,
                trace_id=trace_id,
                timestamp=int(time.time() * 1000),
                sequence_num=sequence_num
            )
        )
        
        # 添加到历史记录
        await self._add_to_history(event)
        
        # 发送给订阅者
        await self._notify_subscribers(event)
        
        logger.debug(f"Event emitted: {event_type.value} for session {session_id}")
        return event
    
    async def emit_error(self, event_type: EventType, error_code: str, 
                        error_message: str, session_id: str, 
                        trace_id: Optional[str] = None, 
                        details: Optional[str] = None) -> Event:
        """发射错误事件"""
        error_data = {
            "code": error_code,
            "message": error_message,
            "details": details,
            "timestamp": int(time.time() * 1000)
        }
        
        if trace_id is None:
            trace_id = self.generate_trace_id()
            
        async with self._lock:
            self.sequence_counter += 1
            sequence_num = self.sequence_counter
        
        event = Event(
            type=event_type,
            data={},
            meta=EventMeta(
                session_id=session_id,
                trace_id=trace_id,
                timestamp=int(time.time() * 1000),
                sequence_num=sequence_num
            ),
            error=error_data
        )
        
        await self._add_to_history(event)
        await self._notify_subscribers(event)
        
        logger.error(f"Error event emitted: {event_type.value} - {error_message}")
        return event
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], Any]):
        """订阅特定事件类型"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def subscribe_all(self, callback: Callable[[Event], Any]):
        """订阅所有事件"""
        self.global_subscribers.append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], Any]):
        """取消订阅"""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
            except ValueError:
                pass
    
    def unsubscribe_all(self, callback: Callable[[Event], Any]):
        """取消全局订阅"""
        try:
            self.global_subscribers.remove(callback)
        except ValueError:
            pass
    
    async def get_events(self, session_id: str, 
                        event_types: Optional[List[EventType]] = None,
                        since_timestamp: Optional[int] = None,
                        limit: int = 100) -> List[Event]:
        """获取事件历史"""
        events = []
        
        for event in reversed(self.event_history):
            if len(events) >= limit:
                break
                
            if event.meta.session_id != session_id:
                continue
                
            if event_types and event.type not in event_types:
                continue
                
            if since_timestamp and event.meta.timestamp <= since_timestamp:
                continue
                
            events.append(event)
        
        return list(reversed(events))
    
    async def clear_history(self, session_id: Optional[str] = None):
        """清理事件历史"""
        if session_id is None:
            self.event_history.clear()
        else:
            self.event_history = [
                event for event in self.event_history 
                if event.meta.session_id != session_id
            ]
    
    async def _add_to_history(self, event: Event):
        """添加事件到历史记录"""
        self.event_history.append(event)
        
        # 限制历史记录大小
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
    
    async def _notify_subscribers(self, event: Event):
        """通知订阅者"""
        # 通知特定事件类型的订阅者
        if event.type in self.subscribers:
            for callback in self.subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}")
        
        # 通知全局订阅者
        for callback in self.global_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in global event subscriber: {e}")
    
    def generate_trace_id(self) -> str:
        """生成追踪ID"""
        return f"trace_{uuid.uuid4().hex[:16]}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_events": len(self.event_history),
            "sequence_counter": self.sequence_counter,
            "subscribers": {
                event_type.value: len(callbacks) 
                for event_type, callbacks in self.subscribers.items()
            },
            "global_subscribers": len(self.global_subscribers)
        }

class EventStream:
    """事件流管理器"""
    
    def __init__(self, emitter: EventEmitter, session_id: str):
        self.emitter = emitter
        self.session_id = session_id
        self.trace_id = emitter.generate_trace_id()
        self._cancelled = False
        
    async def emit(self, event_type: EventType, data: Dict[str, Any]) -> Event:
        """发射事件"""
        if self._cancelled:
            raise RuntimeError("Event stream has been cancelled")
            
        return await self.emitter.emit(event_type, data, self.session_id, self.trace_id)
    
    async def emit_error(self, event_type: EventType, error_code: str, 
                        error_message: str, details: Optional[str] = None) -> Event:
        """发射错误事件"""
        return await self.emitter.emit_error(
            event_type, error_code, error_message, 
            self.session_id, self.trace_id, details
        )
    
    def cancel(self):
        """取消事件流"""
        self._cancelled = True
    
    @property
    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self._cancelled
    
    @asynccontextmanager
    async def step_context(self, step_name: str, step_type: str = "processing"):
        """步骤上下文管理器"""
        step_id = f"{step_name}_{int(time.time() * 1000)}"
        
        # 发射步骤开始事件
        await self.emit(EventType.AGENT_STEP, {
            "step_id": step_id,
            "step_name": step_name,
            "step_type": step_type,
            "status": "started",
            "timestamp": int(time.time() * 1000)
        })
        
        start_time = time.time()
        error = None
        
        try:
            yield step_id
        except Exception as e:
            error = e
            # 发射步骤错误事件
            await self.emit_error(EventType.AGENT_ERROR, 
                                "step_error", str(e), 
                                details=f"Step: {step_name}")
            raise
        finally:
            # 发射步骤完成事件
            duration = int((time.time() - start_time) * 1000)
            await self.emit(EventType.AGENT_STEP, {
                "step_id": step_id,
                "step_name": step_name,
                "step_type": step_type,
                "status": "error" if error else "completed",
                "duration_ms": duration,
                "timestamp": int(time.time() * 1000)
            })

# 全局事件发射器实例
_global_emitter = EventEmitter()

def get_event_emitter() -> EventEmitter:
    """获取全局事件发射器"""
    return _global_emitter

def create_event_stream(session_id: str) -> EventStream:
    """创建事件流"""
    return EventStream(_global_emitter, session_id)

# 便捷函数
async def emit_event(event_type: EventType, data: Dict[str, Any], 
                    session_id: str, trace_id: Optional[str] = None) -> Event:
    """发射事件的便捷函数"""
    return await _global_emitter.emit(event_type, data, session_id, trace_id)

async def emit_agent_plan(intent: str, steps: List[str], context: Dict[str, Any], 
                         session_id: str) -> Event:
    """发射Agent计划事件"""
    return await emit_event(EventType.AGENT_PLAN, {
        "intent": intent,
        "steps": steps,
        "context": context,
        "timestamp": int(time.time() * 1000)
    }, session_id)

async def emit_tool_result(tool_name: str, input_data: Dict[str, Any], 
                          output_data: Dict[str, Any], success: bool, 
                          duration_ms: int, session_id: str, 
                          error: Optional[str] = None) -> Event:
    """发射工具调用结果事件"""
    return await emit_event(EventType.TOOL_RESULT, {
        "tool_name": tool_name,
        "input": input_data,
        "output": output_data,
        "success": success,
        "error": error,
        "duration_ms": duration_ms,
        "timestamp": int(time.time() * 1000)
    }, session_id)

async def emit_tts_chunk(audio_data: bytes, chunk_id: int, format_type: str, 
                        session_id: str) -> Event:
    """发射TTS音频块事件"""
    return await emit_event(EventType.TTS_CHUNK, {
        "audio_data": audio_data.hex(),  # 转换为hex字符串
        "chunk_id": chunk_id,
        "format": format_type,
        "size": len(audio_data),
        "timestamp": int(time.time() * 1000)
    }, session_id)

async def emit_summary(summary_text: str, context: Dict[str, Any], 
                      session_id: str) -> Event:
    """发射摘要事件"""
    return await emit_event(EventType.SUMMARY, {
        "summary": summary_text,
        "context": context,
        "timestamp": int(time.time() * 1000)
    }, session_id)
