"""
STT服务基类
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List
import asyncio
import structlog

from app.models.schemas import Transcription, AudioChunk

logger = structlog.get_logger()


class STTService(ABC):
    """STT服务基类"""
    
    def __init__(self):
        self.active_streams = {}
    
    @abstractmethod
    async def start_stream(
        self, 
        session_id: str,
        language: str = "zh-CN",
        hints: Optional[List[str]] = None
    ) -> None:
        """开始流式识别"""
        pass
    
    @abstractmethod
    async def ingest_audio(
        self, 
        session_id: str,
        audio_chunk: AudioChunk
    ) -> Optional[Transcription]:
        """输入音频数据，返回识别结果"""
        pass
    
    @abstractmethod
    async def finish_stream(self, session_id: str) -> Optional[Transcription]:
        """结束流式识别，返回最终结果"""
        pass
    
    async def cleanup_session(self, session_id: str) -> None:
        """清理会话资源"""
        if session_id in self.active_streams:
            del self.active_streams[session_id]
            logger.info("STT session cleaned up", session_id=session_id)


class MockSTTService(STTService):
    """Mock STT服务，用于测试"""
    
    async def start_stream(
        self, 
        session_id: str,
        language: str = "zh-CN",
        hints: Optional[List[str]] = None
    ) -> None:
        self.active_streams[session_id] = {
            "language": language,
            "hints": hints or [],
            "buffer": []
        }
        logger.info("Mock STT stream started", session_id=session_id)
    
    async def ingest_audio(
        self, 
        session_id: str,
        audio_chunk: AudioChunk
    ) -> Optional[Transcription]:
        if session_id not in self.active_streams:
            return None
        
        # 模拟识别延迟
        await asyncio.sleep(0.1)
        
        # 返回模拟结果
        return Transcription(
            text="这是模拟的语音识别结果",
            confidence=0.95,
            start_time_ms=audio_chunk.timestamp_ms,
            end_time_ms=audio_chunk.timestamp_ms + 1000,
            is_final=False,
            language="zh-CN"
        )
    
    async def finish_stream(self, session_id: str) -> Optional[Transcription]:
        if session_id not in self.active_streams:
            return None
        
        result = Transcription(
            text="这是最终的语音识别结果",
            confidence=0.98,
            start_time_ms=0,
            end_time_ms=2000,
            is_final=True,
            language="zh-CN"
        )
        
        await self.cleanup_session(session_id)
        return result
