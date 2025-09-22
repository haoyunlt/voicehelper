"""
TTS服务基类 - 支持可中断流式合成
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict
import asyncio
import structlog

from app.models.schemas import AudioChunk, TTSRequest

logger = structlog.get_logger()


class TTSService(ABC):
    """TTS服务基类"""
    
    def __init__(self):
        self.active_sessions: Dict[str, asyncio.Event] = {}
    
    @abstractmethod
    async def stream_synthesis(
        self, 
        request: TTSRequest
    ) -> AsyncIterator[AudioChunk]:
        """流式语音合成"""
        pass
    
    async def cancel(self, session_id: str) -> bool:
        """取消指定会话的TTS合成"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].set()
            logger.info("TTS cancelled", session_id=session_id)
            return True
        return False
    
    def _is_cancelled(self, session_id: str) -> bool:
        """检查会话是否被取消"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].is_set()
        return False
    
    def _register_session(self, session_id: str):
        """注册会话"""
        self.active_sessions[session_id] = asyncio.Event()
    
    def _cleanup_session(self, session_id: str):
        """清理会话"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


class MockTTSService(TTSService):
    """Mock TTS服务，用于测试"""
    
    async def stream_synthesis(
        self, 
        request: TTSRequest
    ) -> AsyncIterator[AudioChunk]:
        """模拟流式语音合成"""
        session_id = request.session_id
        self._register_session(session_id)
        
        try:
            # 模拟将文本分割成多个音频片段
            words = request.text.split()
            chunk_duration_ms = 200  # 每个片段200ms
            
            for i, word in enumerate(words):
                # 检查是否被取消
                if self._is_cancelled(session_id):
                    logger.info("TTS synthesis cancelled", session_id=session_id)
                    break
                
                # 模拟音频数据生成延迟
                await asyncio.sleep(0.1)
                
                # 生成模拟音频数据
                audio_data = b'\x00' * 3200  # 16kHz, 16bit, mono, 200ms
                
                chunk = AudioChunk(
                    data=audio_data,
                    timestamp_ms=i * chunk_duration_ms,
                    sample_rate=16000,
                    channels=1,
                    format="pcm"
                )
                
                yield chunk
                
                logger.debug(
                    "TTS chunk generated",
                    session_id=session_id,
                    word=word,
                    chunk_index=i
                )
        
        except Exception as e:
            logger.error(
                "Error in TTS synthesis",
                session_id=session_id,
                error=str(e)
            )
        finally:
            self._cleanup_session(session_id)
