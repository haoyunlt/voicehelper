"""
Deepgram Aura TTS实现
"""
import asyncio
import aiohttp
from typing import AsyncIterator
import structlog

from app.config import settings
from app.models.schemas import AudioChunk, TTSRequest
from .base import TTSService

logger = structlog.get_logger()


class AuraTTS(TTSService):
    """Deepgram Aura TTS服务实现"""
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is required for Aura TTS")
        
        self.base_url = "https://api.deepgram.com/v1/speak"
    
    async def stream_synthesis(
        self, 
        request: TTSRequest
    ) -> AsyncIterator[AudioChunk]:
        """使用Deepgram Aura进行流式语音合成"""
        session_id = request.session_id
        self._register_session(session_id)
        
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": request.text,
                "model": "aura-asteria-en",  # 可配置
                "encoding": "linear16",
                "sample_rate": 16000,
                "container": "none"
            }
            
            # 如果指定了语音类型
            if request.voice:
                payload["model"] = request.voice
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            "Aura TTS API error",
                            session_id=session_id,
                            status=response.status,
                            error=error_text
                        )
                        return
                    
                    # 流式读取音频数据
                    chunk_size = 3200  # 200ms at 16kHz
                    timestamp_ms = 0
                    chunk_index = 0
                    
                    async for data in response.content.iter_chunked(chunk_size):
                        # 检查是否被取消
                        if self._is_cancelled(session_id):
                            logger.info("Aura TTS synthesis cancelled", session_id=session_id)
                            break
                        
                        if data:
                            chunk = AudioChunk(
                                data=data,
                                timestamp_ms=timestamp_ms,
                                sample_rate=16000,
                                channels=1,
                                format="pcm"
                            )
                            
                            yield chunk
                            
                            timestamp_ms += 200  # 每个chunk 200ms
                            chunk_index += 1
                            
                            logger.debug(
                                "Aura TTS chunk generated",
                                session_id=session_id,
                                chunk_index=chunk_index,
                                size=len(data)
                            )
                            
                            # 小延迟，避免过快发送
                            await asyncio.sleep(0.05)
        
        except Exception as e:
            logger.error(
                "Error in Aura TTS synthesis",
                session_id=session_id,
                error=str(e)
            )
        finally:
            self._cleanup_session(session_id)
            logger.info("Aura TTS synthesis completed", session_id=session_id)
