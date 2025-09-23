"""
Deepgram STT实现
"""
import asyncio
import json
import websockets
from typing import Optional, List
import structlog

from app.config import settings
from app.models.schemas import Transcription, AudioChunk
from .base import STTService

logger = structlog.get_logger()


class DeepgramSTT(STTService):
    """Deepgram STT服务实现"""
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.DEEPGRAM_API_KEY
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY is required")
    
    async def start_stream(
        self, 
        session_id: str,
        language: str = "zh-CN",
        hints: Optional[List[str]] = None
    ) -> None:
        """开始Deepgram流式识别"""
        try:
            # Deepgram WebSocket URL
            url = (
                f"wss://api.deepgram.com/v1/listen?"
                f"encoding=linear16&sample_rate=16000&channels=1"
                f"&language={language}&punctuate=true&interim_results=true"
            )
            
            headers = {"Authorization": f"Token {self.api_key}"}
            
            # 建立WebSocket连接
            websocket = await websockets.connect(url, extra_headers=headers)
            
            self.active_streams[session_id] = {
                "websocket": websocket,
                "language": language,
                "hints": hints or [],
                "buffer": []
            }
            
            # 启动接收任务
            asyncio.create_task(self._receive_results(session_id))
            
            logger.info(
                "Deepgram STT stream started", 
                session_id=session_id,
                language=language
            )
            
        except Exception as e:
            logger.error(
                "Failed to start Deepgram stream",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def ingest_audio(
        self, 
        session_id: str,
        audio_chunk: AudioChunk
    ) -> Optional[Transcription]:
        """发送音频数据到Deepgram"""
        if session_id not in self.active_streams:
            logger.warning("STT stream not found", session_id=session_id)
            return None
        
        stream_info = self.active_streams[session_id]
        websocket = stream_info["websocket"]
        
        try:
            # 发送音频数据
            await websocket.send(audio_chunk.data)
            
            # 检查是否有新的识别结果
            if stream_info["buffer"]:
                return stream_info["buffer"].pop(0)
            
        except Exception as e:
            logger.error(
                "Failed to send audio to Deepgram",
                session_id=session_id,
                error=str(e)
            )
        
        return None
    
    async def finish_stream(self, session_id: str) -> Optional[Transcription]:
        """结束Deepgram流式识别"""
        if session_id not in self.active_streams:
            return None
        
        stream_info = self.active_streams[session_id]
        websocket = stream_info["websocket"]
        
        try:
            # 发送结束信号
            await websocket.send(json.dumps({"type": "CloseStream"}))
            
            # 等待最终结果
            await asyncio.sleep(0.5)
            
            # 获取缓冲区中的最后结果
            final_result = None
            if stream_info["buffer"]:
                final_result = stream_info["buffer"][-1]
                final_result.is_final = True
            
            # 关闭连接
            await websocket.close()
            
            logger.info("Deepgram STT stream finished", session_id=session_id)
            
        except Exception as e:
            logger.error(
                "Error finishing Deepgram stream",
                session_id=session_id,
                error=str(e)
            )
        finally:
            await self.cleanup_session(session_id)
        
        return final_result
    
    async def _receive_results(self, session_id: str):
        """接收Deepgram识别结果"""
        if session_id not in self.active_streams:
            return
        
        stream_info = self.active_streams[session_id]
        websocket = stream_info["websocket"]
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if "channel" in data:
                    alternatives = data["channel"]["alternatives"]
                    if alternatives:
                        transcript = alternatives[0]["transcript"]
                        confidence = alternatives[0]["confidence"]
                        
                        if transcript.strip():
                            result = Transcription(
                                text=transcript,
                                confidence=confidence,
                                start_time_ms=int(data.get("start", 0) * 1000),
                                end_time_ms=int(data.get("end", 0) * 1000),
                                is_final=data.get("is_final", False),
                                language=stream_info["language"]
                            )
                            
                            # 添加到缓冲区
                            stream_info["buffer"].append(result)
                            
                            logger.debug(
                                "Deepgram result received",
                                session_id=session_id,
                                text=transcript,
                                is_final=result.is_final
                            )
        
        except Exception as e:
            logger.error(
                "Error receiving Deepgram results",
                session_id=session_id,
                error=str(e)
            )
