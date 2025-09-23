"""
OpenAI Realtime TTS实现
"""
import asyncio
import json
import websockets
from typing import AsyncIterator
import structlog

from app.config import settings
from app.models.schemas import AudioChunk, TTSRequest
from .base import TTSService

logger = structlog.get_logger()


class OpenAIRealtimeTTS(TTSService):
    """OpenAI Realtime API TTS服务实现"""
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI Realtime TTS")
        
        self.base_url = "wss://api.openai.com/v1/realtime"
    
    async def stream_synthesis(
        self, 
        request: TTSRequest
    ) -> AsyncIterator[AudioChunk]:
        """使用OpenAI Realtime API进行流式语音合成"""
        session_id = request.session_id
        self._register_session(session_id)
        
        try:
            # WebSocket连接头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # 连接到OpenAI Realtime API
            async with websockets.connect(
                f"{self.base_url}?model=gpt-4o-realtime-preview-2024-10-01",
                extra_headers=headers
            ) as websocket:
                
                # 发送会话配置
                session_config = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "instructions": "You are a helpful assistant.",
                        "voice": request.voice_id or "alloy",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": "whisper-1"
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 200
                        },
                        "tools": [],
                        "tool_choice": "auto",
                        "temperature": 0.8,
                        "max_response_output_tokens": "inf"
                    }
                }
                
                await websocket.send(json.dumps(session_config))
                
                # 发送TTS请求
                tts_request = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio"],
                        "instructions": f"Please say: {request.text}"
                    }
                }
                
                await websocket.send(json.dumps(tts_request))
                
                # 接收音频流
                timestamp_ms = 0
                chunk_index = 0
                
                async for message in websocket:
                    # 检查是否被取消
                    if self._is_cancelled(session_id):
                        logger.info("OpenAI Realtime TTS synthesis cancelled", session_id=session_id)
                        # 发送取消请求
                        cancel_request = {
                            "type": "response.cancel"
                        }
                        await websocket.send(json.dumps(cancel_request))
                        break
                    
                    try:
                        data = json.loads(message)
                        
                        if data.get("type") == "response.audio.delta":
                            # 音频数据块
                            audio_data = data.get("delta", "")
                            if audio_data:
                                # Base64解码音频数据
                                import base64
                                audio_bytes = base64.b64decode(audio_data)
                                
                                chunk = AudioChunk(
                                    data=audio_bytes,
                                    timestamp_ms=timestamp_ms,
                                    sample_rate=24000,  # OpenAI Realtime使用24kHz
                                    channels=1,
                                    format="pcm"
                                )
                                
                                yield chunk
                                
                                # 估算时间戳增量 (假设每个chunk约20ms)
                                timestamp_ms += len(audio_bytes) * 1000 // (24000 * 2)
                                chunk_index += 1
                                
                                logger.debug(
                                    "OpenAI Realtime TTS chunk generated",
                                    session_id=session_id,
                                    chunk_index=chunk_index,
                                    size=len(audio_bytes)
                                )
                        
                        elif data.get("type") == "response.audio.done":
                            # 音频合成完成
                            logger.info("OpenAI Realtime TTS synthesis completed", session_id=session_id)
                            break
                        
                        elif data.get("type") == "error":
                            # 错误处理
                            error_msg = data.get("error", {}).get("message", "Unknown error")
                            logger.error(
                                "OpenAI Realtime TTS error",
                                session_id=session_id,
                                error=error_msg
                            )
                            break
                    
                    except json.JSONDecodeError as e:
                        logger.error(
                            "Failed to parse OpenAI Realtime response",
                            session_id=session_id,
                            error=str(e)
                        )
                        continue
        
        except Exception as e:
            logger.error(
                "Error in OpenAI Realtime TTS synthesis",
                session_id=session_id,
                error=str(e)
            )
        finally:
            self._cleanup_session(session_id)


class ElevenLabsTTS(TTSService):
    """ElevenLabs TTS服务实现（备选方案）"""
    
    def __init__(self):
        super().__init__()
        self.api_key = settings.ELEVENLABS_API_KEY
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required")
        
        self.base_url = "https://api.elevenlabs.io/v1"
    
    async def stream_synthesis(
        self, 
        request: TTSRequest
    ) -> AsyncIterator[AudioChunk]:
        """使用ElevenLabs进行流式语音合成"""
        session_id = request.session_id
        self._register_session(session_id)
        
        try:
            import aiohttp
            
            voice_id = request.voice_id or "21m00Tcm4TlvDq8ikWAM"  # 默认声音
            url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            payload = {
                "text": request.text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            "ElevenLabs TTS API error",
                            session_id=session_id,
                            status=response.status,
                            error=error_text
                        )
                        return
                    
                    # 流式读取音频数据
                    chunk_size = 4096
                    timestamp_ms = 0
                    chunk_index = 0
                    
                    async for data in response.content.iter_chunked(chunk_size):
                        # 检查是否被取消
                        if self._is_cancelled(session_id):
                            logger.info("ElevenLabs TTS synthesis cancelled", session_id=session_id)
                            break
                        
                        if data:
                            chunk = AudioChunk(
                                data=data,
                                timestamp_ms=timestamp_ms,
                                sample_rate=22050,  # ElevenLabs默认采样率
                                channels=1,
                                format="mp3"
                            )
                            
                            yield chunk
                            
                            # 估算时间戳增量
                            timestamp_ms += len(data) * 1000 // (22050 * 2)
                            chunk_index += 1
                            
                            logger.debug(
                                "ElevenLabs TTS chunk generated",
                                session_id=session_id,
                                chunk_index=chunk_index,
                                size=len(data)
                            )
        
        except Exception as e:
            logger.error(
                "Error in ElevenLabs TTS synthesis",
                session_id=session_id,
                error=str(e)
            )
        finally:
            self._cleanup_session(session_id)
