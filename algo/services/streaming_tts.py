"""
流式TTS服务 - 可中断的文本转语音服务
功能: 分句合成 + 实时取消 + 多提供商支持 + 流式输出
"""

import asyncio
import time
import re
import logging
from typing import AsyncIterator, Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json
import base64

from ..core.events import EventStream, EventType, emit_tts_chunk

logger = logging.getLogger(__name__)

class TTSProvider(Enum):
    """TTS提供商枚举"""
    OPENAI = "openai"
    DEEPGRAM = "deepgram"
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"
    LOCAL = "local"

@dataclass
class TTSConfig:
    """TTS配置"""
    provider: TTSProvider
    api_key: str
    voice: str = "alloy"
    model: str = "tts-1"
    format: str = "pcm16"
    sample_rate: int = 16000
    speed: float = 1.0
    chunk_size: int = 1024
    max_sentence_length: int = 200
    sentence_overlap: int = 10
    timeout: float = 30.0

@dataclass
class TTSChunk:
    """TTS音频块"""
    audio_data: bytes
    chunk_id: int
    format: str
    timestamp: int
    is_final: bool = False

class TTSProviderBase(ABC):
    """TTS提供商基类"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        
    @abstractmethod
    async def synthesize_streaming(self, text: str, session_id: str) -> AsyncIterator[TTSChunk]:
        """流式合成语音"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

class OpenAITTSProvider(TTSProviderBase):
    """OpenAI TTS提供商"""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.base_url = "https://api.openai.com/v1"
        
    async def synthesize_streaming(self, text: str, session_id: str) -> AsyncIterator[TTSChunk]:
        """OpenAI流式TTS"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "input": text,
            "voice": self.config.voice,
            "response_format": self.config.format,
            "speed": self.config.speed
        }
        
        chunk_id = 0
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/audio/speech",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI TTS API error: {response.status} - {error_text}")
                
                async for chunk in response.content.iter_chunked(self.config.chunk_size):
                    if chunk:
                        yield TTSChunk(
                            audio_data=chunk,
                            chunk_id=chunk_id,
                            format=self.config.format,
                            timestamp=int(time.time() * 1000)
                        )
                        chunk_id += 1
                
                # 发送最终块
                yield TTSChunk(
                    audio_data=b"",
                    chunk_id=chunk_id,
                    format=self.config.format,
                    timestamp=int(time.time() * 1000),
                    is_final=True
                )
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"OpenAI TTS health check failed: {e}")
            return False

class DeepgramTTSProvider(TTSProviderBase):
    """Deepgram TTS提供商"""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.base_url = "https://api.deepgram.com/v1/speak"
        
    async def synthesize_streaming(self, text: str, session_id: str) -> AsyncIterator[TTSChunk]:
        """Deepgram流式TTS"""
        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "model": self.config.voice,
            "encoding": self.config.format,
            "sample_rate": self.config.sample_rate
        }
        
        payload = {"text": text}
        chunk_id = 0
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=headers,
                params=params,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Deepgram TTS API error: {response.status} - {error_text}")
                
                async for chunk in response.content.iter_chunked(self.config.chunk_size):
                    if chunk:
                        yield TTSChunk(
                            audio_data=chunk,
                            chunk_id=chunk_id,
                            format=self.config.format,
                            timestamp=int(time.time() * 1000)
                        )
                        chunk_id += 1
                
                yield TTSChunk(
                    audio_data=b"",
                    chunk_id=chunk_id,
                    format=self.config.format,
                    timestamp=int(time.time() * 1000),
                    is_final=True
                )
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            headers = {"Authorization": f"Token {self.config.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.deepgram.com/v1/projects",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Deepgram TTS health check failed: {e}")
            return False

class StreamingTTSService:
    """流式TTS服务主类"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.provider = self._create_provider()
        self.active_sessions: Dict[str, bool] = {}
        self.cancellation_tokens: Dict[str, bool] = {}
        self._lock = asyncio.Lock()
        
    def _create_provider(self) -> TTSProviderBase:
        """创建TTS提供商实例"""
        if self.config.provider == TTSProvider.OPENAI:
            return OpenAITTSProvider(self.config)
        elif self.config.provider == TTSProvider.DEEPGRAM:
            return DeepgramTTSProvider(self.config)
        else:
            raise ValueError(f"Unsupported TTS provider: {self.config.provider}")
    
    async def synthesize_streaming(self, text: str, session_id: str, 
                                 event_stream: Optional[EventStream] = None) -> AsyncIterator[TTSChunk]:
        """流式合成语音"""
        async with self._lock:
            self.active_sessions[session_id] = True
            self.cancellation_tokens[session_id] = False
        
        try:
            # 发射TTS开始事件
            if event_stream:
                await event_stream.emit(EventType.TTS_START, {
                    "text": text,
                    "provider": self.config.provider.value,
                    "voice": self.config.voice,
                    "format": self.config.format,
                    "timestamp": int(time.time() * 1000)
                })
            
            # 分句处理
            sentences = self._split_sentences(text)
            logger.info(f"Split text into {len(sentences)} sentences for session {session_id}")
            
            total_chunks = 0
            
            for i, sentence in enumerate(sentences):
                if await self._is_cancelled(session_id):
                    logger.info(f"TTS cancelled for session {session_id} at sentence {i}")
                    if event_stream:
                        await event_stream.emit(EventType.TTS_CANCELLED, {
                            "reason": "user_cancelled",
                            "processed_sentences": i,
                            "total_sentences": len(sentences),
                            "timestamp": int(time.time() * 1000)
                        })
                    break
                
                logger.debug(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
                
                # 合成单个句子
                sentence_start_time = time.time()
                sentence_chunks = 0
                
                try:
                    async for chunk in self.provider.synthesize_streaming(sentence, session_id):
                        if await self._is_cancelled(session_id):
                            break
                        
                        # 更新块ID为全局序号
                        chunk.chunk_id = total_chunks
                        total_chunks += 1
                        sentence_chunks += 1
                        
                        # 发射TTS块事件
                        if event_stream:
                            await event_stream.emit(EventType.TTS_CHUNK, {
                                "chunk_id": chunk.chunk_id,
                                "sentence_index": i,
                                "format": chunk.format,
                                "size": len(chunk.audio_data),
                                "is_final": chunk.is_final,
                                "timestamp": chunk.timestamp
                            })
                        
                        yield chunk
                        
                        # 检查取消状态
                        if await self._is_cancelled(session_id):
                            break
                
                except Exception as e:
                    logger.error(f"Error synthesizing sentence {i}: {e}")
                    if event_stream:
                        await event_stream.emit_error(
                            EventType.TTS_ERROR,
                            "synthesis_error",
                            f"Failed to synthesize sentence {i}: {str(e)}"
                        )
                    continue
                
                sentence_duration = time.time() - sentence_start_time
                logger.debug(f"Sentence {i+1} completed: {sentence_chunks} chunks in {sentence_duration:.2f}s")
            
            # 发射TTS结束事件
            if event_stream and not await self._is_cancelled(session_id):
                await event_stream.emit(EventType.TTS_END, {
                    "total_chunks": total_chunks,
                    "total_sentences": len(sentences),
                    "timestamp": int(time.time() * 1000)
                })
                
        except Exception as e:
            logger.error(f"TTS synthesis error for session {session_id}: {e}")
            if event_stream:
                await event_stream.emit_error(
                    EventType.TTS_ERROR,
                    "synthesis_error",
                    str(e)
                )
            raise
        
        finally:
            # 清理会话状态
            async with self._lock:
                self.active_sessions.pop(session_id, None)
                self.cancellation_tokens.pop(session_id, None)
    
    async def cancel_session(self, session_id: str) -> bool:
        """取消指定会话的TTS合成"""
        async with self._lock:
            if session_id in self.active_sessions:
                self.cancellation_tokens[session_id] = True
                logger.info(f"TTS cancellation requested for session {session_id}")
                return True
            return False
    
    async def _is_cancelled(self, session_id: str) -> bool:
        """检查会话是否已取消"""
        return self.cancellation_tokens.get(session_id, False)
    
    def _split_sentences(self, text: str) -> List[str]:
        """智能分句"""
        # 基本的句子分割正则表达式
        sentence_endings = r'[.!?。！？]+'
        sentences = re.split(sentence_endings, text)
        
        # 清理和过滤
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 如果句子太长，按逗号进一步分割
                if len(sentence) > self.config.max_sentence_length:
                    sub_sentences = re.split(r'[,，;；]', sentence)
                    for sub_sentence in sub_sentences:
                        sub_sentence = sub_sentence.strip()
                        if sub_sentence:
                            cleaned_sentences.append(sub_sentence)
                else:
                    cleaned_sentences.append(sentence)
        
        # 如果没有分割出句子，返回原文本
        if not cleaned_sentences:
            cleaned_sentences = [text]
        
        return cleaned_sentences
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        provider_healthy = await self.provider.health_check()
        
        return {
            "provider": self.config.provider.value,
            "healthy": provider_healthy,
            "active_sessions": len(self.active_sessions),
            "config": {
                "voice": self.config.voice,
                "model": self.config.model,
                "format": self.config.format,
                "sample_rate": self.config.sample_rate
            },
            "timestamp": int(time.time() * 1000)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.config.provider.value,
            "active_sessions": len(self.active_sessions),
            "active_session_ids": list(self.active_sessions.keys()),
            "cancelled_sessions": sum(1 for cancelled in self.cancellation_tokens.values() if cancelled),
            "config": {
                "voice": self.config.voice,
                "model": self.config.model,
                "format": self.config.format,
                "sample_rate": self.config.sample_rate,
                "chunk_size": self.config.chunk_size
            }
        }

# 工厂函数
def create_tts_service(provider: Union[str, TTSProvider], api_key: str, 
                      voice: str = "alloy", **kwargs) -> StreamingTTSService:
    """创建TTS服务实例"""
    if isinstance(provider, str):
        provider = TTSProvider(provider)
    
    config = TTSConfig(
        provider=provider,
        api_key=api_key,
        voice=voice,
        **kwargs
    )
    
    return StreamingTTSService(config)

# 使用示例
async def example_usage():
    """使用示例"""
    from ..core.events import create_event_stream
    
    # 创建TTS服务
    tts_service = create_tts_service(
        provider="openai",
        api_key="your-api-key",
        voice="alloy",
        format="pcm16",
        sample_rate=16000
    )
    
    # 创建事件流
    session_id = "example_session"
    event_stream = create_event_stream(session_id)
    
    # 合成语音
    text = "这是一个测试文本。我们将把它转换为语音。这个过程是可以被中断的。"
    
    try:
        async for chunk in tts_service.synthesize_streaming(text, session_id, event_stream):
            print(f"Received chunk {chunk.chunk_id}: {len(chunk.audio_data)} bytes")
            
            # 模拟处理延迟
            await asyncio.sleep(0.1)
            
            # 模拟中断条件
            if chunk.chunk_id > 5:
                await tts_service.cancel_session(session_id)
                
    except Exception as e:
        print(f"TTS error: {e}")

if __name__ == "__main__":
    asyncio.run(example_usage())
