"""
OpenAI ASR/TTS适配器
基于V2架构的OpenAI Whisper和TTS实现
"""

import io
import json
import time
import asyncio
from typing import Iterator, Optional, Dict, Any
import openai
from loguru import logger

from .base import BaseAsrAdapter, BaseTtsAdapter


class OpenAIAsrAdapter(BaseAsrAdapter):
    """OpenAI Whisper ASR适配器"""
    
    provider: str = "openai"
    model: str = "whisper-1"
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = openai.OpenAI(api_key=api_key)
        self._audio_buffer = io.BytesIO()
        self._buffer_size = 0
        self._min_chunk_size = 1024 * 16  # 16KB最小块
    
    def start(self, sr: int, codec: str, lang: str) -> str:
        """
        开始ASR会话
        
        Args:
            sr: 采样率
            codec: 音频编码
            lang: 语言代码
            
        Returns:
            会话ID
        """
        self.sample_rate = sr
        self.language = lang
        self._session_id = f"openai_asr_{int(time.time())}"
        self._is_active = True
        self._audio_buffer = io.BytesIO()
        self._buffer_size = 0
        
        logger.info(f"OpenAI ASR会话开始: {self._session_id}, sr={sr}, lang={lang}")
        return self._session_id
    
    def feed(self, seq: int, chunk: bytes) -> None:
        """
        输入音频数据
        
        Args:
            seq: 序列号
            chunk: 音频数据块
        """
        if not self._is_active:
            return
        
        # 累积音频数据
        self._audio_buffer.write(chunk)
        self._buffer_size += len(chunk)
        
        # 当缓冲区达到一定大小时进行识别
        if self._buffer_size >= self._min_chunk_size:
            try:
                self._process_audio_chunk(seq)
            except Exception as e:
                logger.error(f"OpenAI ASR处理失败: {e}")
    
    def stop(self) -> None:
        """停止ASR会话"""
        if not self._is_active:
            return
        
        self._is_active = False
        
        # 处理剩余音频
        if self._buffer_size > 0:
            try:
                self._process_audio_chunk(-1, is_final=True)
            except Exception as e:
                logger.error(f"OpenAI ASR最终处理失败: {e}")
        
        logger.info(f"OpenAI ASR会话结束: {self._session_id}")
    
    def _process_audio_chunk(self, seq: int, is_final: bool = False):
        """处理音频块"""
        if self._buffer_size == 0:
            return
        
        # 获取音频数据
        audio_data = self._audio_buffer.getvalue()
        
        # 重置缓冲区
        self._audio_buffer = io.BytesIO()
        self._buffer_size = 0
        
        def _transcribe():
            # 创建临时音频文件
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"  # Whisper需要文件名
            
            # 调用Whisper API
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language if self.language != "zh-CN" else "zh",
                response_format="verbose_json"
            )
            
            return response
        
        try:
            # 使用重试机制
            response = self._retry(_transcribe)
            
            text = response.text.strip()
            confidence = getattr(response, 'confidence', 0.9)
            
            if text:
                if is_final:
                    self._emit_final(seq, text, confidence)
                else:
                    self._emit_partial(seq, text)
                
                logger.debug(f"OpenAI ASR结果: seq={seq}, text='{text}', confidence={confidence}")
            
        except Exception as e:
            logger.error(f"OpenAI Whisper API调用失败: {e}")
            raise


class OpenAITtsAdapter(BaseTtsAdapter):
    """OpenAI TTS适配器"""
    
    provider: str = "openai"
    model: str = "tts-1"
    voice: str = "alloy"
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = openai.OpenAI(api_key=api_key)
    
    def synthesize(self, text_iter: Iterator[str]) -> Iterator[bytes]:
        """
        流式文本转语音
        
        Args:
            text_iter: 文本迭代器
            
        Yields:
            音频数据块
        """
        request_id = self._generate_request_id()
        
        try:
            for text in text_iter:
                if not text.strip():
                    continue
                
                logger.debug(f"OpenAI TTS合成: '{text[:50]}...'")
                
                def _synthesize():
                    response = self.client.audio.speech.create(
                        model=self.model,
                        voice=self.voice,
                        input=text,
                        response_format="mp3",
                        speed=self.speed
                    )
                    return response.content
                
                # 使用重试机制
                audio_data = self._retry(_synthesize)
                
                if audio_data:
                    # 分块返回音频数据
                    chunk_size = 1024 * 4  # 4KB chunks
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        yield chunk
                
        except Exception as e:
            logger.error(f"OpenAI TTS合成失败: {e}")
            raise
        finally:
            self._untrack_request(request_id)
    
    def cancel(self, request_id: str) -> None:
        """
        取消TTS请求
        
        Args:
            request_id: 请求ID
        """
        # OpenAI TTS API不支持取消，只能从跟踪中移除
        self._untrack_request(request_id)
        logger.info(f"OpenAI TTS请求已取消: {request_id}")
