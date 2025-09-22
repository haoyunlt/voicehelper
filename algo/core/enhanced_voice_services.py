"""
增强的语音服务实现
支持多提供商、降级、缓存、VAD等功能
"""

import asyncio
import time
import json
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .voice_providers import (
    VoiceProvider, VoiceProviderFactory, 
    BaseASRProvider, BaseTTSProvider
)

logger = logging.getLogger(__name__)

@dataclass
class VoiceConfig:
    """语音服务配置"""
    # ASR配置
    primary_asr_provider: VoiceProvider = VoiceProvider.OPENAI
    fallback_asr_providers: List[VoiceProvider] = None
    asr_language: str = "zh-CN"
    asr_timeout: float = 10.0
    
    # TTS配置
    primary_tts_provider: VoiceProvider = VoiceProvider.EDGE_TTS
    fallback_tts_providers: List[VoiceProvider] = None
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    tts_language: str = "zh-CN"
    tts_timeout: float = 15.0
    
    # VAD配置
    enable_vad: bool = True
    vad_aggressiveness: int = 2  # 0-3, 3最激进
    min_speech_duration: float = 0.5  # 最小语音时长
    max_silence_duration: float = 2.0  # 最大静音时长
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 缓存TTL（秒）
    
    # 提供商配置
    provider_configs: Dict[VoiceProvider, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.fallback_asr_providers is None:
            self.fallback_asr_providers = [VoiceProvider.AZURE, VoiceProvider.LOCAL]
        if self.fallback_tts_providers is None:
            self.fallback_tts_providers = [VoiceProvider.AZURE, VoiceProvider.OPENAI]
        if self.provider_configs is None:
            self.provider_configs = {}

class VADProcessor:
    """语音活动检测处理器"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.vad = None
        
        if config.enable_vad:
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(config.vad_aggressiveness)
            except ImportError:
                logger.warning("webrtcvad not available, VAD disabled")
    
    def is_speech(self, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """检测音频是否包含语音"""
        if not self.vad:
            return True  # 如果VAD不可用，假设都是语音
        
        try:
            # webrtcvad要求特定的音频格式
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]  # 确保偶数长度
            
            # 检查音频长度（webrtcvad要求10ms、20ms或30ms的帧）
            frame_duration = 20  # ms
            frame_size = int(sample_rate * frame_duration / 1000) * 2  # 16-bit samples
            
            if len(audio_data) < frame_size:
                return False
            
            # 取前面的完整帧进行检测
            frame_data = audio_data[:frame_size]
            return self.vad.is_speech(frame_data, sample_rate)
            
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return True  # 出错时假设是语音

class VoiceCache:
    """语音缓存"""
    
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
    
    def _get_cache_key(self, text: str, provider: str, voice: str = None) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{text}:{provider}:{voice or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, text: str, provider: str, voice: str = None) -> Optional[bytes]:
        """获取缓存的音频"""
        key = self._get_cache_key(text, provider, voice)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        
        return None
    
    def set(self, text: str, provider: str, audio_data: bytes, voice: str = None):
        """设置缓存"""
        key = self._get_cache_key(text, provider, voice)
        self.cache[key] = {
            "data": audio_data,
            "timestamp": time.time()
        }
    
    def clear_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry["timestamp"] >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]

class EnhancedASRService:
    """增强的ASR服务"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.vad_processor = VADProcessor(config)
        
        # 初始化提供商
        self.providers: Dict[VoiceProvider, BaseASRProvider] = {}
        self._init_providers()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "provider_usage": {},
            "fallback_usage": 0
        }
    
    def _init_providers(self):
        """初始化ASR提供商"""
        all_providers = [self.config.primary_asr_provider] + self.config.fallback_asr_providers
        
        for provider in all_providers:
            try:
                provider_config = self.config.provider_configs.get(provider, {})
                self.providers[provider] = VoiceProviderFactory.create_asr_provider(
                    provider, provider_config
                )
                logger.info(f"Initialized ASR provider: {provider.value}")
            except Exception as e:
                logger.error(f"Failed to initialize ASR provider {provider.value}: {e}")
    
    async def transcribe(self, audio_data: bytes, language: str = None, 
                        is_final: bool = False, session_id: str = "") -> Optional[str]:
        """转写音频（支持降级）"""
        self.stats["total_requests"] += 1
        language = language or self.config.asr_language
        
        # VAD检测
        if self.config.enable_vad and not self.vad_processor.is_speech(audio_data):
            logger.debug("No speech detected by VAD")
            return None
        
        # 尝试主要提供商
        providers_to_try = [self.config.primary_asr_provider] + self.config.fallback_asr_providers
        
        for i, provider in enumerate(providers_to_try):
            if provider not in self.providers:
                continue
            
            try:
                start_time = time.time()
                
                # 设置超时
                result = await asyncio.wait_for(
                    self.providers[provider].transcribe(audio_data, language, is_final),
                    timeout=self.config.asr_timeout
                )
                
                if result:
                    # 更新统计
                    self.stats["successful_requests"] += 1
                    self.stats["provider_usage"][provider.value] = \
                        self.stats["provider_usage"].get(provider.value, 0) + 1
                    
                    if i > 0:  # 使用了降级
                        self.stats["fallback_usage"] += 1
                    
                    latency = time.time() - start_time
                    logger.debug(f"ASR success with {provider.value}, latency: {latency:.3f}s")
                    
                    return result
                
            except asyncio.TimeoutError:
                logger.warning(f"ASR timeout with {provider.value}")
            except Exception as e:
                logger.error(f"ASR error with {provider.value}: {e}")
        
        logger.error("All ASR providers failed")
        return None
    
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None], 
                                 language: str = None, session_id: str = "") -> AsyncGenerator[str, None]:
        """流式转写音频"""
        language = language or self.config.asr_language
        
        # 使用主要提供商进行流式转写
        provider = self.config.primary_asr_provider
        if provider not in self.providers:
            logger.error(f"Primary ASR provider {provider.value} not available")
            return
        
        try:
            async for result in self.providers[provider].transcribe_streaming(audio_stream, language):
                if result:
                    yield result
        except Exception as e:
            logger.error(f"Streaming ASR error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

class EnhancedTTSService:
    """增强的TTS服务"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.cache = VoiceCache(config.cache_ttl) if config.enable_cache else None
        
        # 初始化提供商
        self.providers: Dict[VoiceProvider, BaseTTSProvider] = {}
        self._init_providers()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "cache_hits": 0,
            "provider_usage": {},
            "fallback_usage": 0
        }
    
    def _init_providers(self):
        """初始化TTS提供商"""
        all_providers = [self.config.primary_tts_provider] + self.config.fallback_tts_providers
        
        for provider in all_providers:
            try:
                provider_config = self.config.provider_configs.get(provider, {})
                self.providers[provider] = VoiceProviderFactory.create_tts_provider(
                    provider, provider_config
                )
                logger.info(f"Initialized TTS provider: {provider.value}")
            except Exception as e:
                logger.error(f"Failed to initialize TTS provider {provider.value}: {e}")
    
    async def synthesize(self, text: str, voice: str = None, 
                        language: str = None) -> bytes:
        """合成语音（支持缓存和降级）"""
        self.stats["total_requests"] += 1
        voice = voice or self.config.tts_voice
        language = language or self.config.tts_language
        
        # 检查缓存
        if self.cache:
            cached_audio = self.cache.get(text, self.config.primary_tts_provider.value, voice)
            if cached_audio:
                self.stats["cache_hits"] += 1
                self.stats["successful_requests"] += 1
                logger.debug("TTS cache hit")
                return cached_audio
        
        # 尝试提供商
        providers_to_try = [self.config.primary_tts_provider] + self.config.fallback_tts_providers
        
        for i, provider in enumerate(providers_to_try):
            if provider not in self.providers:
                continue
            
            try:
                start_time = time.time()
                
                result = await asyncio.wait_for(
                    self.providers[provider].synthesize(text, voice, language),
                    timeout=self.config.tts_timeout
                )
                
                if result:
                    # 更新统计
                    self.stats["successful_requests"] += 1
                    self.stats["provider_usage"][provider.value] = \
                        self.stats["provider_usage"].get(provider.value, 0) + 1
                    
                    if i > 0:  # 使用了降级
                        self.stats["fallback_usage"] += 1
                    
                    # 缓存结果
                    if self.cache:
                        self.cache.set(text, provider.value, result, voice)
                    
                    latency = time.time() - start_time
                    logger.debug(f"TTS success with {provider.value}, latency: {latency:.3f}s")
                    
                    return result
                
            except asyncio.TimeoutError:
                logger.warning(f"TTS timeout with {provider.value}")
            except Exception as e:
                logger.error(f"TTS error with {provider.value}: {e}")
        
        logger.error("All TTS providers failed")
        return b""
    
    async def synthesize_streaming(self, text: str, voice: str = None, 
                                 language: str = None) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        voice = voice or self.config.tts_voice
        language = language or self.config.tts_language
        
        # 使用主要提供商进行流式合成
        provider = self.config.primary_tts_provider
        if provider not in self.providers:
            logger.error(f"Primary TTS provider {provider.value} not available")
            return
        
        try:
            async for chunk in self.providers[provider].synthesize_streaming(text, voice, language):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def clear_cache(self):
        """清理缓存"""
        if self.cache:
            self.cache.clear_expired()

class EnhancedVoiceService:
    """增强的语音服务（整合ASR和TTS）"""
    
    def __init__(self, config: VoiceConfig, retrieve_service=None):
        self.config = config
        self.retrieve_service = retrieve_service
        
        # 初始化ASR和TTS服务
        self.asr_service = EnhancedASRService(config)
        self.tts_service = EnhancedTTSService(config)
        
        # 会话管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Enhanced voice service initialized")
    
    async def process_voice_query(self, request) -> AsyncGenerator:
        """处理语音查询（复用原有接口）"""
        try:
            session_id = request.conversation_id
            
            # 初始化会话
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "audio_buffer": b"",
                    "transcript_buffer": "",
                    "last_activity": time.time()
                }
            
            session = self.active_sessions[session_id]
            
            # 解码音频数据
            import base64
            audio_chunk = base64.b64decode(request.audio_chunk)
            session["audio_buffer"] += audio_chunk
            session["last_activity"] = time.time()
            
            # ASR处理
            if len(session["audio_buffer"]) > 8000:  # 约0.5秒的音频
                # 使用增强的ASR服务
                partial_text = await self.asr_service.transcribe(
                    session["audio_buffer"], 
                    is_final=False, 
                    session_id=session_id
                )
                
                if partial_text:
                    from core.models import VoiceQueryResponse
                    yield VoiceQueryResponse(
                        type="asr_partial",
                        seq=request.seq,
                        text=partial_text
                    )
                    
                # 检查是否为完整句子
                if partial_text and self._is_complete_sentence(partial_text):
                    # 最终识别
                    final_text = await self.asr_service.transcribe(
                        session["audio_buffer"], 
                        is_final=True, 
                        session_id=session_id
                    )
                    
                    if final_text:
                        from core.models import VoiceQueryResponse
                        yield VoiceQueryResponse(
                            type="asr_final",
                            seq=request.seq,
                            text=final_text
                        )
                        
                        # 处理RAG查询
                        if self.retrieve_service:
                            async for response in self._process_rag_query(final_text, session_id):
                                yield response
                    
                    # 清空缓冲区
                    session["audio_buffer"] = b""
                    session["transcript_buffer"] = ""
                    
        except Exception as e:
            from core.models import VoiceQueryResponse
            yield VoiceQueryResponse(
                type="error",
                error=f"Voice processing error: {str(e)}"
            )
    
    async def _process_rag_query(self, query: str, session_id: str) -> AsyncGenerator:
        """处理RAG查询并生成语音"""
        try:
            from core.models import QueryRequest, Message, VoiceQueryResponse, Reference
            
            query_request = QueryRequest(
                messages=[Message(role="user", content=query)],
                top_k=5,
                temperature=0.3
            )
            
            full_response = ""
            references = []
            
            async for response in self.retrieve_service.stream_query(query_request):
                response_data = json.loads(response)
                
                if response_data["type"] == "refs" and response_data.get("refs"):
                    references = response_data["refs"]
                    yield VoiceQueryResponse(
                        type="refs",
                        refs=[Reference(**ref) for ref in references]
                    )
                
                elif response_data["type"] == "delta" and response_data.get("content"):
                    content = response_data["content"]
                    full_response += content
                    
                    yield VoiceQueryResponse(
                        type="llm_delta",
                        text=content
                    )
                    
                    # 检查完整句子并进行TTS
                    if self._is_complete_sentence(full_response):
                        sentence = self._extract_last_sentence(full_response)
                        if sentence:
                            async for tts_response in self._synthesize_and_stream(sentence, session_id):
                                yield tts_response
            
            # 处理剩余文本
            if full_response and not self._is_complete_sentence(full_response):
                async for tts_response in self._synthesize_and_stream(full_response, session_id):
                    yield tts_response
            
            yield VoiceQueryResponse(type="done")
            
        except Exception as e:
            yield VoiceQueryResponse(
                type="error",
                error=f"RAG processing error: {str(e)}"
            )
    
    async def _synthesize_and_stream(self, text: str, session_id: str = "") -> AsyncGenerator:
        """合成语音并流式返回"""
        try:
            from core.models import VoiceQueryResponse
            import base64
            
            # 语音友好化处理
            voice_text = self._make_voice_friendly(text)
            
            # 使用增强的TTS服务进行流式合成
            seq = 0
            async for audio_chunk in self.tts_service.synthesize_streaming(voice_text):
                if audio_chunk:
                    # 转换为PCM格式并编码
                    pcm_data = self._convert_to_pcm(audio_chunk)
                    if pcm_data:
                        yield VoiceQueryResponse(
                            type="tts_chunk",
                            seq=seq,
                            pcm=base64.b64encode(pcm_data).decode('utf-8')
                        )
                        seq += 1
                        
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
    
    def _make_voice_friendly(self, text: str) -> str:
        """将文本转换为语音友好格式"""
        import re
        text = re.sub(r'\[\d+\]', '', text)
        text = text.replace('根据检索到的信息', '根据资料')
        text = text.replace('基于以上内容', '综合来看')
        text = text.replace('。', '。 ')
        text = text.replace('！', '！ ')
        text = text.replace('？', '？ ')
        return text.strip()
    
    def _is_complete_sentence(self, text: str) -> bool:
        """判断是否为完整句子"""
        return text.endswith(('。', '！', '？', '.', '!', '?'))
    
    def _extract_last_sentence(self, text: str) -> str:
        """提取最后一个完整句子"""
        import re
        sentences = re.split(r'[。！？.!?]', text)
        if len(sentences) >= 2:
            return sentences[-2] + text[-1]
        return ""
    
    def _convert_to_pcm(self, audio_data: bytes) -> Optional[bytes]:
        """转换音频为PCM格式"""
        try:
            # 使用pydub进行音频格式转换
            from pydub import AudioSegment
            import io
            
            # 假设输入是MP3格式
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            
            # 转换为PCM格式（16kHz, 16-bit, mono）
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            return audio.raw_data
            
        except Exception as e:
            logger.debug(f"Audio conversion error: {e}")
            return audio_data  # 返回原始数据作为降级
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "asr_stats": self.asr_service.get_stats(),
            "tts_stats": self.tts_service.get_stats(),
            "active_sessions": len(self.active_sessions)
        }
    
    def cleanup_inactive_sessions(self):
        """清理非活跃会话"""
        current_time = time.time()
        inactive_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session["last_activity"] > 300:  # 5分钟超时
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            del self.active_sessions[session_id]
        
        # 清理TTS缓存
        self.tts_service.clear_cache()
