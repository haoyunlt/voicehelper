"""
统一语音处理系统

整合VoiceService、VoiceOptimizer和EnhancedVoiceOptimizer的功能
提供150ms超低延迟的语音处理能力
"""

import asyncio
import base64
import json
import time
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

import numpy as np

# 导入统一工具类
from .unified_utils import (
    get_cache_manager,
    get_content_normalizer,
    UnifiedCacheManager
)

logger = logging.getLogger(__name__)


class VoiceProcessingMode(Enum):
    """语音处理模式"""
    STANDARD = "standard"      # 标准模式 (300ms)
    OPTIMIZED = "optimized"    # 优化模式 (150ms)
    ENHANCED = "enhanced"      # 增强模式 (120ms)


@dataclass
class VoiceConfig:
    """语音处理配置"""
    mode: VoiceProcessingMode = VoiceProcessingMode.ENHANCED
    target_latency: float = 120.0  # 目标延迟(ms)
    parallel_workers: int = 4
    enable_cache_prewarming: bool = True
    enable_pipeline_fusion: bool = True
    enable_emotion_detection: bool = True
    sample_rate: int = 16000
    chunk_size: int = 1024


@dataclass
class VoiceMetrics:
    """语音处理指标"""
    asr_latency: float = 0.0
    llm_latency: float = 0.0
    tts_latency: float = 0.0
    total_latency: float = 0.0
    pipeline_efficiency: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class VoiceRequest:
    """统一语音请求"""
    conversation_id: str
    audio_chunk: bytes
    seq: int = 0
    is_final: bool = False
    codec: str = "opus"
    sample_rate: int = 16000
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoiceResponse:
    """统一语音响应"""
    type: str  # asr_partial, asr_final, llm_response, tts_audio, error
    seq: int = 0
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    references: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    metrics: Optional[VoiceMetrics] = None


class UnifiedASRProcessor:
    """统一ASR处理器"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.audio_buffer = deque(maxlen=1000)
        self.processing_queue = asyncio.Queue()
        self.cache_manager = get_cache_manager()
        
        # ASR优化参数
        self.vad_threshold = 0.5
        self.silence_timeout = 1.0
        
    async def process_audio(self, audio_chunk: bytes, is_final: bool = False) -> Tuple[str, bool, float]:
        """
        处理音频数据
        
        Returns:
            (text, is_complete, confidence)
        """
        start_time = time.time()
        
        try:
            # 1. 音频预处理和VAD
            if not self._is_speech(audio_chunk):
                return "", False, 0.0
            
            # 2. 智能分块处理
            if self.config.mode == VoiceProcessingMode.ENHANCED:
                text, confidence = await self._enhanced_asr_processing(audio_chunk, is_final)
            elif self.config.mode == VoiceProcessingMode.OPTIMIZED:
                text, confidence = await self._optimized_asr_processing(audio_chunk, is_final)
            else:
                text, confidence = await self._standard_asr_processing(audio_chunk, is_final)
            
            # 3. 检查是否为完整句子
            is_complete = self._is_complete_sentence(text) or is_final
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"ASR processing time: {processing_time:.2f}ms")
            
            return text, is_complete, confidence
            
        except Exception as e:
            logger.error(f"ASR processing error: {e}")
            return "", False, 0.0
    
    def _is_speech(self, audio_chunk: bytes) -> bool:
        """语音活动检测 (VAD)"""
        # 简化的VAD实现
        if len(audio_chunk) < 100:
            return False
        
        # 计算音频能量
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = np.mean(np.abs(audio_array))
        
        return energy > self.vad_threshold * 1000
    
    async def _enhanced_asr_processing(self, audio_chunk: bytes, is_final: bool) -> Tuple[str, float]:
        """增强ASR处理 - 120ms目标"""
        # 并行处理多个音频块
        chunks = self._split_audio_intelligently(audio_chunk)
        
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._process_audio_chunk(chunk))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        text_parts = []
        confidences = []
        
        for result in results:
            if isinstance(result, tuple):
                text, conf = result
                if text:
                    text_parts.append(text)
                    confidences.append(conf)
        
        final_text = " ".join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return final_text, avg_confidence
    
    async def _optimized_asr_processing(self, audio_chunk: bytes, is_final: bool) -> Tuple[str, float]:
        """优化ASR处理 - 150ms目标"""
        # 流式处理
        return await self._process_audio_chunk(audio_chunk)
    
    async def _standard_asr_processing(self, audio_chunk: bytes, is_final: bool) -> Tuple[str, float]:
        """标准ASR处理 - 300ms"""
        return await self._process_audio_chunk(audio_chunk)
    
    def _split_audio_intelligently(self, audio_chunk: bytes) -> List[bytes]:
        """智能音频分块"""
        chunk_size = len(audio_chunk) // self.config.parallel_workers
        chunks = []
        
        for i in range(0, len(audio_chunk), chunk_size):
            chunk = audio_chunk[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks
    
    async def _process_audio_chunk(self, chunk: bytes) -> Tuple[str, float]:
        """处理单个音频块"""
        # 模拟ASR处理
        await asyncio.sleep(0.05)  # 50ms处理时间
        
        # 这里应该调用实际的ASR服务
        # 暂时返回模拟结果
        text = f"识别文本_{len(chunk)}"
        confidence = 0.95
        
        return text, confidence
    
    def _is_complete_sentence(self, text: str) -> bool:
        """判断是否为完整句子"""
        if not text:
            return False
        
        # 检查句子结束标点
        end_punctuation = ['。', '！', '？', '.', '!', '?']
        return any(text.strip().endswith(punct) for punct in end_punctuation)


class UnifiedTTSProcessor:
    """统一TTS处理器"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.cache_manager = get_cache_manager()
        self.emotion_detector = None
        
        if config.enable_emotion_detection:
            self.emotion_detector = EmotionDetector()
    
    async def synthesize_text(self, text: str, emotion: Optional[str] = None) -> Tuple[bytes, float]:
        """
        合成语音
        
        Returns:
            (audio_data, processing_time_ms)
        """
        start_time = time.time()
        
        try:
            # 1. 检查缓存
            cache_key = f"tts:{text}:{emotion}"
            cached_audio = await self.cache_manager.get(cache_key)
            if cached_audio:
                processing_time = (time.time() - start_time) * 1000
                return cached_audio, processing_time
            
            # 2. 情感检测
            if self.emotion_detector and not emotion:
                emotion = await self.emotion_detector.detect_emotion(text)
            
            # 3. 智能句子分割
            sentences = self._split_text_intelligently(text)
            
            # 4. 并行合成
            if len(sentences) > 1 and self.config.mode == VoiceProcessingMode.ENHANCED:
                audio_data = await self._parallel_synthesize(sentences, emotion)
            else:
                audio_data = await self._synthesize_single(text, emotion)
            
            # 5. 缓存结果
            await self.cache_manager.put(cache_key, audio_data)
            
            processing_time = (time.time() - start_time) * 1000
            return audio_data, processing_time
            
        except Exception as e:
            logger.error(f"TTS processing error: {e}")
            return b"", 0.0
    
    def _split_text_intelligently(self, text: str) -> List[str]:
        """智能文本分割"""
        # 按句子分割
        import re
        sentences = re.split(r'[。！？.!?]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _parallel_synthesize(self, sentences: List[str], emotion: Optional[str] = None) -> bytes:
        """并行合成多个句子"""
        tasks = []
        for sentence in sentences:
            task = asyncio.create_task(self._synthesize_single(sentence, emotion))
            tasks.append(task)
        
        audio_chunks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并音频
        combined_audio = b""
        for chunk in audio_chunks:
            if isinstance(chunk, bytes):
                combined_audio += chunk
        
        return combined_audio
    
    async def _synthesize_single(self, text: str, emotion: Optional[str] = None) -> bytes:
        """合成单个文本"""
        # 模拟TTS处理
        await asyncio.sleep(0.08)  # 80ms处理时间
        
        # 这里应该调用实际的TTS服务
        # 暂时返回模拟音频数据
        audio_data = f"TTS_AUDIO_{text}_{emotion}".encode()
        
        return audio_data


class EmotionDetector:
    """情感检测器"""
    
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['开心', '高兴', '快乐', '兴奋'],
            'sad': ['难过', '伤心', '悲伤', '沮丧'],
            'angry': ['生气', '愤怒', '恼火', '气愤'],
            'neutral': ['好的', '知道', '明白', '了解']
        }
    
    async def detect_emotion(self, text: str) -> str:
        """检测文本情感"""
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emotion
        
        return 'neutral'


class UnifiedVoiceService:
    """统一语音处理服务"""
    
    def __init__(self, retrieve_service=None, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.retrieve_service = retrieve_service
        
        # 初始化处理器
        self.asr_processor = UnifiedASRProcessor(self.config)
        self.tts_processor = UnifiedTTSProcessor(self.config)
        self.cache_manager = get_cache_manager()
        
        # 会话管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # 性能监控
        self.metrics_history: List[VoiceMetrics] = []
    
    async def start(self):
        """启动语音服务"""
        await self.cache_manager.start()
        logger.info(f"Unified voice service started in {self.config.mode.value} mode")
    
    async def stop(self):
        """停止语音服务"""
        await self.cache_manager.stop()
        logger.info("Unified voice service stopped")
    
    async def process_voice_request(self, request: VoiceRequest) -> AsyncGenerator[VoiceResponse, None]:
        """处理语音请求"""
        session_id = request.conversation_id
        metrics = VoiceMetrics()
        total_start_time = time.time()
        
        try:
            # 初始化会话
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "audio_buffer": b"",
                    "transcript_buffer": "",
                    "last_activity": datetime.now(),
                    "context": []
                }
            
            session = self.active_sessions[session_id]
            
            # 解码音频数据
            audio_chunk = base64.b64decode(request.audio_chunk)
            session["audio_buffer"] += audio_chunk
            session["last_activity"] = datetime.now()
            
            # ASR处理
            asr_start = time.time()
            text, is_complete, confidence = await self.asr_processor.process_audio(
                session["audio_buffer"], 
                request.is_final
            )
            metrics.asr_latency = (time.time() - asr_start) * 1000
            
            if text:
                # 发送部分识别结果
                yield VoiceResponse(
                    type="asr_partial",
                    seq=request.seq,
                    text=text
                )
                
                # 如果是完整句子，进行后续处理
                if is_complete:
                    yield VoiceResponse(
                        type="asr_final",
                        seq=request.seq,
                        text=text
                    )
                    
                    # 并行处理：LLM推理 + 缓存预热
                    llm_start = time.time()
                    
                    # LLM处理
                    if self.retrieve_service:
                        async for response in self._process_llm_query(text, session_id):
                            yield response
                    
                    metrics.llm_latency = (time.time() - llm_start) * 1000
                    
                    # 清空缓冲区
                    session["audio_buffer"] = b""
                    session["transcript_buffer"] = ""
            
            # 更新总延迟
            metrics.total_latency = (time.time() - total_start_time) * 1000
            
            # 计算管道效率
            theoretical_min = metrics.asr_latency + metrics.llm_latency + metrics.tts_latency
            if theoretical_min > 0:
                metrics.pipeline_efficiency = theoretical_min / metrics.total_latency
            
            # 记录指标
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)
                    
        except Exception as e:
            yield VoiceResponse(
                type="error",
                error=f"Voice processing error: {str(e)}"
            )
    
    async def _process_llm_query(self, query: str, session_id: str) -> AsyncGenerator[VoiceResponse, None]:
        """处理LLM查询"""
        try:
            if not self.retrieve_service:
                # 模拟LLM响应
                await asyncio.sleep(0.1)
                response_text = f"这是对'{query}'的回复"
                
                yield VoiceResponse(
                    type="llm_response",
                    text=response_text
                )
                
                # TTS合成
                tts_start = time.time()
                audio_data, tts_time = await self.tts_processor.synthesize_text(response_text)
                
                yield VoiceResponse(
                    type="tts_audio",
                    audio_data=audio_data,
                    metrics=VoiceMetrics(tts_latency=tts_time)
                )
                
            else:
                # 使用实际的检索服务
                from .models import QueryRequest, Message
                query_request = QueryRequest(
                    messages=[Message(role="user", content=query)],
                    top_k=5,
                    temperature=0.3
                )
                
                response_text = ""
                references = []
                
                async for response in self.retrieve_service.stream_query(query_request):
                    response_data = json.loads(response)
                    
                    if response_data["type"] == "text":
                        response_text += response_data.get("content", "")
                        
                        yield VoiceResponse(
                            type="llm_response",
                            text=response_text
                        )
                    
                    elif response_data["type"] == "refs":
                        references = response_data.get("refs", [])
                
                # TTS合成最终响应
                if response_text:
                    tts_start = time.time()
                    audio_data, tts_time = await self.tts_processor.synthesize_text(response_text)
                    
                    yield VoiceResponse(
                        type="tts_audio",
                        audio_data=audio_data,
                        references=references,
                        metrics=VoiceMetrics(tts_latency=tts_time)
                    )
                
        except Exception as e:
            yield VoiceResponse(
                type="error",
                error=f"LLM processing error: {str(e)}"
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # 最近100次
        
        avg_asr = np.mean([m.asr_latency for m in recent_metrics])
        avg_llm = np.mean([m.llm_latency for m in recent_metrics])
        avg_tts = np.mean([m.tts_latency for m in recent_metrics])
        avg_total = np.mean([m.total_latency for m in recent_metrics])
        avg_efficiency = np.mean([m.pipeline_efficiency for m in recent_metrics if m.pipeline_efficiency > 0])
        
        return {
            'mode': self.config.mode.value,
            'target_latency': self.config.target_latency,
            'average_latencies': {
                'asr': round(avg_asr, 2),
                'llm': round(avg_llm, 2),
                'tts': round(avg_tts, 2),
                'total': round(avg_total, 2)
            },
            'pipeline_efficiency': round(avg_efficiency, 3),
            'cache_stats': self.cache_manager.get_stats(),
            'active_sessions': len(self.active_sessions),
            'samples_count': len(recent_metrics)
        }


# 全局实例
_voice_service = None


def get_voice_service(retrieve_service=None, config: Optional[VoiceConfig] = None) -> UnifiedVoiceService:
    """获取语音服务单例"""
    global _voice_service
    if _voice_service is None:
        _voice_service = UnifiedVoiceService(retrieve_service, config)
    return _voice_service


# 便捷函数
async def process_voice_audio(
    conversation_id: str, 
    audio_chunk: bytes, 
    seq: int = 0,
    is_final: bool = False
) -> AsyncGenerator[VoiceResponse, None]:
    """处理语音音频"""
    voice_service = get_voice_service()
    
    request = VoiceRequest(
        conversation_id=conversation_id,
        audio_chunk=base64.b64encode(audio_chunk).decode(),
        seq=seq,
        is_final=is_final
    )
    
    async for response in voice_service.process_voice_request(request):
        yield response


async def synthesize_speech(text: str, emotion: Optional[str] = None) -> bytes:
    """合成语音"""
    voice_service = get_voice_service()
    audio_data, _ = await voice_service.tts_processor.synthesize_text(text, emotion)
    return audio_data
