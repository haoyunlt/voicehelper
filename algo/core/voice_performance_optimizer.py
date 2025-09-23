"""
语音处理性能优化模块
支持延迟优化、并发处理、音频质量提升
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import threading
from collections import deque

from loguru import logger


class AudioFormat(Enum):
    """音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    WEBM = "webm"


class ProcessingMode(Enum):
    """处理模式"""
    REALTIME = "realtime"      # 实时处理
    BATCH = "batch"            # 批处理
    STREAMING = "streaming"    # 流式处理
    OFFLINE = "offline"        # 离线处理


@dataclass
class AudioChunk:
    """音频块"""
    data: bytes
    sample_rate: int
    channels: int
    format: AudioFormat
    timestamp: float
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    result_data: Any
    processing_time: float
    quality_score: float
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    p95_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    concurrent_requests: int = 0
    queue_size: int = 0
    recent_processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: datetime = field(default_factory=datetime.now)


class VoicePerformanceOptimizer:
    """语音性能优化器"""
    
    def __init__(
        self,
        max_concurrent_requests: int = 10,
        max_queue_size: int = 100,
        enable_preprocessing: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 3600
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.enable_preprocessing = enable_preprocessing
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # 线程池和队列
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self.processing_queue = asyncio.Queue(maxsize=max_queue_size)
        self.priority_queue = asyncio.PriorityQueue()
        
        # 性能指标
        self.asr_metrics = PerformanceMetrics()
        self.tts_metrics = PerformanceMetrics()
        
        # 缓存
        self.asr_cache: Dict[str, Tuple[str, float]] = {}  # hash -> (result, timestamp)
        self.tts_cache: Dict[str, Tuple[bytes, float]] = {}
        
        # 并发控制
        self.current_asr_requests = 0
        self.current_tts_requests = 0
        self.asr_semaphore = asyncio.Semaphore(max_concurrent_requests // 2)
        self.tts_semaphore = asyncio.Semaphore(max_concurrent_requests // 2)
        
        # 音频预处理器
        self.audio_preprocessor = AudioPreprocessor()
        
        # 启动后台任务
        self._start_background_tasks()
    
    async def optimize_asr_request(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh-CN",
        priority: int = 0,
        enable_cache: bool = True
    ) -> ProcessingResult:
        """优化ASR请求"""
        start_time = time.time()
        
        try:
            # 生成缓存键
            cache_key = self._generate_audio_hash(audio_data) if enable_cache and self.enable_caching else None
            
            # 检查缓存
            if cache_key and cache_key in self.asr_cache:
                cached_result, cached_time = self.asr_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    processing_time = time.time() - start_time
                    self._update_asr_metrics(True, processing_time)
                    
                    logger.debug(f"ASR缓存命中: {cache_key[:8]}...")
                    return ProcessingResult(
                        success=True,
                        result_data=cached_result,
                        processing_time=processing_time,
                        quality_score=1.0,
                        metadata={"cached": True}
                    )
            
            # 并发控制
            async with self.asr_semaphore:
                self.current_asr_requests += 1
                
                try:
                    # 音频预处理
                    if self.enable_preprocessing:
                        audio_data = await self._preprocess_audio_for_asr(
                            audio_data, sample_rate
                        )
                    
                    # 执行ASR
                    result = await self._execute_asr(audio_data, sample_rate, language)
                    
                    # 缓存结果
                    if cache_key and result:
                        self.asr_cache[cache_key] = (result, time.time())
                    
                    processing_time = time.time() - start_time
                    self._update_asr_metrics(True, processing_time)
                    
                    return ProcessingResult(
                        success=True,
                        result_data=result,
                        processing_time=processing_time,
                        quality_score=self._calculate_asr_quality(result),
                        metadata={"cached": False}
                    )
                    
                finally:
                    self.current_asr_requests -= 1
                    
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_asr_metrics(False, processing_time)
            
            logger.error(f"ASR处理失败: {str(e)}")
            return ProcessingResult(
                success=False,
                result_data=None,
                processing_time=processing_time,
                quality_score=0.0,
                error_message=str(e)
            )
    
    async def optimize_tts_request(
        self,
        text: str,
        voice: str = "zh-CN-XiaoxiaoNeural",
        speed: float = 1.0,
        priority: int = 0,
        enable_cache: bool = True
    ) -> ProcessingResult:
        """优化TTS请求"""
        start_time = time.time()
        
        try:
            # 生成缓存键
            cache_key = self._generate_text_hash(text, voice, speed) if enable_cache and self.enable_caching else None
            
            # 检查缓存
            if cache_key and cache_key in self.tts_cache:
                cached_result, cached_time = self.tts_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    processing_time = time.time() - start_time
                    self._update_tts_metrics(True, processing_time)
                    
                    logger.debug(f"TTS缓存命中: {cache_key[:8]}...")
                    return ProcessingResult(
                        success=True,
                        result_data=cached_result,
                        processing_time=processing_time,
                        quality_score=1.0,
                        metadata={"cached": True}
                    )
            
            # 并发控制
            async with self.tts_semaphore:
                self.current_tts_requests += 1
                
                try:
                    # 文本预处理
                    if self.enable_preprocessing:
                        text = self._preprocess_text_for_tts(text)
                    
                    # 执行TTS
                    result = await self._execute_tts(text, voice, speed)
                    
                    # 缓存结果
                    if cache_key and result:
                        self.tts_cache[cache_key] = (result, time.time())
                    
                    processing_time = time.time() - start_time
                    self._update_tts_metrics(True, processing_time)
                    
                    return ProcessingResult(
                        success=True,
                        result_data=result,
                        processing_time=processing_time,
                        quality_score=self._calculate_tts_quality(result),
                        metadata={"cached": False}
                    )
                    
                finally:
                    self.current_tts_requests -= 1
                    
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_tts_metrics(False, processing_time)
            
            logger.error(f"TTS处理失败: {str(e)}")
            return ProcessingResult(
                success=False,
                result_data=None,
                processing_time=processing_time,
                quality_score=0.0,
                error_message=str(e)
            )
    
    async def batch_process_asr(
        self,
        audio_chunks: List[Tuple[bytes, int]],
        language: str = "zh-CN"
    ) -> List[ProcessingResult]:
        """批量处理ASR"""
        tasks = []
        for audio_data, sample_rate in audio_chunks:
            task = self.optimize_asr_request(audio_data, sample_rate, language)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    result_data=None,
                    processing_time=0.0,
                    quality_score=0.0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def stream_process_audio(
        self,
        audio_stream: asyncio.Queue,
        callback: callable,
        chunk_size: int = 1024
    ):
        """流式处理音频"""
        buffer = b""
        
        while True:
            try:
                # 获取音频块
                chunk = await asyncio.wait_for(audio_stream.get(), timeout=1.0)
                if chunk is None:  # 结束信号
                    break
                
                buffer += chunk
                
                # 当缓冲区足够大时处理
                if len(buffer) >= chunk_size:
                    # 异步处理
                    asyncio.create_task(self._process_audio_chunk(buffer, callback))
                    buffer = b""
                    
            except asyncio.TimeoutError:
                # 处理剩余缓冲区
                if buffer:
                    asyncio.create_task(self._process_audio_chunk(buffer, callback))
                    buffer = b""
    
    async def _process_audio_chunk(self, audio_data: bytes, callback: callable):
        """处理音频块"""
        result = await self.optimize_asr_request(audio_data)
        if result.success and result.result_data:
            await callback(result.result_data)
    
    async def _preprocess_audio_for_asr(self, audio_data: bytes, sample_rate: int) -> bytes:
        """ASR音频预处理"""
        # 在线程池中执行CPU密集型操作
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.audio_preprocessor.preprocess_for_asr,
            audio_data,
            sample_rate
        )
    
    def _preprocess_text_for_tts(self, text: str) -> str:
        """TTS文本预处理"""
        # 清理文本
        text = text.strip()
        
        # 处理特殊字符
        text = text.replace("&", "和")
        text = text.replace("<", "小于")
        text = text.replace(">", "大于")
        
        # 处理数字
        # 这里可以添加数字转中文的逻辑
        
        return text
    
    async def _execute_asr(self, audio_data: bytes, sample_rate: int, language: str) -> str:
        """执行ASR"""
        # 这里应该调用实际的ASR服务
        # 为了示例，返回模拟结果
        await asyncio.sleep(0.1)  # 模拟处理时间
        return "这是ASR识别结果"
    
    async def _execute_tts(self, text: str, voice: str, speed: float) -> bytes:
        """执行TTS"""
        # 这里应该调用实际的TTS服务
        # 为了示例，返回模拟结果
        await asyncio.sleep(0.2)  # 模拟处理时间
        return b"fake_audio_data"
    
    def _generate_audio_hash(self, audio_data: bytes) -> str:
        """生成音频哈希"""
        import hashlib
        return hashlib.md5(audio_data).hexdigest()
    
    def _generate_text_hash(self, text: str, voice: str, speed: float) -> str:
        """生成文本哈希"""
        import hashlib
        content = f"{text}:{voice}:{speed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_asr_quality(self, result: str) -> float:
        """计算ASR质量分数"""
        if not result:
            return 0.0
        
        # 简单的质量评估
        # 实际应用中可以使用更复杂的算法
        length_score = min(len(result) / 100, 1.0)  # 长度评分
        return length_score
    
    def _calculate_tts_quality(self, result: bytes) -> float:
        """计算TTS质量分数"""
        if not result:
            return 0.0
        
        # 简单的质量评估
        size_score = min(len(result) / 10000, 1.0)  # 大小评分
        return size_score
    
    def _update_asr_metrics(self, success: bool, processing_time: float):
        """更新ASR指标"""
        self.asr_metrics.total_requests += 1
        self.asr_metrics.recent_processing_times.append(processing_time)
        
        if success:
            self.asr_metrics.successful_requests += 1
        else:
            self.asr_metrics.failed_requests += 1
        
        # 更新处理时间统计
        self.asr_metrics.min_processing_time = min(self.asr_metrics.min_processing_time, processing_time)
        self.asr_metrics.max_processing_time = max(self.asr_metrics.max_processing_time, processing_time)
        
        # 计算平均处理时间
        if self.asr_metrics.recent_processing_times:
            self.asr_metrics.avg_processing_time = sum(self.asr_metrics.recent_processing_times) / len(self.asr_metrics.recent_processing_times)
            
            # 计算P95
            sorted_times = sorted(self.asr_metrics.recent_processing_times)
            p95_index = int(len(sorted_times) * 0.95)
            self.asr_metrics.p95_processing_time = sorted_times[p95_index] if sorted_times else 0.0
        
        self.asr_metrics.concurrent_requests = self.current_asr_requests
        self.asr_metrics.last_updated = datetime.now()
    
    def _update_tts_metrics(self, success: bool, processing_time: float):
        """更新TTS指标"""
        self.tts_metrics.total_requests += 1
        self.tts_metrics.recent_processing_times.append(processing_time)
        
        if success:
            self.tts_metrics.successful_requests += 1
        else:
            self.tts_metrics.failed_requests += 1
        
        # 更新处理时间统计
        self.tts_metrics.min_processing_time = min(self.tts_metrics.min_processing_time, processing_time)
        self.tts_metrics.max_processing_time = max(self.tts_metrics.max_processing_time, processing_time)
        
        # 计算平均处理时间
        if self.tts_metrics.recent_processing_times:
            self.tts_metrics.avg_processing_time = sum(self.tts_metrics.recent_processing_times) / len(self.tts_metrics.recent_processing_times)
            
            # 计算P95
            sorted_times = sorted(self.tts_metrics.recent_processing_times)
            p95_index = int(len(sorted_times) * 0.95)
            self.tts_metrics.p95_processing_time = sorted_times[p95_index] if sorted_times else 0.0
        
        self.tts_metrics.concurrent_requests = self.current_tts_requests
        self.tts_metrics.last_updated = datetime.now()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            "asr": {
                "total_requests": self.asr_metrics.total_requests,
                "success_rate": self.asr_metrics.successful_requests / max(self.asr_metrics.total_requests, 1),
                "avg_processing_time": self.asr_metrics.avg_processing_time,
                "p95_processing_time": self.asr_metrics.p95_processing_time,
                "concurrent_requests": self.asr_metrics.concurrent_requests,
                "cache_size": len(self.asr_cache),
            },
            "tts": {
                "total_requests": self.tts_metrics.total_requests,
                "success_rate": self.tts_metrics.successful_requests / max(self.tts_metrics.total_requests, 1),
                "avg_processing_time": self.tts_metrics.avg_processing_time,
                "p95_processing_time": self.tts_metrics.p95_processing_time,
                "concurrent_requests": self.tts_metrics.concurrent_requests,
                "cache_size": len(self.tts_cache),
            },
            "system": {
                "max_concurrent_requests": self.max_concurrent_requests,
                "thread_pool_size": self.thread_pool._max_workers,
                "preprocessing_enabled": self.enable_preprocessing,
                "caching_enabled": self.enable_caching,
            }
        }
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 定期清理缓存
        asyncio.create_task(self._cleanup_cache_periodically())
        
        # 定期计算吞吐量
        asyncio.create_task(self._calculate_throughput_periodically())
    
    async def _cleanup_cache_periodically(self):
        """定期清理缓存"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟清理一次
                
                current_time = time.time()
                
                # 清理ASR缓存
                expired_keys = []
                for key, (_, timestamp) in self.asr_cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.asr_cache[key]
                
                # 清理TTS缓存
                expired_keys = []
                for key, (_, timestamp) in self.tts_cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.tts_cache[key]
                
                if expired_keys:
                    logger.debug(f"清理过期缓存: ASR={len([k for k in expired_keys if k in self.asr_cache])}, TTS={len([k for k in expired_keys if k in self.tts_cache])}")
                    
            except Exception as e:
                logger.error(f"缓存清理失败: {e}")
    
    async def _calculate_throughput_periodically(self):
        """定期计算吞吐量"""
        last_asr_requests = 0
        last_tts_requests = 0
        
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟计算一次
                
                # 计算ASR吞吐量
                current_asr_requests = self.asr_metrics.total_requests
                asr_throughput = (current_asr_requests - last_asr_requests) / 60.0
                self.asr_metrics.throughput_per_second = asr_throughput
                last_asr_requests = current_asr_requests
                
                # 计算TTS吞吐量
                current_tts_requests = self.tts_metrics.total_requests
                tts_throughput = (current_tts_requests - last_tts_requests) / 60.0
                self.tts_metrics.throughput_per_second = tts_throughput
                last_tts_requests = current_tts_requests
                
            except Exception as e:
                logger.error(f"吞吐量计算失败: {e}")


class AudioPreprocessor:
    """音频预处理器"""
    
    def preprocess_for_asr(self, audio_data: bytes, sample_rate: int) -> bytes:
        """ASR音频预处理"""
        try:
            # 这里应该实现实际的音频预处理逻辑
            # 包括：降噪、音量归一化、格式转换等
            
            # 为了示例，直接返回原始数据
            return audio_data
            
        except Exception as e:
            logger.error(f"音频预处理失败: {e}")
            return audio_data
    
    def normalize_volume(self, audio_data: bytes) -> bytes:
        """音量归一化"""
        # 实现音量归一化逻辑
        return audio_data
    
    def reduce_noise(self, audio_data: bytes) -> bytes:
        """降噪处理"""
        # 实现降噪逻辑
        return audio_data
    
    def convert_format(self, audio_data: bytes, target_format: AudioFormat) -> bytes:
        """格式转换"""
        # 实现格式转换逻辑
        return audio_data


# 全局语音性能优化器实例
voice_optimizer = VoicePerformanceOptimizer()
