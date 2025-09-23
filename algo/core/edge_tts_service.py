import asyncio
import edge_tts
import io
import logging
import time
import hashlib
from typing import AsyncGenerator, Optional, Dict, Any, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TTSConfig:
    """TTS配置"""
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    output_format: str = "mp3"  # mp3, wav, opus
    cache_enabled: bool = True
    cache_dir: str = "data/tts_cache"
    max_cache_size_mb: int = 500

@dataclass
class TTSRequest:
    """TTS请求"""
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None
    volume: Optional[str] = None
    output_format: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high, 2=urgent

@dataclass
class TTSResponse:
    """TTS响应"""
    audio_data: bytes
    format: str
    duration_ms: int
    text_length: int
    processing_time_ms: float
    cached: bool
    voice_used: str

class EdgeTTSService:
    """基于Edge-TTS的高性能语音合成服务"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程池用于并行处理
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 缓存管理
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "size_mb": 0.0
        }
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency_ms": 0.0,
            "total_audio_duration_ms": 0,
            "cache_hit_rate": 0.0
        }
        
        # 可用语音列表缓存
        self._available_voices: Optional[List[Dict[str, Any]]] = None
        self._voices_cache_time: Optional[float] = None
    
    async def initialize(self):
        """初始化TTS服务"""
        try:
            # 预加载可用语音列表
            await self.get_available_voices()
            
            # 清理过期缓存
            await self._cleanup_cache()
            
            logger.info("Edge-TTS服务初始化完成")
            
        except Exception as e:
            logger.error(f"TTS服务初始化失败: {e}")
            raise
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """语音合成"""
        start_time = time.time()
        
        try:
            # 使用配置中的默认值填充请求
            voice = request.voice or self.config.voice
            rate = request.rate or self.config.rate
            pitch = request.pitch or self.config.pitch
            volume = request.volume or self.config.volume
            output_format = request.output_format or self.config.output_format
            
            # 检查缓存
            cache_key = self._generate_cache_key(request.text, voice, rate, pitch, volume)
            cached_audio = await self._get_from_cache(cache_key)
            
            if cached_audio:
                self.cache_stats["hits"] += 1
                processing_time = (time.time() - start_time) * 1000
                
                return TTSResponse(
                    audio_data=cached_audio,
                    format=output_format,
                    duration_ms=self._estimate_duration(request.text),
                    text_length=len(request.text),
                    processing_time_ms=processing_time,
                    cached=True,
                    voice_used=voice
                )
            
            self.cache_stats["misses"] += 1
            
            # 执行语音合成
            audio_data = await self._synthesize_audio(
                request.text, voice, rate, pitch, volume
            )
            
            # 保存到缓存
            if self.config.cache_enabled:
                await self._save_to_cache(cache_key, audio_data)
            
            processing_time = (time.time() - start_time) * 1000
            duration_ms = self._estimate_duration(request.text)
            
            # 更新统计
            self._update_stats(processing_time, duration_ms, True)
            
            return TTSResponse(
                audio_data=audio_data,
                format=output_format,
                duration_ms=duration_ms,
                text_length=len(request.text),
                processing_time_ms=processing_time,
                cached=False,
                voice_used=voice
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, 0, False)
            logger.error(f"TTS合成失败: {e}")
            raise
    
    async def synthesize_stream(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        """流式语音合成"""
        try:
            voice = request.voice or self.config.voice
            rate = request.rate or self.config.rate
            pitch = request.pitch or self.config.pitch
            volume = request.volume or self.config.volume
            
            communicate = edge_tts.Communicate(
                text=request.text,
                voice=voice,
                rate=rate,
                pitch=pitch,
                volume=volume
            )
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    # 可以用于实现字级别的同步
                    logger.debug(f"Word boundary: {chunk}")
                    
        except Exception as e:
            logger.error(f"流式TTS合成失败: {e}")
            raise
    
    async def _synthesize_audio(
        self, 
        text: str, 
        voice: str, 
        rate: str, 
        pitch: str, 
        volume: str
    ) -> bytes:
        """执行音频合成"""
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume
        )
        
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        return audio_data
    
    def _generate_cache_key(
        self, 
        text: str, 
        voice: str, 
        rate: str, 
        pitch: str, 
        volume: str
    ) -> str:
        """生成缓存键"""
        content = f"{text}|{voice}|{rate}|{pitch}|{volume}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[bytes]:
        """从缓存获取音频数据"""
        if not self.config.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        
        try:
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'rb') as f:
                    return await f.read()
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
        
        return None
    
    async def _save_to_cache(self, cache_key: str, audio_data: bytes):
        """保存音频数据到缓存"""
        if not self.config.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        
        try:
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(audio_data)
            
            # 更新缓存大小统计
            self.cache_stats["size_mb"] += len(audio_data) / (1024 * 1024)
            
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    async def _cleanup_cache(self):
        """清理过期缓存"""
        try:
            cache_files = list(self.cache_dir.glob("*.mp3"))
            total_size = sum(f.stat().st_size for f in cache_files)
            current_size_mb = total_size / (1024 * 1024)
            
            if current_size_mb > self.config.max_cache_size_mb:
                # 按修改时间排序，删除最旧的文件
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                
                while current_size_mb > self.config.max_cache_size_mb * 0.8:
                    if not cache_files:
                        break
                    
                    file_to_delete = cache_files.pop(0)
                    file_size = file_to_delete.stat().st_size
                    file_to_delete.unlink()
                    current_size_mb -= file_size / (1024 * 1024)
                    
                    logger.info(f"删除缓存文件: {file_to_delete.name}")
            
            self.cache_stats["size_mb"] = current_size_mb
            
        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")
    
    def _estimate_duration(self, text: str) -> int:
        """估算音频时长（毫秒）"""
        # 简单估算：中文约每分钟200字，英文约每分钟150词
        char_count = len(text)
        if any('\u4e00' <= char <= '\u9fff' for char in text):  # 包含中文
            words_per_minute = 200
        else:  # 英文或其他
            words_per_minute = 150
        
        duration_minutes = char_count / words_per_minute
        return int(duration_minutes * 60 * 1000)  # 转换为毫秒
    
    def _update_stats(self, processing_time: float, duration_ms: int, success: bool):
        """更新性能统计"""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
            self.stats["total_audio_duration_ms"] += duration_ms
            
            # 计算移动平均延迟
            alpha = 0.1
            self.stats["average_latency_ms"] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats["average_latency_ms"]
            )
        else:
            self.stats["failed_requests"] += 1
        
        # 更新缓存命中率
        total_cache_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_cache_requests > 0:
            self.stats["cache_hit_rate"] = self.cache_stats["hits"] / total_cache_requests
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """获取可用语音列表"""
        # 缓存1小时
        if (self._available_voices and self._voices_cache_time and 
            time.time() - self._voices_cache_time < 3600):
            return self._available_voices
        
        try:
            voices = await edge_tts.list_voices()
            self._available_voices = [
                {
                    "name": voice["Name"],
                    "display_name": voice["DisplayName"],
                    "locale": voice["Locale"],
                    "gender": voice["Gender"],
                    "suggested_codec": voice.get("SuggestedCodec", "audio-24khz-48kbitrate-mono-mp3"),
                    "friendly_name": voice.get("FriendlyName", ""),
                    "status": voice.get("Status", "GA")
                }
                for voice in voices
            ]
            self._voices_cache_time = time.time()
            
            logger.info(f"加载了 {len(self._available_voices)} 个可用语音")
            
        except Exception as e:
            logger.error(f"获取语音列表失败: {e}")
            if not self._available_voices:
                self._available_voices = []
        
        return self._available_voices
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = self.stats["successful_requests"] / self.stats["total_requests"]
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_stats": self.cache_stats,
            "config": {
                "voice": self.config.voice,
                "cache_enabled": self.config.cache_enabled,
                "cache_size_mb": self.cache_stats["size_mb"],
                "max_cache_size_mb": self.config.max_cache_size_mb
            }
        }
    
    async def clear_cache(self):
        """清空缓存"""
        try:
            cache_files = list(self.cache_dir.glob("*.mp3"))
            for cache_file in cache_files:
                cache_file.unlink()
            
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "size_mb": 0.0
            }
            
            logger.info(f"清空了 {len(cache_files)} 个缓存文件")
            
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

# 使用示例和测试
async def test_edge_tts():
    """测试Edge-TTS服务"""
    config = TTSConfig(
        voice="zh-CN-XiaoxiaoNeural",
        cache_enabled=True,
        max_cache_size_mb=100
    )
    
    tts = EdgeTTSService(config)
    await tts.initialize()
    
    # 测试语音合成
    request = TTSRequest(
        text="你好，我是VoiceHelper语音助手，很高兴为您服务！",
        voice="zh-CN-XiaoxiaoNeural"
    )
    
    print("开始语音合成测试...")
    
    # 第一次合成（缓存未命中）
    response1 = await tts.synthesize(request)
    print(f"第一次合成:")
    print(f"  音频大小: {len(response1.audio_data)} bytes")
    print(f"  处理时间: {response1.processing_time_ms:.2f}ms")
    print(f"  是否缓存: {response1.cached}")
    print(f"  预估时长: {response1.duration_ms}ms")
    
    # 第二次合成（缓存命中）
    response2 = await tts.synthesize(request)
    print(f"第二次合成:")
    print(f"  音频大小: {len(response2.audio_data)} bytes")
    print(f"  处理时间: {response2.processing_time_ms:.2f}ms")
    print(f"  是否缓存: {response2.cached}")
    
    # 测试流式合成
    print("开始流式合成测试...")
    chunks = []
    async for chunk in tts.synthesize_stream(request):
        chunks.append(chunk)
    
    print(f"流式合成完成，共 {len(chunks)} 个音频块")
    
    # 获取可用语音
    voices = await tts.get_available_voices()
    print(f"可用语音数量: {len(voices)}")
    
    # 打印统计信息
    stats = tts.get_stats()
    print(f"性能统计: {stats}")

if __name__ == "__main__":
    asyncio.run(test_edge_tts())
