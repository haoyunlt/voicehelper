import asyncio
import torch
import whisper
import numpy as np
import webrtcvad
import collections
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class ASRConfig:
    """ASR配置"""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "zh"
    device: str = "auto"  # auto, cpu, cuda
    vad_aggressiveness: int = 2  # 0-3, 越高越激进
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    silence_timeout_ms: int = 1000

@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    confidence: float
    is_final: bool
    language: str
    processing_time_ms: float
    timestamp: float

class WhisperRealtimeASR:
    """基于OpenAI Whisper的实时ASR服务"""
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.model = None
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 音频缓冲区管理
        self.frame_size = int(config.sample_rate * config.frame_duration_ms / 1000)
        self.audio_buffer = collections.deque(maxlen=100)  # 3秒缓冲
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speaking = False
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_latency": 0.0,
            "error_count": 0
        }
    
    async def initialize(self):
        """初始化Whisper模型"""
        try:
            logger.info(f"加载Whisper模型: {self.config.model_size}")
            
            # 确定设备
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            # 在线程池中加载模型，避免阻塞
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                whisper.load_model,
                self.config.model_size,
                device
            )
            
            logger.info(f"Whisper模型加载完成，设备: {device}")
            
        except Exception as e:
            logger.error(f"Whisper模型加载失败: {e}")
            raise
    
    async def process_audio_stream(
        self, 
        audio_chunk: bytes, 
        session_id: str = "default"
    ) -> AsyncGenerator[ASRResult, None]:
        """处理实时音频流"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # VAD检测
            is_speech = self._detect_speech(audio_chunk)
            
            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                
                # 添加到音频缓冲区
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                self.audio_buffer.extend(audio_data)
                
                # 检测说话开始
                if not self.is_speaking and self.speech_frames > 3:  # 90ms连续语音
                    self.is_speaking = True
                    yield ASRResult(
                        text="",
                        confidence=0.0,
                        is_final=False,
                        language=self.config.language,
                        processing_time_ms=0.0,
                        timestamp=start_time
                    )
                
                # 实时识别（部分结果）
                if len(self.audio_buffer) > self.frame_size * 10:  # 300ms音频
                    partial_result = await self._transcribe_partial()
                    if partial_result:
                        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
                        yield ASRResult(
                            text=partial_result,
                            confidence=0.8,  # 部分结果置信度较低
                            is_final=False,
                            language=self.config.language,
                            processing_time_ms=processing_time,
                            timestamp=start_time
                        )
            
            else:  # 静音
                self.silence_frames += 1
                
                # 检测说话结束
                if (self.is_speaking and 
                    self.silence_frames > self.config.silence_timeout_ms // self.config.frame_duration_ms):
                    
                    self.is_speaking = False
                    
                    # 执行最终识别
                    if len(self.audio_buffer) > 0:
                        final_result = await self._transcribe_final()
                        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
                        
                        yield ASRResult(
                            text=final_result,
                            confidence=0.95,  # 最终结果置信度较高
                            is_final=True,
                            language=self.config.language,
                            processing_time_ms=processing_time,
                            timestamp=start_time
                        )
                        
                        # 清空缓冲区
                        self.audio_buffer.clear()
                        self.speech_frames = 0
                        
                        # 更新统计
                        self._update_stats(processing_time, True)
            
        except Exception as e:
            logger.error(f"ASR处理错误: {e}")
            self.stats["error_count"] += 1
            yield ASRResult(
                text="",
                confidence=0.0,
                is_final=True,
                language=self.config.language,
                processing_time_ms=0.0,
                timestamp=asyncio.get_event_loop().time()
            )
    
    def _detect_speech(self, audio_chunk: bytes) -> bool:
        """语音活动检测"""
        try:
            return self.vad.is_speech(audio_chunk, self.config.sample_rate)
        except Exception as e:
            logger.warning(f"VAD检测失败: {e}")
            return False
    
    async def _transcribe_partial(self) -> str:
        """部分转录（快速，低精度）"""
        try:
            # 使用最近的音频片段
            recent_audio = list(self.audio_buffer)[-self.frame_size * 20:]  # 最近600ms
            audio_array = np.array(recent_audio, dtype=np.float32) / 32768.0
            
            # 在线程池中执行转录
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._whisper_transcribe,
                audio_array,
                {"task": "transcribe", "language": self.config.language, "fp16": False}
            )
            
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.warning(f"部分转录失败: {e}")
            return ""
    
    async def _transcribe_final(self) -> str:
        """最终转录（高精度）"""
        try:
            # 使用完整音频缓冲区
            audio_array = np.array(list(self.audio_buffer), dtype=np.float32) / 32768.0
            
            # 在线程池中执行转录
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._whisper_transcribe,
                audio_array,
                {
                    "task": "transcribe", 
                    "language": self.config.language, 
                    "fp16": torch.cuda.is_available(),
                    "temperature": 0.0  # 确定性输出
                }
            )
            
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"最终转录失败: {e}")
            return ""
    
    def _whisper_transcribe(self, audio: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
        """Whisper转录（同步方法，在线程池中执行）"""
        return self.model.transcribe(audio, **options)
    
    def _update_stats(self, processing_time: float, success: bool):
        """更新性能统计"""
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
            # 计算移动平均延迟
            alpha = 0.1  # 平滑因子
            self.stats["average_latency"] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats["average_latency"]
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        success_rate = (
            self.stats["successful_requests"] / max(1, self.stats["total_requests"])
        )
        return {
            **self.stats,
            "success_rate": success_rate,
            "model_info": {
                "model_size": self.config.model_size,
                "language": self.config.language,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }

# 使用示例和测试
async def test_whisper_asr():
    """测试Whisper ASR服务"""
    config = ASRConfig(
        model_size="base",
        language="zh",
        vad_aggressiveness=2
    )
    
    asr = WhisperRealtimeASR(config)
    await asr.initialize()
    
    # 模拟音频流处理
    print("开始音频流处理测试...")
    
    # 这里应该是真实的音频数据
    # 为了测试，我们使用模拟数据
    sample_audio = np.random.randint(-1000, 1000, 480, dtype=np.int16).tobytes()
    
    async for result in asr.process_audio_stream(sample_audio):
        print(f"ASR结果: {result.text}")
        print(f"置信度: {result.confidence}")
        print(f"是否最终: {result.is_final}")
        print(f"处理时间: {result.processing_time_ms:.2f}ms")
        print("---")
        
        if result.is_final:
            break
    
    # 打印统计信息
    stats = asr.get_stats()
    print(f"性能统计: {stats}")

if __name__ == "__main__":
    asyncio.run(test_whisper_asr())
