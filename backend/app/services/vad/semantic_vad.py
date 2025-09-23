"""
语义VAD - 语音活动检测与端点检测
"""
import numpy as np
from typing import List, Tuple, Optional
import asyncio
import structlog
from collections import deque

from app.config import settings
from app.models.schemas import EventType

logger = structlog.get_logger()


class SemanticVAD:
    """语义语音活动检测器"""
    
    def __init__(
        self,
        min_speech_ms: int = 120,
        min_silence_ms: int = 200,
        energy_thresh: float = -45.0,
        sample_rate: int = 16000
    ):
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.energy_thresh = energy_thresh
        self.sample_rate = sample_rate
        
        # 状态跟踪
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        
        # 能量历史缓冲区
        self.energy_history = deque(maxlen=50)  # 保留最近50个能量值
        self.adaptive_threshold = energy_thresh
        
        # 事件回调
        self.event_callbacks = {}
    
    def register_callback(self, session_id: str, callback):
        """注册事件回调函数"""
        self.event_callbacks[session_id] = callback
    
    def unregister_callback(self, session_id: str):
        """注销事件回调"""
        if session_id in self.event_callbacks:
            del self.event_callbacks[session_id]
    
    async def feed(
        self, 
        session_id: str,
        pcm_chunk: bytes, 
        timestamp_ms: int
    ) -> List[Tuple[str, int]]:
        """
        输入音频数据，返回检测到的事件
        
        Returns:
            List of (event_type, timestamp_ms) tuples
        """
        events = []
        
        try:
            # 计算音频能量
            energy_db = self._calculate_energy(pcm_chunk)
            
            # 更新自适应阈值
            self._update_adaptive_threshold(energy_db)
            
            # 检测语音活动
            is_voice_active = energy_db > self.adaptive_threshold
            
            # 状态机处理
            if not self.is_speaking and is_voice_active:
                # 开始说话
                if self.speech_start_time is None:
                    self.speech_start_time = timestamp_ms
                elif timestamp_ms - self.speech_start_time >= self.min_speech_ms:
                    # 确认开始说话
                    self.is_speaking = True
                    self.silence_start_time = None
                    events.append((EventType.SPEECH_START, timestamp_ms))
                    
                    logger.debug(
                        "Speech started",
                        session_id=session_id,
                        timestamp_ms=timestamp_ms,
                        energy_db=energy_db
                    )
                    
                    # 触发TTS取消
                    await self._trigger_tts_cancel(session_id, timestamp_ms)
            
            elif self.is_speaking and not is_voice_active:
                # 可能结束说话
                if self.silence_start_time is None:
                    self.silence_start_time = timestamp_ms
                elif timestamp_ms - self.silence_start_time >= self.min_silence_ms:
                    # 确认结束说话
                    self.is_speaking = False
                    self.speech_start_time = None
                    events.append((EventType.SPEECH_END, timestamp_ms))
                    
                    logger.debug(
                        "Speech ended",
                        session_id=session_id,
                        timestamp_ms=timestamp_ms,
                        energy_db=energy_db
                    )
            
            elif self.is_speaking and is_voice_active:
                # 继续说话，重置静音计时
                self.silence_start_time = None
            
            elif not self.is_speaking and not is_voice_active:
                # 继续静音，重置语音计时
                self.speech_start_time = None
        
        except Exception as e:
            logger.error(
                "Error in VAD processing",
                session_id=session_id,
                error=str(e)
            )
        
        return events
    
    def _calculate_energy(self, pcm_chunk: bytes) -> float:
        """计算音频能量（dB）"""
        try:
            # 转换为numpy数组
            audio_data = np.frombuffer(pcm_chunk, dtype=np.int16)
            
            if len(audio_data) == 0:
                return -60.0
            
            # 计算RMS能量
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            
            # 转换为dB
            if rms > 0:
                energy_db = 20 * np.log10(rms / 32768.0)  # 16-bit PCM参考
            else:
                energy_db = -60.0
            
            return energy_db
        
        except Exception as e:
            logger.error("Error calculating audio energy", error=str(e))
            return -60.0
    
    def _update_adaptive_threshold(self, current_energy: float):
        """更新自适应阈值"""
        self.energy_history.append(current_energy)
        
        if len(self.energy_history) >= 10:
            # 计算背景噪声水平
            sorted_energies = sorted(self.energy_history)
            noise_floor = np.percentile(sorted_energies, 25)  # 25th percentile
            
            # 自适应阈值 = 噪声底线 + 固定偏移
            self.adaptive_threshold = max(
                noise_floor + 10.0,  # 至少比噪声高10dB
                self.energy_thresh    # 不低于配置的最小阈值
            )
    
    async def _trigger_tts_cancel(self, session_id: str, timestamp_ms: int):
        """触发TTS取消事件"""
        if session_id in self.event_callbacks:
            try:
                callback = self.event_callbacks[session_id]
                await callback("tts_cancel", {
                    "session_id": session_id,
                    "timestamp_ms": timestamp_ms,
                    "reason": "speech_detected"
                })
            except Exception as e:
                logger.error(
                    "Error triggering TTS cancel",
                    session_id=session_id,
                    error=str(e)
                )
    
    def reset(self):
        """重置VAD状态"""
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.energy_history.clear()
        self.adaptive_threshold = self.energy_thresh
        
        logger.debug("VAD state reset")
    
    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "is_speaking": self.is_speaking,
            "adaptive_threshold": self.adaptive_threshold,
            "energy_history_size": len(self.energy_history),
            "speech_start_time": self.speech_start_time,
            "silence_start_time": self.silence_start_time
        }
