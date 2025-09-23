"""
VAD (Voice Activity Detection) 端点检测服务
功能: 语音活动检测 + 端点检测 + 参数可配 + 多算法支持
"""

import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

from .events import EventStream, EventType

logger = logging.getLogger(__name__)

class VADAlgorithm(Enum):
    """VAD算法类型"""
    ENERGY_BASED = "energy_based"
    SPECTRAL_CENTROID = "spectral_centroid"
    ZERO_CROSSING_RATE = "zero_crossing_rate"
    HYBRID = "hybrid"

class VADState(Enum):
    """VAD状态"""
    SILENCE = "silence"
    SPEECH = "speech"
    TRANSITION = "transition"

@dataclass
class VADConfig:
    """VAD配置参数"""
    # 基础参数
    algorithm: VADAlgorithm = VADAlgorithm.HYBRID
    sample_rate: int = 16000
    frame_size: int = 320  # 20ms @ 16kHz
    
    # 阈值参数
    silence_threshold: float = 0.01
    speech_threshold: float = 0.02
    energy_threshold: float = 0.001
    
    # 时间参数 (毫秒)
    min_speech_duration: int = 300    # 最短语音持续时间
    max_silence_duration: int = 800   # 最长静音持续时间
    backfill_duration: int = 200      # 回填时长
    hangover_duration: int = 100      # 挂起时长
    
    # 激进模式 (降低延迟)
    aggressive_mode: bool = True
    aggressive_speech_threshold: float = 0.005
    aggressive_min_speech: int = 150
    aggressive_max_silence: int = 400
    
    # 频谱参数
    spectral_centroid_threshold: float = 1000.0
    zero_crossing_threshold: float = 0.1
    
    # 平滑参数
    smoothing_factor: float = 0.1
    history_length: int = 10

@dataclass
class VADResult:
    """VAD检测结果"""
    is_speech: bool
    confidence: float
    energy: float
    state: VADState
    timestamp: int
    frame_index: int
    
    # 端点检测结果
    speech_start: Optional[int] = None
    speech_end: Optional[int] = None
    endpoint_detected: bool = False

class EndpointDetector:
    """端点检测器"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        """重置检测器状态"""
        self.state = VADState.SILENCE
        self.frame_count = 0
        self.speech_frames = 0
        self.silence_frames = 0
        self.speech_start_frame = None
        self.last_speech_frame = None
        
        # 历史记录
        self.energy_history: List[float] = []
        self.vad_history: List[bool] = []
        
        # 平滑处理
        self.smoothed_energy = 0.0
        
        # 统计信息
        self.total_speech_time = 0
        self.total_silence_time = 0
        self.speech_segments = 0
        
    def process_frame(self, audio_frame: np.ndarray, timestamp: int) -> VADResult:
        """处理单个音频帧"""
        self.frame_count += 1
        
        # 计算音频特征
        features = self._extract_features(audio_frame)
        
        # VAD决策
        is_speech, confidence = self._make_vad_decision(features)
        
        # 更新历史记录
        self._update_history(features['energy'], is_speech)
        
        # 状态机处理
        previous_state = self.state
        endpoint_result = self._update_state_machine(is_speech, timestamp)
        
        # 创建结果
        result = VADResult(
            is_speech=is_speech,
            confidence=confidence,
            energy=features['energy'],
            state=self.state,
            timestamp=timestamp,
            frame_index=self.frame_count,
            speech_start=endpoint_result.get('speech_start'),
            speech_end=endpoint_result.get('speech_end'),
            endpoint_detected=endpoint_result.get('endpoint_detected', False)
        )
        
        # 记录状态变化
        if previous_state != self.state:
            logger.debug(f"VAD state change: {previous_state.value} -> {self.state.value} at frame {self.frame_count}")
        
        return result
    
    def _extract_features(self, audio_frame: np.ndarray) -> Dict[str, float]:
        """提取音频特征"""
        features = {}
        
        # 能量特征
        energy = np.sum(audio_frame ** 2) / len(audio_frame)
        features['energy'] = float(energy)
        
        # 平滑能量
        self.smoothed_energy = (1 - self.config.smoothing_factor) * self.smoothed_energy + \
                              self.config.smoothing_factor * energy
        features['smoothed_energy'] = self.smoothed_energy
        
        # 过零率
        zero_crossings = np.sum(np.diff(np.sign(audio_frame)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(audio_frame)
        
        # 频谱质心 (简化版本)
        if len(audio_frame) > 1:
            fft = np.abs(np.fft.fft(audio_frame))
            freqs = np.fft.fftfreq(len(audio_frame), 1.0 / self.config.sample_rate)
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * fft[:len(fft)//2]) / np.sum(fft[:len(fft)//2])
            features['spectral_centroid'] = abs(spectral_centroid)
        else:
            features['spectral_centroid'] = 0.0
        
        return features
    
    def _make_vad_decision(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """VAD决策"""
        if self.config.algorithm == VADAlgorithm.ENERGY_BASED:
            return self._energy_based_vad(features)
        elif self.config.algorithm == VADAlgorithm.SPECTRAL_CENTROID:
            return self._spectral_centroid_vad(features)
        elif self.config.algorithm == VADAlgorithm.ZERO_CROSSING_RATE:
            return self._zero_crossing_vad(features)
        elif self.config.algorithm == VADAlgorithm.HYBRID:
            return self._hybrid_vad(features)
        else:
            return self._energy_based_vad(features)
    
    def _energy_based_vad(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """基于能量的VAD"""
        energy = features['smoothed_energy']
        
        if self.config.aggressive_mode:
            threshold = self.config.aggressive_speech_threshold
        else:
            threshold = self.config.speech_threshold
        
        is_speech = energy > threshold
        confidence = min(1.0, energy / threshold) if threshold > 0 else 0.0
        
        return is_speech, confidence
    
    def _spectral_centroid_vad(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """基于频谱质心的VAD"""
        centroid = features['spectral_centroid']
        energy = features['energy']
        
        # 结合能量和频谱质心
        energy_condition = energy > self.config.energy_threshold
        spectral_condition = centroid > self.config.spectral_centroid_threshold
        
        is_speech = energy_condition and spectral_condition
        confidence = min(1.0, (energy / self.config.energy_threshold) * 
                        (centroid / self.config.spectral_centroid_threshold))
        
        return is_speech, confidence
    
    def _zero_crossing_vad(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """基于过零率的VAD"""
        zcr = features['zero_crossing_rate']
        energy = features['energy']
        
        # 语音通常有适中的过零率
        energy_condition = energy > self.config.energy_threshold
        zcr_condition = zcr < self.config.zero_crossing_threshold
        
        is_speech = energy_condition and zcr_condition
        confidence = min(1.0, energy / self.config.energy_threshold) if energy_condition else 0.0
        
        return is_speech, confidence
    
    def _hybrid_vad(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """混合VAD算法"""
        # 获取各种特征的决策
        energy_decision, energy_conf = self._energy_based_vad(features)
        spectral_decision, spectral_conf = self._spectral_centroid_vad(features)
        zcr_decision, zcr_conf = self._zero_crossing_vad(features)
        
        # 加权投票
        decisions = [energy_decision, spectral_decision, zcr_decision]
        confidences = [energy_conf, spectral_conf, zcr_conf]
        weights = [0.6, 0.3, 0.1]  # 能量权重最高
        
        # 计算加权置信度
        weighted_confidence = sum(w * c for w, c in zip(weights, confidences))
        
        # 至少两个算法同意才认为是语音
        speech_votes = sum(decisions)
        is_speech = speech_votes >= 2 or (speech_votes >= 1 and weighted_confidence > 0.7)
        
        return is_speech, weighted_confidence
    
    def _update_history(self, energy: float, is_speech: bool):
        """更新历史记录"""
        self.energy_history.append(energy)
        self.vad_history.append(is_speech)
        
        # 限制历史长度
        if len(self.energy_history) > self.config.history_length:
            self.energy_history.pop(0)
            self.vad_history.pop(0)
    
    def _update_state_machine(self, is_speech: bool, timestamp: int) -> Dict[str, Any]:
        """更新状态机"""
        result = {}
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.last_speech_frame = self.frame_count
        else:
            self.silence_frames += 1
            
        # 获取时间参数
        if self.config.aggressive_mode:
            min_speech_frames = self.config.aggressive_min_speech * self.config.sample_rate // 1000 // self.config.frame_size
            max_silence_frames = self.config.aggressive_max_silence * self.config.sample_rate // 1000 // self.config.frame_size
        else:
            min_speech_frames = self.config.min_speech_duration * self.config.sample_rate // 1000 // self.config.frame_size
            max_silence_frames = self.config.max_silence_duration * self.config.sample_rate // 1000 // self.config.frame_size
        
        # 状态转换逻辑
        if self.state == VADState.SILENCE:
            if is_speech:
                if self.speech_frames >= min_speech_frames:
                    # 检测到语音开始
                    self.state = VADState.SPEECH
                    self.speech_start_frame = self.frame_count - self.speech_frames + 1
                    result['speech_start'] = timestamp - (self.speech_frames - 1) * self.config.frame_size * 1000 // self.config.sample_rate
                    result['endpoint_detected'] = True
                    self.speech_segments += 1
                    logger.info(f"Speech start detected at frame {self.speech_start_frame}")
        
        elif self.state == VADState.SPEECH:
            if not is_speech:
                if self.silence_frames >= max_silence_frames:
                    # 检测到语音结束
                    self.state = VADState.SILENCE
                    speech_end_frame = self.last_speech_frame
                    result['speech_end'] = timestamp - self.silence_frames * self.config.frame_size * 1000 // self.config.sample_rate
                    result['endpoint_detected'] = True
                    
                    # 计算语音段时长
                    if self.speech_start_frame:
                        speech_duration = (speech_end_frame - self.speech_start_frame + 1) * self.config.frame_size * 1000 // self.config.sample_rate
                        self.total_speech_time += speech_duration
                        logger.info(f"Speech end detected at frame {speech_end_frame}, duration: {speech_duration}ms")
                    
                    # 重置计数器
                    self.speech_frames = 0
            else:
                # 继续语音状态
                self.silence_frames = 0
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "state": self.state.value,
            "frame_count": self.frame_count,
            "speech_frames": self.speech_frames,
            "silence_frames": self.silence_frames,
            "total_speech_time_ms": self.total_speech_time,
            "speech_segments": self.speech_segments,
            "current_energy": self.smoothed_energy,
            "config": {
                "algorithm": self.config.algorithm.value,
                "aggressive_mode": self.config.aggressive_mode,
                "speech_threshold": self.config.speech_threshold,
                "min_speech_duration": self.config.min_speech_duration,
                "max_silence_duration": self.config.max_silence_duration
            }
        }

class VADService:
    """VAD服务封装"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.detectors: Dict[str, EndpointDetector] = {}
        self._lock = asyncio.Lock()
        
    async def create_session(self, session_id: str) -> EndpointDetector:
        """创建VAD会话"""
        async with self._lock:
            detector = EndpointDetector(self.config)
            self.detectors[session_id] = detector
            logger.info(f"Created VAD session: {session_id}")
            return detector
    
    async def process_audio(self, session_id: str, audio_data: np.ndarray, 
                           timestamp: int, event_stream: Optional[EventStream] = None) -> VADResult:
        """处理音频数据"""
        if session_id not in self.detectors:
            await self.create_session(session_id)
        
        detector = self.detectors[session_id]
        result = detector.process_frame(audio_data, timestamp)
        
        # 发射VAD事件
        if event_stream:
            if result.endpoint_detected:
                if result.speech_start is not None:
                    await event_stream.emit(EventType.VAD_SPEECH_START, {
                        "timestamp": result.speech_start,
                        "confidence": result.confidence,
                        "energy": result.energy
                    })
                elif result.speech_end is not None:
                    await event_stream.emit(EventType.VAD_SPEECH_END, {
                        "timestamp": result.speech_end,
                        "confidence": result.confidence,
                        "energy": result.energy
                    })
            
            # 定期发送VAD状态
            if result.frame_index % 50 == 0:  # 每50帧发送一次
                await event_stream.emit(EventType.VAD_SILENCE if result.state == VADState.SILENCE else EventType.VAD_SPEECH_START, {
                    "state": result.state.value,
                    "confidence": result.confidence,
                    "energy": result.energy,
                    "frame_index": result.frame_index
                })
        
        return result
    
    async def remove_session(self, session_id: str):
        """移除VAD会话"""
        async with self._lock:
            if session_id in self.detectors:
                del self.detectors[session_id]
                logger.info(f"Removed VAD session: {session_id}")
    
    async def update_config(self, session_id: str, new_config: Dict[str, Any]):
        """更新会话配置"""
        if session_id in self.detectors:
            detector = self.detectors[session_id]
            
            # 更新配置参数
            for key, value in new_config.items():
                if hasattr(detector.config, key):
                    setattr(detector.config, key, value)
            
            logger.info(f"Updated VAD config for session {session_id}: {new_config}")
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话统计信息"""
        if session_id in self.detectors:
            return self.detectors[session_id].get_stats()
        return None
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有会话统计信息"""
        return {
            "total_sessions": len(self.detectors),
            "active_sessions": list(self.detectors.keys()),
            "sessions": {
                session_id: detector.get_stats()
                for session_id, detector in self.detectors.items()
            }
        }

# 工厂函数
def create_vad_service(algorithm: str = "hybrid", aggressive_mode: bool = True, **kwargs) -> VADService:
    """创建VAD服务"""
    config = VADConfig(
        algorithm=VADAlgorithm(algorithm),
        aggressive_mode=aggressive_mode,
        **kwargs
    )
    return VADService(config)

# 使用示例
async def example_usage():
    """使用示例"""
    from ..core.events import create_event_stream
    
    # 创建VAD服务
    vad_service = create_vad_service(
        algorithm="hybrid",
        aggressive_mode=True,
        speech_threshold=0.01,
        min_speech_duration=200,
        max_silence_duration=500
    )
    
    # 创建会话
    session_id = "example_session"
    event_stream = create_event_stream(session_id)
    
    # 模拟音频处理
    sample_rate = 16000
    frame_size = 320
    
    for i in range(100):
        # 生成模拟音频帧
        if 20 <= i <= 60:  # 模拟语音段
            audio_frame = np.random.normal(0, 0.1, frame_size)
        else:  # 模拟静音段
            audio_frame = np.random.normal(0, 0.01, frame_size)
        
        timestamp = int(time.time() * 1000)
        
        # 处理音频
        result = await vad_service.process_audio(session_id, audio_frame, timestamp, event_stream)
        
        print(f"Frame {i}: State={result.state.value}, Speech={result.is_speech}, "
              f"Confidence={result.confidence:.3f}, Energy={result.energy:.6f}")
        
        if result.endpoint_detected:
            if result.speech_start:
                print(f"  -> Speech START detected at {result.speech_start}")
            if result.speech_end:
                print(f"  -> Speech END detected at {result.speech_end}")
        
        await asyncio.sleep(0.02)  # 20ms间隔
    
    # 获取统计信息
    stats = vad_service.get_session_stats(session_id)
    print(f"\nFinal stats: {stats}")

if __name__ == "__main__":
    asyncio.run(example_usage())
