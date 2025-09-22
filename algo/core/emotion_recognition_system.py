"""
实时语音情感识别系统
支持7种基础情感：neutral, happy, sad, angry, fear, surprise, disgust
基于深度学习和音频特征分析
"""

import asyncio
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import time
import json

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"

@dataclass
class EmotionResult:
    dominant_emotion: EmotionType
    confidence: float
    emotion_scores: Dict[str, float]
    audio_features: Dict[str, float]
    timestamp: float
    processing_time: float

class EmotionFeatureExtractor:
    """音频特征提取器"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        """提取音频特征"""
        features = {}
        
        try:
            # 1. 基础统计特征
            features.update(self._extract_basic_features(audio))
            
            # 2. 频谱特征
            features.update(self._extract_spectral_features(audio))
            
            # 3. 韵律特征
            features.update(self._extract_prosodic_features(audio))
            
            # 4. MFCC特征
            features.update(self._extract_mfcc_features(audio))
            
            # 5. 语音质量特征
            features.update(self._extract_voice_quality_features(audio))
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # 返回默认特征
            features = {f"feature_{i}": 0.0 for i in range(50)}
            
        return features
    
    def _extract_basic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """基础统计特征"""
        return {
            'energy': float(np.sum(audio ** 2)),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio)[0])),
            'amplitude_mean': float(np.mean(np.abs(audio))),
            'amplitude_std': float(np.std(np.abs(audio))),
            'amplitude_max': float(np.max(np.abs(audio))),
            'amplitude_min': float(np.min(np.abs(audio)))
        }
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """频谱特征"""
        # 计算频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        # 计算频谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        
        # 计算频谱对比度
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        
        # 计算频谱滚降
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
            'spectral_contrast_mean': float(np.mean(spectral_contrast)),
            'spectral_contrast_std': float(np.std(spectral_contrast)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff))
        }
    
    def _extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """韵律特征（音调、节奏等）"""
        # 基频提取
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # 过滤有效的基频值
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 0:
            f0_mean = float(np.mean(f0_valid))
            f0_std = float(np.std(f0_valid))
            f0_range = float(np.max(f0_valid) - np.min(f0_valid))
        else:
            f0_mean = f0_std = f0_range = 0.0
        
        return {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_range': f0_range,
            'voiced_ratio': float(np.mean(voiced_flag)),
            'voicing_confidence': float(np.mean(voiced_probs))
        }
    
    def _extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, float]:
        """MFCC特征"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        features = {}
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        return features
    
    def _extract_voice_quality_features(self, audio: np.ndarray) -> Dict[str, float]:
        """语音质量特征"""
        # 计算谐波噪声比
        harmonic, percussive = librosa.effects.hpss(audio)
        hnr = np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-8)
        
        # 计算抖动和颤音
        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400)
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 1:
            jitter = np.std(np.diff(f0_valid)) / np.mean(f0_valid) if np.mean(f0_valid) > 0 else 0
        else:
            jitter = 0
        
        return {
            'harmonic_noise_ratio': float(hnr),
            'jitter': float(jitter),
            'spectral_flatness': float(np.mean(librosa.feature.spectral_flatness(y=audio)[0]))
        }

class EmotionClassifier(nn.Module):
    """情感分类神经网络"""
    
    def __init__(self, input_dim: int = 768, num_emotions: int = 7, dropout: float = 0.3):
        super().__init__()
        
        # Wav2Vec2特征提取器
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # 情感分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, num_emotions)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8, dropout=dropout)
        
    def forward(self, input_values, attention_mask=None):
        # Wav2Vec2特征提取
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # 应用注意力机制
        attended_features, _ = self.attention(
            hidden_states.transpose(0, 1),  # [seq_len, batch, hidden_dim]
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1)
        )
        attended_features = attended_features.transpose(0, 1)  # [batch, seq_len, hidden_dim]
        
        # 全局平均池化
        pooled_features = torch.mean(attended_features, dim=1)  # [batch, hidden_dim]
        
        # 情感分类
        logits = self.classifier(pooled_features)
        
        return logits

class RealtimeEmotionRecognizer:
    """实时情感识别器"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.emotions = [e.value for e in EmotionType]
        
        # 初始化模型
        self.model = EmotionClassifier()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化处理器
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = EmotionFeatureExtractor()
        
        # 性能统计
        self.processing_times = []
        
        logger.info("Realtime emotion recognizer initialized")
    
    async def recognize_emotion(self, audio_data: np.ndarray, sample_rate: int = 16000) -> EmotionResult:
        """识别语音情感"""
        start_time = time.time()
        
        try:
            # 预处理音频
            processed_audio = self._preprocess_audio(audio_data, sample_rate)
            
            # 提取传统音频特征
            audio_features = self.feature_extractor.extract_features(processed_audio)
            
            # 深度学习特征提取和分类
            emotion_scores = await self._classify_emotion_deep(processed_audio)
            
            # 融合传统特征和深度学习结果
            final_scores = self._fuse_predictions(emotion_scores, audio_features)
            
            # 确定主导情感
            dominant_emotion_idx = np.argmax(list(final_scores.values()))
            dominant_emotion = EmotionType(self.emotions[dominant_emotion_idx])
            confidence = float(max(final_scores.values()))
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return EmotionResult(
                dominant_emotion=dominant_emotion,
                confidence=confidence,
                emotion_scores=final_scores,
                audio_features=audio_features,
                timestamp=time.time(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Emotion recognition error: {e}")
            # 返回默认结果
            return EmotionResult(
                dominant_emotion=EmotionType.NEUTRAL,
                confidence=0.0,
                emotion_scores={emotion: 0.0 for emotion in self.emotions},
                audio_features={},
                timestamp=time.time(),
                processing_time=time.time() - start_time
            )
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """音频预处理"""
        # 重采样到16kHz
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # 标准化
        audio_data = librosa.util.normalize(audio_data)
        
        # 去除静音
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        
        # 确保最小长度
        min_length = 16000  # 1秒
        if len(audio_data) < min_length:
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')
        
        return audio_data
    
    async def _classify_emotion_deep(self, audio_data: np.ndarray) -> Dict[str, float]:
        """深度学习情感分类"""
        try:
            # 预处理音频
            inputs = self.processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            
            # 移动到设备
            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                logits = self.model(input_values, attention_mask)
                probabilities = F.softmax(logits, dim=-1)
                scores = probabilities.cpu().numpy().flatten()
            
            return dict(zip(self.emotions, scores))
            
        except Exception as e:
            logger.error(f"Deep learning classification error: {e}")
            # 返回均匀分布
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
    
    def _fuse_predictions(self, deep_scores: Dict[str, float], audio_features: Dict[str, float]) -> Dict[str, float]:
        """融合深度学习预测和传统特征"""
        # 基于音频特征的规则调整
        adjusted_scores = deep_scores.copy()
        
        try:
            # 基于能量调整
            energy = audio_features.get('energy', 0)
            if energy > 0.1:  # 高能量
                adjusted_scores['angry'] *= 1.2
                adjusted_scores['happy'] *= 1.1
                adjusted_scores['surprise'] *= 1.1
            else:  # 低能量
                adjusted_scores['sad'] *= 1.2
                adjusted_scores['neutral'] *= 1.1
            
            # 基于基频调整
            f0_mean = audio_features.get('f0_mean', 0)
            if f0_mean > 200:  # 高音调
                adjusted_scores['happy'] *= 1.2
                adjusted_scores['surprise'] *= 1.1
                adjusted_scores['fear'] *= 1.1
            elif f0_mean > 0 and f0_mean < 150:  # 低音调
                adjusted_scores['sad'] *= 1.2
                adjusted_scores['angry'] *= 1.1
            
            # 基于语音质量调整
            hnr = audio_features.get('harmonic_noise_ratio', 1)
            if hnr < 0.5:  # 低谐波噪声比（粗糙声音）
                adjusted_scores['angry'] *= 1.3
                adjusted_scores['disgust'] *= 1.1
            
            # 归一化
            total = sum(adjusted_scores.values())
            if total > 0:
                adjusted_scores = {k: v/total for k, v in adjusted_scores.items()}
            
        except Exception as e:
            logger.error(f"Prediction fusion error: {e}")
            return deep_scores
        
        return adjusted_scores
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'total_predictions': len(self.processing_times)
        }

class EmotionTracker:
    """情感跟踪器 - 跟踪情感变化趋势"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.emotion_history: List[EmotionResult] = []
        
    def add_result(self, result: EmotionResult):
        """添加情感识别结果"""
        self.emotion_history.append(result)
        
        # 保持窗口大小
        if len(self.emotion_history) > self.window_size:
            self.emotion_history.pop(0)
    
    def get_emotion_trend(self) -> Dict[str, float]:
        """获取情感趋势"""
        if not self.emotion_history:
            return {}
        
        # 计算每种情感的平均分数
        emotion_sums = {}
        for result in self.emotion_history:
            for emotion, score in result.emotion_scores.items():
                emotion_sums[emotion] = emotion_sums.get(emotion, 0) + score
        
        # 计算平均值
        emotion_trends = {
            emotion: score / len(self.emotion_history) 
            for emotion, score in emotion_sums.items()
        }
        
        return emotion_trends
    
    def get_dominant_emotion_sequence(self) -> List[str]:
        """获取主导情感序列"""
        return [result.dominant_emotion.value for result in self.emotion_history]
    
    def detect_emotion_change(self, threshold: float = 0.3) -> Optional[Tuple[str, str]]:
        """检测情感变化"""
        if len(self.emotion_history) < 2:
            return None
        
        current = self.emotion_history[-1]
        previous = self.emotion_history[-2]
        
        if (current.dominant_emotion != previous.dominant_emotion and 
            current.confidence > threshold):
            return (previous.dominant_emotion.value, current.dominant_emotion.value)
        
        return None

# 使用示例和工厂函数
async def create_emotion_recognizer(model_path: Optional[str] = None, device: str = 'cpu') -> RealtimeEmotionRecognizer:
    """创建情感识别器"""
    recognizer = RealtimeEmotionRecognizer(model_path, device)
    
    # 预热模型
    dummy_audio = np.random.randn(16000).astype(np.float32)
    await recognizer.recognize_emotion(dummy_audio)
    
    logger.info("Emotion recognizer ready")
    return recognizer

if __name__ == "__main__":
    # 测试代码
    async def test_emotion_recognition():
        recognizer = await create_emotion_recognizer()
        tracker = EmotionTracker()
        
        # 模拟音频数据
        for i in range(5):
            audio_data = np.random.randn(16000 * 2).astype(np.float32)  # 2秒音频
            result = await recognizer.recognize_emotion(audio_data)
            tracker.add_result(result)
            
            print(f"Frame {i+1}: {result.dominant_emotion.value} ({result.confidence:.3f})")
            print(f"Processing time: {result.processing_time:.3f}s")
        
        # 显示趋势
        trends = tracker.get_emotion_trend()
        print("\nEmotion trends:", trends)
        
        # 显示性能统计
        stats = recognizer.get_performance_stats()
        print("Performance stats:", stats)
    
    asyncio.run(test_emotion_recognition())
