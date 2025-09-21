"""
情感识别与表达系统 - v1.8.0
多模态情感计算，支持语音情感识别和情感化语音合成
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import deque
import librosa
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """情感类型枚举"""
    NEUTRAL = "neutral"      # 中性
    HAPPY = "happy"          # 开心
    SAD = "sad"              # 悲伤
    ANGRY = "angry"          # 愤怒
    SURPRISED = "surprised"  # 惊讶
    FEARFUL = "fearful"      # 恐惧
    DISGUSTED = "disgusted"  # 厌恶
    EXCITED = "excited"      # 兴奋
    CALM = "calm"           # 平静
    CONFUSED = "confused"    # 困惑

class ModalityType(Enum):
    """模态类型"""
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"

@dataclass
class EmotionScore:
    """情感得分"""
    emotion: EmotionType
    confidence: float
    intensity: float  # 情感强度 0-1
    modality: ModalityType
    timestamp: float = field(default_factory=time.time)

@dataclass
class EmotionContext:
    """情感上下文"""
    user_emotion_history: List[EmotionScore] = field(default_factory=list)
    conversation_emotion_trend: List[EmotionScore] = field(default_factory=list)
    current_dominant_emotion: Optional[EmotionScore] = None
    emotion_transition_count: int = 0

class AudioEmotionExtractor:
    """音频情感特征提取器"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.frame_length = 1024
        self.hop_length = 512
        
    async def extract_audio_features(self, audio_data: bytes) -> Dict[str, np.ndarray]:
        """
        提取音频情感特征
        
        Args:
            audio_data: 音频数据
            
        Returns:
            Dict: 音频特征字典
        """
        try:
            # 将bytes转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # 提取各种音频特征
            features = {}
            
            # 1. 基频特征 (F0)
            f0 = self._extract_f0(audio_array)
            features['f0_mean'] = np.mean(f0[f0 > 0])
            features['f0_std'] = np.std(f0[f0 > 0])
            features['f0_range'] = np.max(f0) - np.min(f0[f0 > 0])
            
            # 2. 能量特征
            energy = np.sum(audio_array ** 2)
            features['energy'] = energy
            features['rms_energy'] = np.sqrt(np.mean(audio_array ** 2))
            
            # 3. 频谱特征
            stft = librosa.stft(audio_array, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            
            # 频谱重心
            spectral_centroid = librosa.feature.spectral_centroid(S=magnitude)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            # 频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            
            # 4. MFCC特征
            mfcc = librosa.feature.mfcc(y=audio_array, sr=self.sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
                features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            
            # 5. 语音节奏特征
            tempo, _ = librosa.beat.beat_track(y=audio_array, sr=self.sample_rate)
            features['tempo'] = tempo
            
            # 6. 零交叉率
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction error: {e}")
            return {}
    
    def _extract_f0(self, audio_array: np.ndarray) -> np.ndarray:
        """提取基频"""
        try:
            # 使用简单的自相关方法提取F0
            # 实际应用中可以使用更高级的算法如PYIN
            autocorr = np.correlate(audio_array, audio_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # 寻找峰值
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peaks.append(i)
            
            if peaks:
                # 估算基频
                period = peaks[0] if peaks else len(autocorr) // 4
                f0 = self.sample_rate / period if period > 0 else 0
                return np.array([f0])
            else:
                return np.array([0])
                
        except Exception:
            return np.array([0])

class TextEmotionAnalyzer:
    """文本情感分析器"""
    
    def __init__(self):
        # 情感词典 - 实际应用中应使用更完整的词典
        self.emotion_lexicon = {
            EmotionType.HAPPY: ['开心', '高兴', '快乐', '愉快', '兴奋', '满意', '喜悦', '欣喜'],
            EmotionType.SAD: ['难过', '悲伤', '沮丧', '失落', '郁闷', '痛苦', '忧伤', '哀伤'],
            EmotionType.ANGRY: ['愤怒', '生气', '恼火', '暴躁', '愤慨', '恼怒', '气愤', '愤恨'],
            EmotionType.SURPRISED: ['惊讶', '震惊', '意外', '吃惊', '惊奇', '惊愕', '诧异'],
            EmotionType.FEARFUL: ['害怕', '恐惧', '担心', '忧虑', '紧张', '焦虑', '不安'],
            EmotionType.DISGUSTED: ['厌恶', '恶心', '反感', '讨厌', '憎恶', '嫌弃'],
            EmotionType.EXCITED: ['激动', '兴奋', '热情', '狂热', '亢奋', '振奋'],
            EmotionType.CALM: ['平静', '冷静', '淡定', '安静', '宁静', '从容'],
            EmotionType.CONFUSED: ['困惑', '迷惑', '疑惑', '不解', '茫然', '纳闷']
        }
        
        # 情感强度词
        self.intensity_modifiers = {
            '非常': 1.5, '特别': 1.4, '极其': 1.6, '超级': 1.3,
            '有点': 0.6, '稍微': 0.5, '略微': 0.4, '一点': 0.3
        }
    
    async def analyze_text_emotion(self, text: str) -> List[EmotionScore]:
        """
        分析文本情感
        
        Args:
            text: 输入文本
            
        Returns:
            List[EmotionScore]: 情感得分列表
        """
        try:
            emotions = []
            text_lower = text.lower()
            
            # 遍历每种情感类型
            for emotion_type, keywords in self.emotion_lexicon.items():
                emotion_score = 0
                emotion_count = 0
                intensity_multiplier = 1.0
                
                # 检查情感词
                for keyword in keywords:
                    if keyword in text_lower:
                        emotion_count += 1
                        emotion_score += 1
                
                # 检查强度修饰词
                for modifier, multiplier in self.intensity_modifiers.items():
                    if modifier in text_lower:
                        intensity_multiplier = max(intensity_multiplier, multiplier)
                
                if emotion_score > 0:
                    # 计算置信度和强度
                    confidence = min(emotion_score / len(keywords), 1.0)
                    intensity = min(emotion_score * intensity_multiplier / 3, 1.0)
                    
                    emotions.append(EmotionScore(
                        emotion=emotion_type,
                        confidence=confidence,
                        intensity=intensity,
                        modality=ModalityType.TEXT
                    ))
            
            # 如果没有检测到明显情感，返回中性情感
            if not emotions:
                emotions.append(EmotionScore(
                    emotion=EmotionType.NEUTRAL,
                    confidence=0.8,
                    intensity=0.5,
                    modality=ModalityType.TEXT
                ))
            
            # 按置信度排序
            emotions.sort(key=lambda x: x.confidence, reverse=True)
            
            return emotions
            
        except Exception as e:
            logger.error(f"Text emotion analysis error: {e}")
            return [EmotionScore(
                emotion=EmotionType.NEUTRAL,
                confidence=0.5,
                intensity=0.5,
                modality=ModalityType.TEXT
            )]

class MultiModalEmotionFusion:
    """多模态情感融合器"""
    
    def __init__(self):
        # 模态权重
        self.modality_weights = {
            ModalityType.AUDIO: 0.6,  # 语音情感权重更高
            ModalityType.TEXT: 0.4
        }
        
        # 情感兼容性矩阵
        self.emotion_compatibility = self._build_compatibility_matrix()
    
    def _build_compatibility_matrix(self) -> Dict[Tuple[EmotionType, EmotionType], float]:
        """构建情感兼容性矩阵"""
        # 定义情感之间的兼容性得分 (0-1)
        compatibility = {}
        
        emotions = list(EmotionType)
        for e1 in emotions:
            for e2 in emotions:
                if e1 == e2:
                    compatibility[(e1, e2)] = 1.0
                elif (e1, e2) in [(EmotionType.HAPPY, EmotionType.EXCITED), 
                                  (EmotionType.EXCITED, EmotionType.HAPPY),
                                  (EmotionType.SAD, EmotionType.FEARFUL),
                                  (EmotionType.FEARFUL, EmotionType.SAD),
                                  (EmotionType.CALM, EmotionType.NEUTRAL),
                                  (EmotionType.NEUTRAL, EmotionType.CALM)]:
                    compatibility[(e1, e2)] = 0.8
                elif (e1, e2) in [(EmotionType.HAPPY, EmotionType.SAD),
                                  (EmotionType.SAD, EmotionType.HAPPY),
                                  (EmotionType.ANGRY, EmotionType.CALM),
                                  (EmotionType.CALM, EmotionType.ANGRY)]:
                    compatibility[(e1, e2)] = 0.1
                else:
                    compatibility[(e1, e2)] = 0.5
        
        return compatibility
    
    async def fuse_emotions(self, 
                          audio_emotions: List[EmotionScore], 
                          text_emotions: List[EmotionScore]) -> List[EmotionScore]:
        """
        融合多模态情感
        
        Args:
            audio_emotions: 音频情感列表
            text_emotions: 文本情感列表
            
        Returns:
            List[EmotionScore]: 融合后的情感列表
        """
        try:
            fused_emotions = {}
            
            # 处理音频情感
            for audio_emotion in audio_emotions:
                emotion_type = audio_emotion.emotion
                weighted_score = (audio_emotion.confidence * audio_emotion.intensity * 
                                self.modality_weights[ModalityType.AUDIO])
                
                if emotion_type not in fused_emotions:
                    fused_emotions[emotion_type] = {
                        'total_score': 0,
                        'max_confidence': 0,
                        'max_intensity': 0,
                        'modalities': set()
                    }
                
                fused_emotions[emotion_type]['total_score'] += weighted_score
                fused_emotions[emotion_type]['max_confidence'] = max(
                    fused_emotions[emotion_type]['max_confidence'], 
                    audio_emotion.confidence
                )
                fused_emotions[emotion_type]['max_intensity'] = max(
                    fused_emotions[emotion_type]['max_intensity'], 
                    audio_emotion.intensity
                )
                fused_emotions[emotion_type]['modalities'].add(ModalityType.AUDIO)
            
            # 处理文本情感
            for text_emotion in text_emotions:
                emotion_type = text_emotion.emotion
                weighted_score = (text_emotion.confidence * text_emotion.intensity * 
                                self.modality_weights[ModalityType.TEXT])
                
                if emotion_type not in fused_emotions:
                    fused_emotions[emotion_type] = {
                        'total_score': 0,
                        'max_confidence': 0,
                        'max_intensity': 0,
                        'modalities': set()
                    }
                
                fused_emotions[emotion_type]['total_score'] += weighted_score
                fused_emotions[emotion_type]['max_confidence'] = max(
                    fused_emotions[emotion_type]['max_confidence'], 
                    text_emotion.confidence
                )
                fused_emotions[emotion_type]['max_intensity'] = max(
                    fused_emotions[emotion_type]['max_intensity'], 
                    text_emotion.intensity
                )
                fused_emotions[emotion_type]['modalities'].add(ModalityType.TEXT)
            
            # 应用跨模态一致性增强
            fused_emotions = self._apply_cross_modal_consistency(fused_emotions)
            
            # 转换为EmotionScore列表
            result_emotions = []
            for emotion_type, data in fused_emotions.items():
                # 多模态融合的置信度更高
                confidence_boost = 1.2 if len(data['modalities']) > 1 else 1.0
                final_confidence = min(data['max_confidence'] * confidence_boost, 1.0)
                
                result_emotions.append(EmotionScore(
                    emotion=emotion_type,
                    confidence=final_confidence,
                    intensity=data['max_intensity'],
                    modality=ModalityType.MULTIMODAL if len(data['modalities']) > 1 else list(data['modalities'])[0]
                ))
            
            # 按总得分排序
            result_emotions.sort(key=lambda x: x.confidence * x.intensity, reverse=True)
            
            return result_emotions
            
        except Exception as e:
            logger.error(f"Emotion fusion error: {e}")
            return [EmotionScore(
                emotion=EmotionType.NEUTRAL,
                confidence=0.5,
                intensity=0.5,
                modality=ModalityType.MULTIMODAL
            )]
    
    def _apply_cross_modal_consistency(self, fused_emotions: Dict) -> Dict:
        """应用跨模态一致性增强"""
        # 如果多个模态检测到兼容的情感，增强其得分
        emotion_types = list(fused_emotions.keys())
        
        for i, emotion1 in enumerate(emotion_types):
            for j, emotion2 in enumerate(emotion_types[i+1:], i+1):
                compatibility = self.emotion_compatibility.get((emotion1, emotion2), 0.5)
                
                if compatibility > 0.7:  # 高兼容性
                    # 增强两个情感的得分
                    boost_factor = 1.1
                    fused_emotions[emotion1]['total_score'] *= boost_factor
                    fused_emotions[emotion2]['total_score'] *= boost_factor
        
        return fused_emotions

class EmotionalTTSController:
    """情感化TTS控制器"""
    
    def __init__(self):
        # 情感到语音参数的映射
        self.emotion_voice_params = {
            EmotionType.HAPPY: {
                'pitch_shift': 1.2,      # 音调提高
                'speed_rate': 1.1,       # 语速稍快
                'volume_gain': 1.0,      # 音量正常
                'prosody_style': 'cheerful'
            },
            EmotionType.SAD: {
                'pitch_shift': 0.8,      # 音调降低
                'speed_rate': 0.9,       # 语速稍慢
                'volume_gain': 0.8,      # 音量降低
                'prosody_style': 'sad'
            },
            EmotionType.ANGRY: {
                'pitch_shift': 1.3,      # 音调提高
                'speed_rate': 1.2,       # 语速加快
                'volume_gain': 1.2,      # 音量提高
                'prosody_style': 'angry'
            },
            EmotionType.EXCITED: {
                'pitch_shift': 1.4,      # 音调大幅提高
                'speed_rate': 1.3,       # 语速明显加快
                'volume_gain': 1.1,      # 音量稍高
                'prosody_style': 'excited'
            },
            EmotionType.CALM: {
                'pitch_shift': 1.0,      # 音调正常
                'speed_rate': 0.95,      # 语速稍慢
                'volume_gain': 0.9,      # 音量稍低
                'prosody_style': 'calm'
            },
            EmotionType.NEUTRAL: {
                'pitch_shift': 1.0,      # 音调正常
                'speed_rate': 1.0,       # 语速正常
                'volume_gain': 1.0,      # 音量正常
                'prosody_style': 'neutral'
            }
        }
    
    async def generate_emotional_speech(self, 
                                      text: str, 
                                      emotion: EmotionScore) -> Dict[str, Any]:
        """
        生成情感化语音
        
        Args:
            text: 要合成的文本
            emotion: 情感信息
            
        Returns:
            Dict: 包含音频数据和参数的字典
        """
        try:
            # 获取情感对应的语音参数
            voice_params = self.emotion_voice_params.get(
                emotion.emotion, 
                self.emotion_voice_params[EmotionType.NEUTRAL]
            )
            
            # 根据情感强度调整参数
            intensity_factor = emotion.intensity
            adjusted_params = {
                'pitch_shift': 1.0 + (voice_params['pitch_shift'] - 1.0) * intensity_factor,
                'speed_rate': 1.0 + (voice_params['speed_rate'] - 1.0) * intensity_factor,
                'volume_gain': 1.0 + (voice_params['volume_gain'] - 1.0) * intensity_factor,
                'prosody_style': voice_params['prosody_style']
            }
            
            # 模拟TTS合成（实际应调用真实TTS服务）
            await asyncio.sleep(0.05)  # 模拟合成延迟
            
            # 生成模拟音频数据
            audio_data = self._generate_mock_emotional_audio(text, adjusted_params)
            
            return {
                'audio_data': audio_data,
                'emotion': emotion.emotion.value,
                'intensity': emotion.intensity,
                'voice_params': adjusted_params,
                'synthesis_time': 0.05
            }
            
        except Exception as e:
            logger.error(f"Emotional TTS generation error: {e}")
            return {
                'audio_data': b'',
                'error': str(e)
            }
    
    def _generate_mock_emotional_audio(self, text: str, params: Dict[str, Any]) -> bytes:
        """生成模拟的情感音频数据"""
        # 这里应该是真实的TTS合成逻辑
        # 根据参数调整音频特征
        base_length = len(text) * 100  # 基础音频长度
        
        # 根据语速调整长度
        adjusted_length = int(base_length / params['speed_rate'])
        
        # 生成模拟音频数据
        audio_data = np.random.bytes(adjusted_length)
        
        return audio_data

class EmotionRecognizer:
    """主情感识别器"""
    
    def __init__(self):
        self.audio_extractor = AudioEmotionExtractor()
        self.text_analyzer = TextEmotionAnalyzer()
        self.fusion_engine = MultiModalEmotionFusion()
        self.tts_controller = EmotionalTTSController()
        self.emotion_context = EmotionContext()
        
    async def recognize_emotion(self, 
                              audio_data: Optional[bytes] = None, 
                              text_data: Optional[str] = None) -> Dict[str, Any]:
        """
        多模态情感识别
        
        Args:
            audio_data: 音频数据
            text_data: 文本数据
            
        Returns:
            Dict: 情感识别结果
        """
        try:
            start_time = time.time()
            
            audio_emotions = []
            text_emotions = []
            
            # 并行处理音频和文本情感
            tasks = []
            
            if audio_data:
                tasks.append(asyncio.create_task(self._recognize_audio_emotion(audio_data)))
            
            if text_data:
                tasks.append(asyncio.create_task(self.text_analyzer.analyze_text_emotion(text_data)))
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            result_idx = 0
            if audio_data:
                audio_emotions = results[result_idx] if not isinstance(results[result_idx], Exception) else []
                result_idx += 1
            
            if text_data:
                text_emotions = results[result_idx] if not isinstance(results[result_idx], Exception) else []
            
            # 多模态融合
            if audio_emotions and text_emotions:
                fused_emotions = await self.fusion_engine.fuse_emotions(audio_emotions, text_emotions)
            elif audio_emotions:
                fused_emotions = audio_emotions
            elif text_emotions:
                fused_emotions = text_emotions
            else:
                fused_emotions = [EmotionScore(
                    emotion=EmotionType.NEUTRAL,
                    confidence=0.5,
                    intensity=0.5,
                    modality=ModalityType.MULTIMODAL
                )]
            
            # 更新情感上下文
            self._update_emotion_context(fused_emotions)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'emotions': fused_emotions,
                'dominant_emotion': fused_emotions[0] if fused_emotions else None,
                'audio_emotions': audio_emotions,
                'text_emotions': text_emotions,
                'emotion_context': self.emotion_context,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Emotion recognition error: {e}")
            return {
                'error': str(e),
                'emotions': []
            }
    
    async def _recognize_audio_emotion(self, audio_data: bytes) -> List[EmotionScore]:
        """识别音频情感"""
        try:
            # 提取音频特征
            features = await self.audio_extractor.extract_audio_features(audio_data)
            
            if not features:
                return []
            
            # 基于特征的简单情感分类（实际应使用训练好的模型）
            emotions = []
            
            # 基于基频判断情感
            f0_mean = features.get('f0_mean', 0)
            if f0_mean > 200:  # 高音调
                emotions.append(EmotionScore(
                    emotion=EmotionType.EXCITED,
                    confidence=0.7,
                    intensity=0.8,
                    modality=ModalityType.AUDIO
                ))
            elif f0_mean < 100:  # 低音调
                emotions.append(EmotionScore(
                    emotion=EmotionType.SAD,
                    confidence=0.6,
                    intensity=0.6,
                    modality=ModalityType.AUDIO
                ))
            
            # 基于能量判断情感
            energy = features.get('energy', 0)
            if energy > 1000:  # 高能量
                emotions.append(EmotionScore(
                    emotion=EmotionType.ANGRY,
                    confidence=0.6,
                    intensity=0.7,
                    modality=ModalityType.AUDIO
                ))
            elif energy < 100:  # 低能量
                emotions.append(EmotionScore(
                    emotion=EmotionType.CALM,
                    confidence=0.5,
                    intensity=0.5,
                    modality=ModalityType.AUDIO
                ))
            
            # 如果没有明显特征，返回中性
            if not emotions:
                emotions.append(EmotionScore(
                    emotion=EmotionType.NEUTRAL,
                    confidence=0.5,
                    intensity=0.5,
                    modality=ModalityType.AUDIO
                ))
            
            return emotions
            
        except Exception as e:
            logger.error(f"Audio emotion recognition error: {e}")
            return []
    
    def _update_emotion_context(self, emotions: List[EmotionScore]):
        """更新情感上下文"""
        if not emotions:
            return
        
        current_emotion = emotions[0]
        
        # 更新用户情感历史
        self.emotion_context.user_emotion_history.append(current_emotion)
        if len(self.emotion_context.user_emotion_history) > 100:
            self.emotion_context.user_emotion_history.pop(0)
        
        # 更新对话情感趋势
        self.emotion_context.conversation_emotion_trend.append(current_emotion)
        if len(self.emotion_context.conversation_emotion_trend) > 20:
            self.emotion_context.conversation_emotion_trend.pop(0)
        
        # 检测情感转换
        if (self.emotion_context.current_dominant_emotion and 
            self.emotion_context.current_dominant_emotion.emotion != current_emotion.emotion):
            self.emotion_context.emotion_transition_count += 1
        
        # 更新当前主导情感
        self.emotion_context.current_dominant_emotion = current_emotion
    
    async def generate_emotional_response(self, 
                                        response_text: str, 
                                        target_emotion: Optional[EmotionScore] = None) -> Dict[str, Any]:
        """
        生成情感化回复
        
        Args:
            response_text: 回复文本
            target_emotion: 目标情感（如果不指定，使用上下文情感）
            
        Returns:
            Dict: 情感化回复结果
        """
        try:
            # 确定目标情感
            if not target_emotion:
                target_emotion = self._determine_response_emotion()
            
            # 生成情感化语音
            tts_result = await self.tts_controller.generate_emotional_speech(
                response_text, target_emotion
            )
            
            return {
                'text': response_text,
                'target_emotion': target_emotion,
                'audio_data': tts_result.get('audio_data'),
                'voice_params': tts_result.get('voice_params'),
                'synthesis_time': tts_result.get('synthesis_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Emotional response generation error: {e}")
            return {
                'text': response_text,
                'error': str(e)
            }
    
    def _determine_response_emotion(self) -> EmotionScore:
        """确定回复情感"""
        # 基于用户情感历史和对话趋势确定合适的回复情感
        if not self.emotion_context.current_dominant_emotion:
            return EmotionScore(
                emotion=EmotionType.NEUTRAL,
                confidence=0.8,
                intensity=0.5,
                modality=ModalityType.MULTIMODAL
            )
        
        user_emotion = self.emotion_context.current_dominant_emotion.emotion
        
        # 情感回应策略
        response_emotion_map = {
            EmotionType.HAPPY: EmotionType.HAPPY,
            EmotionType.EXCITED: EmotionType.HAPPY,
            EmotionType.SAD: EmotionType.CALM,
            EmotionType.ANGRY: EmotionType.CALM,
            EmotionType.FEARFUL: EmotionType.CALM,
            EmotionType.CONFUSED: EmotionType.CALM,
            EmotionType.NEUTRAL: EmotionType.NEUTRAL,
            EmotionType.CALM: EmotionType.CALM
        }
        
        response_emotion_type = response_emotion_map.get(user_emotion, EmotionType.NEUTRAL)
        
        return EmotionScore(
            emotion=response_emotion_type,
            confidence=0.8,
            intensity=0.6,
            modality=ModalityType.MULTIMODAL
        )
    
    def get_emotion_analytics(self) -> Dict[str, Any]:
        """获取情感分析统计"""
        if not self.emotion_context.user_emotion_history:
            return {}
        
        # 统计各种情感的分布
        emotion_counts = {}
        total_intensity = 0
        
        for emotion_score in self.emotion_context.user_emotion_history:
            emotion_type = emotion_score.emotion.value
            emotion_counts[emotion_type] = emotion_counts.get(emotion_type, 0) + 1
            total_intensity += emotion_score.intensity
        
        # 计算平均情感强度
        avg_intensity = total_intensity / len(self.emotion_context.user_emotion_history)
        
        # 最常见的情感
        most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            'emotion_distribution': emotion_counts,
            'average_intensity': avg_intensity,
            'most_common_emotion': most_common_emotion[0],
            'emotion_transition_count': self.emotion_context.emotion_transition_count,
            'total_interactions': len(self.emotion_context.user_emotion_history)
        }

# 使用示例
async def main():
    """示例用法"""
    recognizer = EmotionRecognizer()
    
    # 模拟音频和文本数据
    mock_audio = b"mock_audio_data" * 100
    mock_text = "我今天非常开心，因为完成了一个重要的项目！"
    
    # 情感识别
    result = await recognizer.recognize_emotion(
        audio_data=mock_audio,
        text_data=mock_text
    )
    
    print("=== 情感识别结果 ===")
    if result.get('emotions'):
        for i, emotion in enumerate(result['emotions'][:3]):
            print(f"{i+1}. {emotion.emotion.value}: 置信度={emotion.confidence:.2f}, 强度={emotion.intensity:.2f}")
    
    print(f"处理时间: {result.get('processing_time_ms', 0):.2f}ms")
    
    # 生成情感化回复
    response_result = await recognizer.generate_emotional_response(
        "恭喜你完成项目！这真是太棒了！"
    )
    
    print("\n=== 情感化回复 ===")
    print(f"回复文本: {response_result['text']}")
    if response_result.get('target_emotion'):
        emotion = response_result['target_emotion']
        print(f"目标情感: {emotion.emotion.value} (强度: {emotion.intensity:.2f})")
    
    # 情感分析统计
    analytics = recognizer.get_emotion_analytics()
    if analytics:
        print("\n=== 情感统计 ===")
        print(f"情感分布: {analytics.get('emotion_distribution', {})}")
        print(f"平均强度: {analytics.get('average_intensity', 0):.2f}")
        print(f"最常见情感: {analytics.get('most_common_emotion', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
