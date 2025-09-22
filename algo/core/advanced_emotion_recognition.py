"""
VoiceHelper v1.20.0 - 高级情感识别系统
实现多模态情感分析、情感融合和历史上下文学习
"""

import asyncio
import time
import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class EmotionAnalysisResult:
    """情感分析结果"""
    primary_emotion: str
    confidence: float
    emotion_vector: Dict[str, float]
    temporal_pattern: Dict[str, float]
    processing_time: float

@dataclass
class AudioEmotionFeatures:
    """音频情感特征"""
    pitch_mean: float
    pitch_variance: float
    energy_mean: float
    energy_variance: float
    spectral_features: Dict[str, float]
    prosodic_features: Dict[str, float]

@dataclass
class TextEmotionFeatures:
    """文本情感特征"""
    sentiment_score: float
    emotion_keywords: List[str]
    linguistic_features: Dict[str, float]
    semantic_features: Dict[str, float]

class EmotionHistory:
    """情感历史管理器"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.user_histories = defaultdict(lambda: deque(maxlen=max_history))
        self.emotion_transitions = defaultdict(lambda: defaultdict(int))
        
    def update(self, user_id: str, emotion_result: EmotionAnalysisResult):
        """更新用户情感历史"""
        history = self.user_histories[user_id]
        
        # 记录情感转换
        if history:
            prev_emotion = history[-1].primary_emotion
            curr_emotion = emotion_result.primary_emotion
            self.emotion_transitions[user_id][f"{prev_emotion}->{curr_emotion}"] += 1
        
        # 添加到历史记录
        history.append(emotion_result)
        
    def get_context(self, user_id: str) -> Dict:
        """获取用户情感上下文"""
        history = self.user_histories[user_id]
        if not history:
            return {"recent_emotions": [], "dominant_emotion": "neutral", "stability": 1.0}
        
        # 分析最近的情感
        recent_emotions = [item.primary_emotion for item in list(history)[-10:]]
        
        # 计算主导情感
        emotion_counts = defaultdict(int)
        for emotion in recent_emotions:
            emotion_counts[emotion] += 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        
        # 计算情感稳定性
        stability = self._calculate_emotional_stability(recent_emotions)
        
        return {
            "recent_emotions": recent_emotions,
            "dominant_emotion": dominant_emotion,
            "stability": stability,
            "transition_patterns": dict(self.emotion_transitions[user_id])
        }
    
    def _calculate_emotional_stability(self, emotions: List[str]) -> float:
        """计算情感稳定性"""
        if len(emotions) < 2:
            return 1.0
        
        # 计算情感变化次数
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        stability = 1.0 - (changes / (len(emotions) - 1))
        
        return max(0.0, min(1.0, stability))

class AudioEmotionModel:
    """音频情感分析模型"""
    
    def __init__(self):
        self.emotion_classes = ["happy", "sad", "angry", "neutral", "excited", "calm", "frustrated"]
        self.feature_extractor = AudioFeatureExtractor()
        
    async def analyze(self, audio: bytes) -> Dict[str, float]:
        """分析音频情感"""
        try:
            # 提取音频特征
            features = await self.feature_extractor.extract_features(audio)
            
            # 模拟情感分析模型
            emotion_scores = await self._predict_emotions(features)
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Audio emotion analysis error: {e}")
            return {"neutral": 1.0}
    
    async def _predict_emotions(self, features: AudioEmotionFeatures) -> Dict[str, float]:
        """预测情感分数"""
        # 模拟深度学习模型预测
        await asyncio.sleep(0.02)  # 模拟计算时间
        
        # 基于特征的简化情感预测
        scores = {}
        
        # 基于音调特征
        if features.pitch_mean > 200:  # 高音调
            scores["happy"] = 0.6
            scores["excited"] = 0.3
        elif features.pitch_mean < 150:  # 低音调
            scores["sad"] = 0.5
            scores["calm"] = 0.4
        else:
            scores["neutral"] = 0.7
        
        # 基于能量特征
        if features.energy_mean > 0.8:
            scores["angry"] = scores.get("angry", 0) + 0.3
            scores["excited"] = scores.get("excited", 0) + 0.2
        elif features.energy_mean < 0.3:
            scores["sad"] = scores.get("sad", 0) + 0.2
            scores["calm"] = scores.get("calm", 0) + 0.3
        
        # 归一化分数
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {"neutral": 1.0}
        
        return scores

class AudioFeatureExtractor:
    """音频特征提取器"""
    
    async def extract_features(self, audio: bytes) -> AudioEmotionFeatures:
        """提取音频特征"""
        # 模拟音频特征提取
        await asyncio.sleep(0.01)
        
        # 模拟特征值
        return AudioEmotionFeatures(
            pitch_mean=random.gauss(180, 30),
            pitch_variance=random.gauss(20, 5),
            energy_mean=random.gauss(0.5, 0.2),
            energy_variance=random.gauss(0.1, 0.05),
            spectral_features={
                "spectral_centroid": random.gauss(2000, 500),
                "spectral_rolloff": random.gauss(4000, 1000),
                "mfcc_mean": random.gauss(0, 1)
            },
            prosodic_features={
                "speaking_rate": random.gauss(150, 30),
                "pause_duration": random.gauss(0.5, 0.2)
            }
        )

class TextEmotionModel:
    """文本情感分析模型"""
    
    def __init__(self):
        self.emotion_keywords = {
            "happy": ["开心", "高兴", "快乐", "兴奋", "愉快", "满意"],
            "sad": ["难过", "伤心", "沮丧", "失望", "痛苦", "悲伤"],
            "angry": ["生气", "愤怒", "恼火", "烦躁", "不满", "愤慨"],
            "neutral": ["好的", "知道", "明白", "了解", "清楚"],
            "excited": ["激动", "兴奋", "热情", "期待", "振奋"],
            "calm": ["平静", "冷静", "安静", "放松", "淡定"]
        }
        
    async def analyze(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        try:
            # 提取文本特征
            features = await self._extract_text_features(text)
            
            # 预测情感
            emotion_scores = await self._predict_text_emotions(features)
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Text emotion analysis error: {e}")
            return {"neutral": 1.0}
    
    async def _extract_text_features(self, text: str) -> TextEmotionFeatures:
        """提取文本特征"""
        await asyncio.sleep(0.01)  # 模拟处理时间
        
        # 检测情感关键词
        detected_keywords = []
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected_keywords.append((emotion, keyword))
        
        # 计算情感分数
        sentiment_score = self._calculate_sentiment_score(text)
        
        return TextEmotionFeatures(
            sentiment_score=sentiment_score,
            emotion_keywords=detected_keywords,
            linguistic_features={
                "text_length": len(text),
                "exclamation_count": text.count("!") + text.count("！"),
                "question_count": text.count("?") + text.count("？")
            },
            semantic_features={
                "positive_words": len([w for w in text if w in ["好", "棒", "赞"]]),
                "negative_words": len([w for w in text if w in ["不", "没", "坏"]])
            }
        )
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """计算情感倾向分数"""
        positive_words = ["好", "棒", "赞", "喜欢", "满意", "开心"]
        negative_words = ["不好", "差", "讨厌", "不满", "难过", "生气"]
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count + neg_count == 0:
            return 0.0  # 中性
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    async def _predict_text_emotions(self, features: TextEmotionFeatures) -> Dict[str, float]:
        """基于文本特征预测情感"""
        scores = defaultdict(float)
        
        # 基于关键词
        for emotion, keyword in features.emotion_keywords:
            scores[emotion] += 0.3
        
        # 基于情感倾向
        if features.sentiment_score > 0.3:
            scores["happy"] += 0.4
        elif features.sentiment_score < -0.3:
            scores["sad"] += 0.4
        else:
            scores["neutral"] += 0.4
        
        # 基于语言特征
        if features.linguistic_features["exclamation_count"] > 0:
            scores["excited"] += 0.2
            scores["angry"] += 0.1
        
        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {"neutral": 1.0}
        
        return dict(scores)

class EmotionFusionModel:
    """情感融合模型"""
    
    def __init__(self):
        self.audio_weight = 0.6  # 音频情感权重
        self.text_weight = 0.4   # 文本情感权重
        self.history_weight = 0.2  # 历史情感权重
        
    def fuse(
        self, 
        audio_emotion: Dict[str, float], 
        text_emotion: Dict[str, float], 
        historical_context: Dict
    ) -> EmotionAnalysisResult:
        """融合多模态情感"""
        
        # 获取所有可能的情感类别
        all_emotions = set(audio_emotion.keys()) | set(text_emotion.keys())
        
        # 融合情感分数
        fused_scores = {}
        for emotion in all_emotions:
            audio_score = audio_emotion.get(emotion, 0.0)
            text_score = text_emotion.get(emotion, 0.0)
            
            # 加权融合
            fused_score = (
                audio_score * self.audio_weight + 
                text_score * self.text_weight
            )
            
            # 考虑历史上下文
            if historical_context.get("dominant_emotion") == emotion:
                fused_score *= (1 + self.history_weight)
            
            fused_scores[emotion] = fused_score
        
        # 归一化
        total = sum(fused_scores.values())
        if total > 0:
            fused_scores = {k: v/total for k, v in fused_scores.items()}
        
        # 确定主要情感
        primary_emotion = max(fused_scores.items(), key=lambda x: x[1])[0]
        confidence = fused_scores[primary_emotion]
        
        # 计算时间模式
        temporal_pattern = self._calculate_temporal_pattern(
            historical_context.get("recent_emotions", [])
        )
        
        return EmotionAnalysisResult(
            primary_emotion=primary_emotion,
            confidence=confidence,
            emotion_vector=fused_scores,
            temporal_pattern=temporal_pattern,
            processing_time=0.0  # 将在调用处设置
        )
    
    def _calculate_temporal_pattern(self, recent_emotions: List[str]) -> Dict[str, float]:
        """计算时间情感模式"""
        if not recent_emotions:
            return {"stability": 1.0, "trend": "stable"}
        
        # 计算情感稳定性
        unique_emotions = len(set(recent_emotions))
        stability = 1.0 - (unique_emotions - 1) / max(1, len(recent_emotions))
        
        # 分析情感趋势
        if len(recent_emotions) >= 3:
            recent_3 = recent_emotions[-3:]
            if all(e == recent_3[0] for e in recent_3):
                trend = "stable"
            elif recent_3[-1] in ["happy", "excited"] and recent_3[0] in ["sad", "angry"]:
                trend = "improving"
            elif recent_3[-1] in ["sad", "angry"] and recent_3[0] in ["happy", "excited"]:
                trend = "declining"
            else:
                trend = "fluctuating"
        else:
            trend = "stable"
        
        return {
            "stability": stability,
            "trend": trend,
            "emotion_diversity": unique_emotions / max(1, len(recent_emotions))
        }

class AdvancedEmotionRecognition:
    """v1.20.0 高级情感识别系统"""
    
    def __init__(self):
        self.audio_emotion_model = AudioEmotionModel()
        self.text_emotion_model = TextEmotionModel()
        self.fusion_model = EmotionFusionModel()
        self.emotion_history = EmotionHistory()
        
    async def analyze_multimodal_emotion(
        self, 
        audio: bytes, 
        text: str, 
        user_id: str
    ) -> EmotionAnalysisResult:
        """多模态情感分析"""
        start_time = time.time()
        
        try:
            # 并行分析音频和文本情感
            audio_emotion_task = self.audio_emotion_model.analyze(audio)
            text_emotion_task = self.text_emotion_model.analyze(text)
            
            audio_emotion, text_emotion = await asyncio.gather(
                audio_emotion_task, text_emotion_task
            )
            
            # 获取历史上下文
            historical_context = self.emotion_history.get_context(user_id)
            
            # 情感融合
            fused_emotion = self.fusion_model.fuse(
                audio_emotion=audio_emotion,
                text_emotion=text_emotion,
                historical_context=historical_context
            )
            
            # 设置处理时间
            fused_emotion.processing_time = time.time() - start_time
            
            # 更新情感历史
            self.emotion_history.update(user_id, fused_emotion)
            
            return fused_emotion
            
        except Exception as e:
            logger.error(f"Multimodal emotion analysis error: {e}")
            # 返回默认情感结果
            return EmotionAnalysisResult(
                primary_emotion="neutral",
                confidence=0.5,
                emotion_vector={"neutral": 1.0},
                temporal_pattern={"stability": 1.0, "trend": "stable"},
                processing_time=time.time() - start_time
            )
    
    async def analyze_text_only(self, text: str, user_id: str) -> EmotionAnalysisResult:
        """仅文本情感分析"""
        start_time = time.time()
        
        try:
            # 文本情感分析
            text_emotion = await self.text_emotion_model.analyze(text)
            
            # 获取历史上下文
            historical_context = self.emotion_history.get_context(user_id)
            
            # 创建空的音频情感（全部为0）
            empty_audio_emotion = {emotion: 0.0 for emotion in text_emotion.keys()}
            
            # 情感融合（主要基于文本）
            fused_emotion = self.fusion_model.fuse(
                audio_emotion=empty_audio_emotion,
                text_emotion=text_emotion,
                historical_context=historical_context
            )
            
            # 设置处理时间
            fused_emotion.processing_time = time.time() - start_time
            
            # 更新情感历史
            self.emotion_history.update(user_id, fused_emotion)
            
            return fused_emotion
            
        except Exception as e:
            logger.error(f"Text emotion analysis error: {e}")
            return EmotionAnalysisResult(
                primary_emotion="neutral",
                confidence=0.5,
                emotion_vector={"neutral": 1.0},
                temporal_pattern={"stability": 1.0, "trend": "stable"},
                processing_time=time.time() - start_time
            )
    
    def get_user_emotion_summary(self, user_id: str) -> Dict:
        """获取用户情感摘要"""
        context = self.emotion_history.get_context(user_id)
        
        return {
            "dominant_emotion": context.get("dominant_emotion", "neutral"),
            "emotional_stability": context.get("stability", 1.0),
            "recent_pattern": context.get("recent_emotions", [])[-5:],
            "transition_patterns": context.get("transition_patterns", {}),
            "total_interactions": len(self.emotion_history.user_histories[user_id])
        }

# 导入生产级情感识别系统
try:
    from .production_emotion_model import predict_emotion_production, production_emotion_model
    USE_PRODUCTION_MODEL = True
except ImportError:
    USE_PRODUCTION_MODEL = False

# 全局实例
advanced_emotion_recognition = AdvancedEmotionRecognition()

async def analyze_emotion(
    audio: bytes = None, 
    text: str = None, 
    user_id: str = "default"
) -> EmotionAnalysisResult:
    """情感分析便捷函数 - 优先使用生产级模型"""
    if USE_PRODUCTION_MODEL:
        # 使用生产级情感识别系统
        result = await predict_emotion_production(
            audio_data=audio,
            text=text,
            context={"user_id": user_id}
        )
        
        # 转换为原有格式
        return EmotionAnalysisResult(
            primary_emotion=result.emotion,
            confidence=result.confidence,
            emotion_vector=result.probabilities,
            temporal_pattern={"stability": 1.0, "trend": "stable"},
            processing_time=result.processing_time
        )
    else:
        # 使用原有系统
        if audio and text:
            return await advanced_emotion_recognition.analyze_multimodal_emotion(audio, text, user_id)
        elif text:
            return await advanced_emotion_recognition.analyze_text_only(text, user_id)
        else:
            raise ValueError("至少需要提供音频或文本数据")

if __name__ == "__main__":
    # 测试代码
    async def test_emotion_recognition():
        # 测试文本情感分析
        text_result = await analyze_emotion(
            text="我今天非常开心，工作进展很顺利！",
            user_id="test_user"
        )
        
        print(f"文本情感分析结果:")
        print(f"  主要情感: {text_result.primary_emotion}")
        print(f"  置信度: {text_result.confidence:.2f}")
        print(f"  情感向量: {text_result.emotion_vector}")
        print(f"  处理时间: {text_result.processing_time*1000:.2f}ms")
        
        # 测试多模态情感分析
        test_audio = b"test_audio_data" * 100
        multimodal_result = await analyze_emotion(
            audio=test_audio,
            text="我有点担心这个项目能不能按时完成",
            user_id="test_user"
        )
        
        print(f"\n多模态情感分析结果:")
        print(f"  主要情感: {multimodal_result.primary_emotion}")
        print(f"  置信度: {multimodal_result.confidence:.2f}")
        print(f"  情感向量: {multimodal_result.emotion_vector}")
        print(f"  时间模式: {multimodal_result.temporal_pattern}")
        
        # 获取用户情感摘要
        summary = advanced_emotion_recognition.get_user_emotion_summary("test_user")
        print(f"\n用户情感摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # 运行测试
    asyncio.run(test_emotion_recognition())
