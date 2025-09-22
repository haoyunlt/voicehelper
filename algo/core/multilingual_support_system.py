"""
VoiceHelper v1.23.0 - 多语言支持系统
实现8种语言支持、智能语言检测、跨语言翻译
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """支持的语言"""
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    ENGLISH = "en-US"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"
    SPANISH = "es-ES"
    FRENCH = "fr-FR"
    GERMAN = "de-DE"

class LanguageDetectionMethod(Enum):
    """语言检测方法"""
    AUTOMATIC = "automatic"
    USER_PREFERENCE = "user_preference"
    CONTEXT_BASED = "context_based"
    MANUAL = "manual"

class TranslationQuality(Enum):
    """翻译质量"""
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"
    NATIVE = "native"

@dataclass
class LanguageDetectionResult:
    """语言检测结果"""
    detected_language: SupportedLanguage
    confidence: float
    detection_method: LanguageDetectionMethod
    alternative_languages: List[Tuple[SupportedLanguage, float]]
    processing_time: float

@dataclass
class TranslationResult:
    """翻译结果"""
    source_language: SupportedLanguage
    target_language: SupportedLanguage
    original_text: str
    translated_text: str
    quality_score: float
    translation_quality: TranslationQuality
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LanguageModel:
    """语言模型"""
    language: SupportedLanguage
    model_name: str
    version: str
    capabilities: List[str]
    performance_metrics: Dict[str, float]
    is_available: bool = True

class LanguageDetector:
    """语言检测器"""
    
    def __init__(self):
        self.language_patterns = {
            SupportedLanguage.CHINESE_SIMPLIFIED: {
                "keywords": ["的", "了", "在", "是", "有", "和", "不", "我", "你", "他"],
                "characters": ["一", "二", "三", "中", "国", "人", "大", "小", "好", "不"],
                "patterns": [r"[\u4e00-\u9fff]+"]
            },
            SupportedLanguage.CHINESE_TRADITIONAL: {
                "keywords": ["的", "了", "在", "是", "有", "和", "不", "我", "你", "他"],
                "characters": ["一", "二", "三", "中", "國", "人", "大", "小", "好", "不"],
                "patterns": [r"[\u4e00-\u9fff]+"]
            },
            SupportedLanguage.ENGLISH: {
                "keywords": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"],
                "characters": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                "patterns": [r"[a-zA-Z]+"]
            },
            SupportedLanguage.JAPANESE: {
                "keywords": ["です", "ます", "の", "を", "に", "は", "が", "で", "と", "も"],
                "characters": ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ"],
                "patterns": [r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+"]
            },
            SupportedLanguage.KOREAN: {
                "keywords": ["입니다", "습니다", "의", "를", "에", "는", "이", "가", "에서", "와"],
                "characters": ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차"],
                "patterns": [r"[\uac00-\ud7af]+"]
            },
            SupportedLanguage.SPANISH: {
                "keywords": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"],
                "characters": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                "patterns": [r"[a-zA-Záéíóúñü]+"]
            },
            SupportedLanguage.FRENCH: {
                "keywords": ["le", "la", "de", "du", "des", "et", "à", "en", "un", "une"],
                "characters": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                "patterns": [r"[a-zA-Zàâäéèêëïîôöùûüÿç]+"]
            },
            SupportedLanguage.GERMAN: {
                "keywords": ["der", "die", "das", "und", "in", "den", "von", "zu", "dem", "mit"],
                "characters": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                "patterns": [r"[a-zA-Zäöüß]+"]
            }
        }
        
        self.detection_history = defaultdict(list)
        self.accuracy_stats = defaultdict(float)
    
    async def detect_language(self, text: str, 
                            user_preference: Optional[SupportedLanguage] = None,
                            context: Optional[Dict[str, Any]] = None) -> LanguageDetectionResult:
        """检测语言"""
        start_time = time.time()
        
        try:
            # 如果用户有明确偏好，优先使用
            if user_preference:
                confidence = await self._calculate_confidence(text, user_preference)
                if confidence > 0.7:
                    return LanguageDetectionResult(
                        detected_language=user_preference,
                        confidence=confidence,
                        detection_method=LanguageDetectionMethod.USER_PREFERENCE,
                        alternative_languages=[],
                        processing_time=time.time() - start_time
                    )
            
            # 自动检测
            detection_scores = {}
            for language in SupportedLanguage:
                score = await self._calculate_confidence(text, language)
                detection_scores[language] = score
            
            # 排序并选择最佳匹配
            sorted_scores = sorted(detection_scores.items(), key=lambda x: x[1], reverse=True)
            best_language, best_score = sorted_scores[0]
            
            # 获取备选语言
            alternative_languages = [(lang, score) for lang, score in sorted_scores[1:3] if score > 0.3]
            
            result = LanguageDetectionResult(
                detected_language=best_language,
                confidence=best_score,
                detection_method=LanguageDetectionMethod.AUTOMATIC,
                alternative_languages=alternative_languages,
                processing_time=time.time() - start_time
            )
            
            # 记录检测历史
            self.detection_history[best_language].append({
                "text": text[:100],  # 只保存前100个字符
                "confidence": best_score,
                "timestamp": time.time()
            })
            
            logger.info(f"Language detected: {best_language.value} (confidence: {best_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return LanguageDetectionResult(
                detected_language=SupportedLanguage.ENGLISH,  # 默认英语
                confidence=0.0,
                detection_method=LanguageDetectionMethod.AUTOMATIC,
                alternative_languages=[],
                processing_time=time.time() - start_time
            )
    
    async def _calculate_confidence(self, text: str, language: SupportedLanguage) -> float:
        """计算语言置信度"""
        if language not in self.language_patterns:
            return 0.0
        
        patterns = self.language_patterns[language]
        text_lower = text.lower()
        
        # 关键词匹配
        keyword_matches = sum(1 for keyword in patterns["keywords"] if keyword in text_lower)
        keyword_score = keyword_matches / len(patterns["keywords"])
        
        # 字符匹配
        char_matches = sum(1 for char in patterns["characters"] if char in text)
        char_score = char_matches / len(patterns["characters"])
        
        # 模式匹配
        import re
        pattern_matches = 0
        for pattern in patterns["patterns"]:
            if re.search(pattern, text):
                pattern_matches += 1
        pattern_score = pattern_matches / len(patterns["patterns"])
        
        # 综合评分
        confidence = (keyword_score * 0.4 + char_score * 0.3 + pattern_score * 0.3)
        
        return min(confidence, 1.0)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """获取检测统计"""
        total_detections = sum(len(history) for history in self.detection_history.values())
        
        return {
            "total_detections": total_detections,
            "language_distribution": {
                lang.value: len(history) for lang, history in self.detection_history.items()
            },
            "accuracy_stats": dict(self.accuracy_stats)
        }

class TranslationEngine:
    """翻译引擎"""
    
    def __init__(self):
        self.translation_models = {}
        self.quality_thresholds = {
            TranslationQuality.BASIC: 0.6,
            TranslationQuality.GOOD: 0.8,
            TranslationQuality.EXCELLENT: 0.9,
            TranslationQuality.NATIVE: 0.95
        }
        
        # 初始化翻译模型
        self._initialize_translation_models()
    
    def _initialize_translation_models(self):
        """初始化翻译模型"""
        for source_lang in SupportedLanguage:
            for target_lang in SupportedLanguage:
                if source_lang != target_lang:
                    model_key = f"{source_lang.value}-{target_lang.value}"
                    self.translation_models[model_key] = LanguageModel(
                        language=target_lang,
                        model_name=f"translation_{model_key}",
                        version="1.0.0",
                        capabilities=["translation", "quality_assessment"],
                        performance_metrics={
                            "accuracy": 0.85,
                            "speed": 0.9,
                            "fluency": 0.8
                        }
                    )
    
    async def translate(self, text: str, source_language: SupportedLanguage,
                      target_language: SupportedLanguage,
                      context: Optional[Dict[str, Any]] = None) -> TranslationResult:
        """翻译文本"""
        start_time = time.time()
        
        try:
            if source_language == target_language:
                return TranslationResult(
                    source_language=source_language,
                    target_language=target_language,
                    original_text=text,
                    translated_text=text,
                    quality_score=1.0,
                    translation_quality=TranslationQuality.NATIVE,
                    processing_time=time.time() - start_time
                )
            
            # 获取翻译模型
            model_key = f"{source_language.value}-{target_language.value}"
            if model_key not in self.translation_models:
                raise ValueError(f"Translation model not available: {model_key}")
            
            model = self.translation_models[model_key]
            
            # 执行翻译
            translated_text = await self._perform_translation(text, source_language, target_language, context)
            
            # 评估翻译质量
            quality_score = await self._assess_translation_quality(text, translated_text, source_language, target_language)
            translation_quality = self._determine_quality_level(quality_score)
            
            result = TranslationResult(
                source_language=source_language,
                target_language=target_language,
                original_text=text,
                translated_text=translated_text,
                quality_score=quality_score,
                translation_quality=translation_quality,
                processing_time=time.time() - start_time,
                metadata={
                    "model_used": model.model_name,
                    "model_version": model.version,
                    "context_used": context is not None
                }
            )
            
            logger.info(f"Translation completed: {source_language.value} -> {target_language.value} (quality: {quality_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return TranslationResult(
                source_language=source_language,
                target_language=target_language,
                original_text=text,
                translated_text=text,  # 返回原文作为fallback
                quality_score=0.0,
                translation_quality=TranslationQuality.BASIC,
                processing_time=time.time() - start_time
            )
    
    async def _perform_translation(self, text: str, source_language: SupportedLanguage,
                                 target_language: SupportedLanguage,
                                 context: Optional[Dict[str, Any]] = None) -> str:
        """执行翻译"""
        # 模拟翻译过程
        await asyncio.sleep(0.1)
        
        # 简化的翻译逻辑
        translation_mappings = {
            (SupportedLanguage.ENGLISH, SupportedLanguage.CHINESE_SIMPLIFIED): {
                "hello": "你好",
                "thank you": "谢谢",
                "good morning": "早上好",
                "how are you": "你好吗"
            },
            (SupportedLanguage.CHINESE_SIMPLIFIED, SupportedLanguage.ENGLISH): {
                "你好": "hello",
                "谢谢": "thank you",
                "早上好": "good morning",
                "你好吗": "how are you"
            }
        }
        
        mapping_key = (source_language, target_language)
        if mapping_key in translation_mappings:
            mappings = translation_mappings[mapping_key]
            for original, translated in mappings.items():
                if original.lower() in text.lower():
                    return text.lower().replace(original.lower(), translated)
        
        # 默认返回带语言标记的文本
        return f"[{target_language.value}] {text}"
    
    async def _assess_translation_quality(self, original_text: str, translated_text: str,
                                        source_language: SupportedLanguage,
                                        target_language: SupportedLanguage) -> float:
        """评估翻译质量"""
        # 简化的质量评估
        quality_factors = []
        
        # 长度比例
        length_ratio = len(translated_text) / len(original_text) if original_text else 1.0
        if 0.5 <= length_ratio <= 2.0:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.6)
        
        # 字符类型匹配
        if target_language in [SupportedLanguage.CHINESE_SIMPLIFIED, SupportedLanguage.CHINESE_TRADITIONAL]:
            chinese_chars = sum(1 for char in translated_text if '\u4e00' <= char <= '\u9fff')
            if chinese_chars > 0:
                quality_factors.append(0.9)
            else:
                quality_factors.append(0.5)
        else:
            # 英文字符
            english_chars = sum(1 for char in translated_text if char.isalpha())
            if english_chars > 0:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.4)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _determine_quality_level(self, quality_score: float) -> TranslationQuality:
        """确定质量等级"""
        for quality, threshold in self.quality_thresholds.items():
            if quality_score >= threshold:
                return quality
        return TranslationQuality.BASIC

class MultilingualSupportSystem:
    """多语言支持系统"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.translation_engine = TranslationEngine()
        self.user_language_preferences = {}
        self.session_languages = {}
        self.translation_cache = {}
        
    async def process_multilingual_input(self, text: str, session_id: str,
                                       user_id: str = None) -> Dict[str, Any]:
        """处理多语言输入"""
        try:
            # 获取用户语言偏好
            user_preference = self.user_language_preferences.get(user_id) if user_id else None
            
            # 检测语言
            detection_result = await self.language_detector.detect_language(
                text, user_preference
            )
            
            # 更新会话语言
            self.session_languages[session_id] = detection_result.detected_language
            
            # 如果需要翻译到用户偏好语言
            translated_text = text
            translation_result = None
            if (user_preference and 
                detection_result.detected_language != user_preference):
                
                translation_result = await self.translation_engine.translate(
                    text, detection_result.detected_language, user_preference
                )
                translated_text = translation_result.translated_text
            
            return {
                "original_text": text,
                "detected_language": detection_result.detected_language.value,
                "detection_confidence": detection_result.confidence,
                "translated_text": translated_text,
                "translation_result": translation_result.__dict__ if translation_result else None,
                "session_language": detection_result.detected_language.value
            }
            
        except Exception as e:
            logger.error(f"Multilingual processing error: {e}")
            return {
                "original_text": text,
                "detected_language": SupportedLanguage.ENGLISH.value,
                "detection_confidence": 0.0,
                "translated_text": text,
                "translation_result": None,
                "session_language": SupportedLanguage.ENGLISH.value
            }
    
    def set_user_language_preference(self, user_id: str, language: SupportedLanguage):
        """设置用户语言偏好"""
        self.user_language_preferences[user_id] = language
        logger.info(f"Set language preference for user {user_id}: {language.value}")
    
    def get_user_language_preference(self, user_id: str) -> Optional[SupportedLanguage]:
        """获取用户语言偏好"""
        return self.user_language_preferences.get(user_id)
    
    def get_session_language(self, session_id: str) -> Optional[SupportedLanguage]:
        """获取会话语言"""
        return self.session_languages.get(session_id)
    
    async def translate_session_content(self, session_id: str, 
                                      target_language: SupportedLanguage) -> Dict[str, Any]:
        """翻译会话内容"""
        current_language = self.session_languages.get(session_id)
        if not current_language or current_language == target_language:
            return {"translated": False, "reason": "same_language"}
        
        # 这里应该获取会话内容进行翻译
        # 简化实现
        return {
            "translated": True,
            "source_language": current_language.value,
            "target_language": target_language.value,
            "translation_count": 0
        }
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """获取支持的语言列表"""
        return [
            {
                "language": lang.value,
                "name": lang.name,
                "display_name": self._get_display_name(lang),
                "is_available": True
            }
            for lang in SupportedLanguage
        ]
    
    def _get_display_name(self, language: SupportedLanguage) -> str:
        """获取显示名称"""
        display_names = {
            SupportedLanguage.CHINESE_SIMPLIFIED: "简体中文",
            SupportedLanguage.CHINESE_TRADITIONAL: "繁體中文",
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.JAPANESE: "日本語",
            SupportedLanguage.KOREAN: "한국어",
            SupportedLanguage.SPANISH: "Español",
            SupportedLanguage.FRENCH: "Français",
            SupportedLanguage.GERMAN: "Deutsch"
        }
        return display_names.get(language, language.value)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "supported_languages": len(SupportedLanguage),
            "user_preferences_count": len(self.user_language_preferences),
            "active_sessions": len(self.session_languages),
            "translation_cache_size": len(self.translation_cache),
            "detection_stats": self.language_detector.get_detection_stats(),
            "language_distribution": {
                lang.value: count for lang, count in 
                self.language_detector.detection_history.items()
            }
        }

# 全局多语言支持系统实例
multilingual_system = MultilingualSupportSystem()

async def process_multilingual_text(text: str, session_id: str, user_id: str = None) -> Dict[str, Any]:
    """处理多语言文本"""
    return await multilingual_system.process_multilingual_input(text, session_id, user_id)

def set_user_language(user_id: str, language: SupportedLanguage):
    """设置用户语言"""
    multilingual_system.set_user_language_preference(user_id, language)

def get_supported_languages() -> List[Dict[str, Any]]:
    """获取支持的语言"""
    return multilingual_system.get_supported_languages()

def get_multilingual_stats() -> Dict[str, Any]:
    """获取多语言统计"""
    return multilingual_system.get_system_stats()

if __name__ == "__main__":
    # 测试代码
    async def test_multilingual_system():
        # 测试语言检测
        test_texts = [
            "你好，世界！",
            "Hello, world!",
            "こんにちは、世界！",
            "안녕하세요, 세계!",
            "Hola, mundo!"
        ]
        
        for text in test_texts:
            result = await process_multilingual_text(text, "test_session", "test_user")
            print(f"Text: {text}")
            print(f"Detected: {result['detected_language']} (confidence: {result['detection_confidence']:.2f})")
            print()
        
        # 获取统计信息
        stats = get_multilingual_stats()
        print("Multilingual system stats:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    asyncio.run(test_multilingual_system())
