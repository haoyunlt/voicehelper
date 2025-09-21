"""
用户画像系统 - v1.8.0
构建深度个性化用户体验的用户画像和智能推荐系统
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """交互类型"""
    CHAT = "chat"
    VOICE = "voice"
    IMAGE = "image"
    DOCUMENT = "document"
    SEARCH = "search"
    FEEDBACK = "feedback"

class UserPreferenceType(Enum):
    """用户偏好类型"""
    COMMUNICATION_STYLE = "communication_style"  # 沟通风格
    CONTENT_TYPE = "content_type"               # 内容类型
    INTERACTION_MODE = "interaction_mode"       # 交互模式
    TOPIC_INTEREST = "topic_interest"           # 话题兴趣
    RESPONSE_LENGTH = "response_length"         # 回复长度
    FORMALITY_LEVEL = "formality_level"         # 正式程度

class PersonalityTrait(Enum):
    """性格特征"""
    OPENNESS = "openness"           # 开放性
    CONSCIENTIOUSNESS = "conscientiousness"  # 尽责性
    EXTRAVERSION = "extraversion"   # 外向性
    AGREEABLENESS = "agreeableness" # 宜人性
    NEUROTICISM = "neuroticism"     # 神经质

@dataclass
class UserInteraction:
    """用户交互记录"""
    user_id: str
    interaction_type: InteractionType
    content: str
    response: str = ""
    satisfaction_score: Optional[float] = None
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserPreference:
    """用户偏好"""
    preference_type: UserPreferenceType
    value: Any
    confidence: float
    last_updated: datetime = field(default_factory=datetime.now)
    evidence_count: int = 1

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preferences: Dict[UserPreferenceType, UserPreference] = field(default_factory=dict)
    personality_traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    interaction_history: List[UserInteraction] = field(default_factory=list)
    topic_interests: Dict[str, float] = field(default_factory=dict)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

class InteractionAnalyzer:
    """交互分析器"""
    
    def __init__(self):
        # 沟通风格关键词
        self.communication_style_keywords = {
            'formal': ['您', '请问', '麻烦', '谢谢', '不好意思', '打扰'],
            'casual': ['你', '咋样', '怎么样', '好的', '行', '嗯'],
            'professional': ['业务', '工作', '项目', '方案', '分析', '报告'],
            'friendly': ['哈哈', '😊', '谢谢', '太好了', '棒', '赞']
        }
        
        # 话题分类关键词
        self.topic_keywords = {
            'technology': ['AI', '人工智能', '技术', '算法', '编程', '开发', '软件'],
            'business': ['商业', '市场', '销售', '营销', '管理', '策略', '投资'],
            'education': ['学习', '教育', '课程', '知识', '培训', '考试', '学校'],
            'entertainment': ['电影', '音乐', '游戏', '娱乐', '明星', '综艺', '小说'],
            'health': ['健康', '医疗', '运动', '饮食', '养生', '锻炼', '疾病'],
            'travel': ['旅游', '旅行', '景点', '酒店', '机票', '攻略', '度假'],
            'lifestyle': ['生活', '家居', '美食', '购物', '时尚', '美容', '宠物']
        }
        
        # 情感词典
        self.sentiment_keywords = {
            'positive': ['好', '棒', '优秀', '满意', '喜欢', '赞', '完美', '太好了'],
            'negative': ['不好', '差', '糟糕', '不满意', '讨厌', '失望', '问题'],
            'neutral': ['一般', '还行', '普通', '正常', '可以', '了解']
        }
    
    async def analyze_interaction(self, interaction: UserInteraction) -> Dict[str, Any]:
        """
        分析单次交互
        
        Args:
            interaction: 用户交互记录
            
        Returns:
            Dict: 分析结果
        """
        try:
            content = interaction.content.lower()
            
            analysis = {
                'communication_style': self._analyze_communication_style(content),
                'topics': self._analyze_topics(content),
                'sentiment': self._analyze_sentiment(content),
                'complexity': self._analyze_complexity(content),
                'length_preference': self._analyze_length_preference(interaction),
                'interaction_patterns': self._analyze_interaction_patterns(interaction)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Interaction analysis error: {e}")
            return {}
    
    def _analyze_communication_style(self, content: str) -> Dict[str, float]:
        """分析沟通风格"""
        style_scores = {}
        
        for style, keywords in self.communication_style_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            style_scores[style] = score / len(keywords)
        
        return style_scores
    
    def _analyze_topics(self, content: str) -> Dict[str, float]:
        """分析话题兴趣"""
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            topic_scores[topic] = score / len(keywords)
        
        return topic_scores
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """分析情感倾向"""
        sentiment_scores = {}
        
        for sentiment, keywords in self.sentiment_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            sentiment_scores[sentiment] = score / len(keywords)
        
        return sentiment_scores
    
    def _analyze_complexity(self, content: str) -> float:
        """分析内容复杂度"""
        # 基于字符数、词汇数、标点符号等计算复杂度
        char_count = len(content)
        word_count = len(content.split())
        punctuation_count = sum(1 for char in content if char in '，。！？；：')
        
        # 简单的复杂度计算
        complexity = (char_count / 100 + word_count / 50 + punctuation_count / 10) / 3
        return min(complexity, 1.0)
    
    def _analyze_length_preference(self, interaction: UserInteraction) -> str:
        """分析回复长度偏好"""
        content_length = len(interaction.content)
        
        if content_length < 20:
            return 'short'
        elif content_length < 100:
            return 'medium'
        else:
            return 'long'
    
    def _analyze_interaction_patterns(self, interaction: UserInteraction) -> Dict[str, Any]:
        """分析交互模式"""
        return {
            'interaction_type': interaction.interaction_type.value,
            'duration': interaction.duration,
            'hour_of_day': interaction.timestamp.hour,
            'day_of_week': interaction.timestamp.weekday(),
            'has_feedback': interaction.satisfaction_score is not None
        }

class PersonalityAnalyzer:
    """性格分析器"""
    
    def __init__(self):
        # Big Five性格模型的关键词映射
        self.personality_keywords = {
            PersonalityTrait.OPENNESS: {
                'high': ['创新', '创意', '想象', '好奇', '探索', '新颖', '艺术'],
                'low': ['传统', '保守', '常规', '实用', '现实']
            },
            PersonalityTrait.CONSCIENTIOUSNESS: {
                'high': ['计划', '组织', '准时', '细心', '负责', '完成', '目标'],
                'low': ['随意', '灵活', '自由', '即兴']
            },
            PersonalityTrait.EXTRAVERSION: {
                'high': ['社交', '活跃', '外向', '聚会', '交流', '分享', '热情'],
                'low': ['安静', '独处', '内向', '思考', '独立']
            },
            PersonalityTrait.AGREEABLENESS: {
                'high': ['友善', '合作', '帮助', '理解', '同情', '和谐', '信任'],
                'low': ['竞争', '批评', '怀疑', '直接']
            },
            PersonalityTrait.NEUROTICISM: {
                'high': ['担心', '焦虑', '紧张', '压力', '情绪', '敏感'],
                'low': ['冷静', '稳定', '放松', '乐观', '平静']
            }
        }
    
    async def analyze_personality(self, interactions: List[UserInteraction]) -> Dict[PersonalityTrait, float]:
        """
        分析用户性格特征
        
        Args:
            interactions: 用户交互历史
            
        Returns:
            Dict[PersonalityTrait, float]: 性格特征得分
        """
        try:
            personality_scores = {}
            
            # 合并所有交互内容
            all_content = ' '.join([interaction.content for interaction in interactions])
            content_lower = all_content.lower()
            
            for trait, keywords in self.personality_keywords.items():
                high_score = sum(1 for keyword in keywords['high'] if keyword in content_lower)
                low_score = sum(1 for keyword in keywords['low'] if keyword in content_lower)
                
                # 计算特征得分 (0-1)
                total_keywords = len(keywords['high']) + len(keywords['low'])
                if total_keywords > 0:
                    score = (high_score - low_score + total_keywords) / (2 * total_keywords)
                    personality_scores[trait] = max(0.0, min(1.0, score))
                else:
                    personality_scores[trait] = 0.5  # 默认中性
            
            # 基于交互模式调整性格得分
            personality_scores = await self._adjust_personality_by_behavior(
                personality_scores, interactions
            )
            
            return personality_scores
            
        except Exception as e:
            logger.error(f"Personality analysis error: {e}")
            return {trait: 0.5 for trait in PersonalityTrait}
    
    async def _adjust_personality_by_behavior(self, 
                                           personality_scores: Dict[PersonalityTrait, float],
                                           interactions: List[UserInteraction]) -> Dict[PersonalityTrait, float]:
        """基于行为模式调整性格得分"""
        try:
            if not interactions:
                return personality_scores
            
            # 分析交互频率（外向性指标）
            interaction_frequency = len(interactions) / max(1, (datetime.now() - interactions[0].timestamp).days)
            if interaction_frequency > 5:  # 高频交互
                personality_scores[PersonalityTrait.EXTRAVERSION] = min(1.0, 
                    personality_scores[PersonalityTrait.EXTRAVERSION] + 0.1)
            
            # 分析回复长度（开放性指标）
            avg_length = np.mean([len(interaction.content) for interaction in interactions])
            if avg_length > 100:  # 长回复倾向于高开放性
                personality_scores[PersonalityTrait.OPENNESS] = min(1.0,
                    personality_scores[PersonalityTrait.OPENNESS] + 0.1)
            
            # 分析满意度反馈（宜人性指标）
            feedback_interactions = [i for i in interactions if i.satisfaction_score is not None]
            if feedback_interactions:
                avg_satisfaction = np.mean([i.satisfaction_score for i in feedback_interactions])
                if avg_satisfaction > 0.8:  # 高满意度用户通常更宜人
                    personality_scores[PersonalityTrait.AGREEABLENESS] = min(1.0,
                        personality_scores[PersonalityTrait.AGREEABLENESS] + 0.1)
            
            return personality_scores
            
        except Exception as e:
            logger.error(f"Personality adjustment error: {e}")
            return personality_scores

class PreferenceInferrer:
    """偏好推断器"""
    
    def __init__(self):
        self.preference_thresholds = {
            UserPreferenceType.COMMUNICATION_STYLE: 0.3,
            UserPreferenceType.CONTENT_TYPE: 0.2,
            UserPreferenceType.INTERACTION_MODE: 0.4,
            UserPreferenceType.TOPIC_INTEREST: 0.25,
            UserPreferenceType.RESPONSE_LENGTH: 0.35,
            UserPreferenceType.FORMALITY_LEVEL: 0.3
        }
    
    async def infer_preferences(self, 
                              interactions: List[UserInteraction],
                              analysis_results: List[Dict[str, Any]]) -> Dict[UserPreferenceType, UserPreference]:
        """
        推断用户偏好
        
        Args:
            interactions: 用户交互历史
            analysis_results: 交互分析结果
            
        Returns:
            Dict[UserPreferenceType, UserPreference]: 用户偏好
        """
        try:
            preferences = {}
            
            if not analysis_results:
                return preferences
            
            # 推断沟通风格偏好
            comm_style_pref = await self._infer_communication_style(analysis_results)
            if comm_style_pref:
                preferences[UserPreferenceType.COMMUNICATION_STYLE] = comm_style_pref
            
            # 推断内容类型偏好
            content_type_pref = await self._infer_content_type(interactions, analysis_results)
            if content_type_pref:
                preferences[UserPreferenceType.CONTENT_TYPE] = content_type_pref
            
            # 推断交互模式偏好
            interaction_mode_pref = await self._infer_interaction_mode(interactions)
            if interaction_mode_pref:
                preferences[UserPreferenceType.INTERACTION_MODE] = interaction_mode_pref
            
            # 推断话题兴趣偏好
            topic_interest_pref = await self._infer_topic_interests(analysis_results)
            if topic_interest_pref:
                preferences[UserPreferenceType.TOPIC_INTEREST] = topic_interest_pref
            
            # 推断回复长度偏好
            length_pref = await self._infer_response_length(analysis_results)
            if length_pref:
                preferences[UserPreferenceType.RESPONSE_LENGTH] = length_pref
            
            # 推断正式程度偏好
            formality_pref = await self._infer_formality_level(analysis_results)
            if formality_pref:
                preferences[UserPreferenceType.FORMALITY_LEVEL] = formality_pref
            
            return preferences
            
        except Exception as e:
            logger.error(f"Preference inference error: {e}")
            return {}
    
    async def _infer_communication_style(self, analysis_results: List[Dict[str, Any]]) -> Optional[UserPreference]:
        """推断沟通风格偏好"""
        try:
            style_scores = defaultdict(float)
            
            for result in analysis_results:
                comm_styles = result.get('communication_style', {})
                for style, score in comm_styles.items():
                    style_scores[style] += score
            
            if style_scores:
                # 找到最高得分的风格
                best_style = max(style_scores.items(), key=lambda x: x[1])
                
                if best_style[1] > self.preference_thresholds[UserPreferenceType.COMMUNICATION_STYLE]:
                    confidence = min(best_style[1] / len(analysis_results), 1.0)
                    
                    return UserPreference(
                        preference_type=UserPreferenceType.COMMUNICATION_STYLE,
                        value=best_style[0],
                        confidence=confidence,
                        evidence_count=len(analysis_results)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Communication style inference error: {e}")
            return None
    
    async def _infer_content_type(self, 
                                interactions: List[UserInteraction],
                                analysis_results: List[Dict[str, Any]]) -> Optional[UserPreference]:
        """推断内容类型偏好"""
        try:
            content_types = defaultdict(int)
            
            for interaction in interactions:
                content_types[interaction.interaction_type.value] += 1
            
            if content_types:
                most_used_type = max(content_types.items(), key=lambda x: x[1])
                
                if most_used_type[1] / len(interactions) > self.preference_thresholds[UserPreferenceType.CONTENT_TYPE]:
                    confidence = most_used_type[1] / len(interactions)
                    
                    return UserPreference(
                        preference_type=UserPreferenceType.CONTENT_TYPE,
                        value=most_used_type[0],
                        confidence=confidence,
                        evidence_count=most_used_type[1]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Content type inference error: {e}")
            return None
    
    async def _infer_interaction_mode(self, interactions: List[UserInteraction]) -> Optional[UserPreference]:
        """推断交互模式偏好"""
        try:
            # 分析交互时间模式
            hours = [interaction.timestamp.hour for interaction in interactions]
            
            if hours:
                # 计算最常用的时间段
                hour_counts = defaultdict(int)
                for hour in hours:
                    if 6 <= hour < 12:
                        hour_counts['morning'] += 1
                    elif 12 <= hour < 18:
                        hour_counts['afternoon'] += 1
                    elif 18 <= hour < 24:
                        hour_counts['evening'] += 1
                    else:
                        hour_counts['night'] += 1
                
                if hour_counts:
                    preferred_time = max(hour_counts.items(), key=lambda x: x[1])
                    
                    if preferred_time[1] / len(interactions) > self.preference_thresholds[UserPreferenceType.INTERACTION_MODE]:
                        confidence = preferred_time[1] / len(interactions)
                        
                        return UserPreference(
                            preference_type=UserPreferenceType.INTERACTION_MODE,
                            value=f"prefer_{preferred_time[0]}",
                            confidence=confidence,
                            evidence_count=preferred_time[1]
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Interaction mode inference error: {e}")
            return None
    
    async def _infer_topic_interests(self, analysis_results: List[Dict[str, Any]]) -> Optional[UserPreference]:
        """推断话题兴趣偏好"""
        try:
            topic_scores = defaultdict(float)
            
            for result in analysis_results:
                topics = result.get('topics', {})
                for topic, score in topics.items():
                    topic_scores[topic] += score
            
            if topic_scores:
                # 找到最感兴趣的话题
                top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                
                if top_topics and top_topics[0][1] > self.preference_thresholds[UserPreferenceType.TOPIC_INTEREST]:
                    confidence = min(top_topics[0][1] / len(analysis_results), 1.0)
                    
                    return UserPreference(
                        preference_type=UserPreferenceType.TOPIC_INTEREST,
                        value=[topic for topic, _ in top_topics],
                        confidence=confidence,
                        evidence_count=len(analysis_results)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Topic interest inference error: {e}")
            return None
    
    async def _infer_response_length(self, analysis_results: List[Dict[str, Any]]) -> Optional[UserPreference]:
        """推断回复长度偏好"""
        try:
            length_preferences = defaultdict(int)
            
            for result in analysis_results:
                length_pref = result.get('length_preference')
                if length_pref:
                    length_preferences[length_pref] += 1
            
            if length_preferences:
                preferred_length = max(length_preferences.items(), key=lambda x: x[1])
                
                if preferred_length[1] / len(analysis_results) > self.preference_thresholds[UserPreferenceType.RESPONSE_LENGTH]:
                    confidence = preferred_length[1] / len(analysis_results)
                    
                    return UserPreference(
                        preference_type=UserPreferenceType.RESPONSE_LENGTH,
                        value=preferred_length[0],
                        confidence=confidence,
                        evidence_count=preferred_length[1]
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Response length inference error: {e}")
            return None
    
    async def _infer_formality_level(self, analysis_results: List[Dict[str, Any]]) -> Optional[UserPreference]:
        """推断正式程度偏好"""
        try:
            formal_score = 0
            casual_score = 0
            
            for result in analysis_results:
                comm_styles = result.get('communication_style', {})
                formal_score += comm_styles.get('formal', 0)
                casual_score += comm_styles.get('casual', 0)
            
            total_score = formal_score + casual_score
            if total_score > 0:
                formality_ratio = formal_score / total_score
                
                if abs(formality_ratio - 0.5) > self.preference_thresholds[UserPreferenceType.FORMALITY_LEVEL]:
                    if formality_ratio > 0.5:
                        value = 'formal'
                        confidence = formality_ratio
                    else:
                        value = 'casual'
                        confidence = 1 - formality_ratio
                    
                    return UserPreference(
                        preference_type=UserPreferenceType.FORMALITY_LEVEL,
                        value=value,
                        confidence=confidence,
                        evidence_count=len(analysis_results)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Formality level inference error: {e}")
            return None

class UserProfileManager:
    """用户画像管理器"""
    
    def __init__(self):
        self.interaction_analyzer = InteractionAnalyzer()
        self.personality_analyzer = PersonalityAnalyzer()
        self.preference_inferrer = PreferenceInferrer()
        
        # 用户画像存储
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # 更新阈值
        self.update_thresholds = {
            'min_interactions': 5,      # 最少交互次数
            'confidence_threshold': 0.6, # 置信度阈值
            'update_interval_hours': 24  # 更新间隔（小时）
        }
    
    async def add_interaction(self, interaction: UserInteraction):
        """添加用户交互记录"""
        try:
            user_id = interaction.user_id
            
            # 获取或创建用户画像
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            
            # 添加交互记录
            profile.interaction_history.append(interaction)
            
            # 限制历史记录长度
            if len(profile.interaction_history) > 1000:
                profile.interaction_history = profile.interaction_history[-500:]
            
            # 检查是否需要更新画像
            should_update = await self._should_update_profile(profile)
            if should_update:
                await self.update_user_profile(user_id)
            
        except Exception as e:
            logger.error(f"Add interaction error: {e}")
    
    async def update_user_profile(self, user_id: str):
        """更新用户画像"""
        try:
            if user_id not in self.user_profiles:
                logger.warning(f"User profile not found: {user_id}")
                return
            
            profile = self.user_profiles[user_id]
            
            # 分析最近的交互
            recent_interactions = profile.interaction_history[-100:]  # 最近100次交互
            
            if len(recent_interactions) < self.update_thresholds['min_interactions']:
                return
            
            # 分析交互
            analysis_results = []
            for interaction in recent_interactions:
                analysis = await self.interaction_analyzer.analyze_interaction(interaction)
                analysis_results.append(analysis)
            
            # 分析性格特征
            personality_traits = await self.personality_analyzer.analyze_personality(recent_interactions)
            profile.personality_traits = personality_traits
            
            # 推断偏好
            preferences = await self.preference_inferrer.infer_preferences(
                recent_interactions, analysis_results
            )
            
            # 更新偏好（只更新高置信度的偏好）
            for pref_type, preference in preferences.items():
                if preference.confidence >= self.update_thresholds['confidence_threshold']:
                    profile.preferences[pref_type] = preference
            
            # 更新话题兴趣
            await self._update_topic_interests(profile, analysis_results)
            
            # 更新行为模式
            await self._update_behavioral_patterns(profile, recent_interactions)
            
            profile.last_updated = datetime.now()
            
            logger.info(f"Updated user profile for {user_id}")
            
        except Exception as e:
            logger.error(f"Update user profile error: {e}")
    
    async def _should_update_profile(self, profile: UserProfile) -> bool:
        """判断是否应该更新画像"""
        try:
            # 检查交互数量
            if len(profile.interaction_history) < self.update_thresholds['min_interactions']:
                return False
            
            # 检查更新间隔
            time_since_update = datetime.now() - profile.last_updated
            if time_since_update.total_seconds() < self.update_thresholds['update_interval_hours'] * 3600:
                return False
            
            # 检查是否有新的交互
            recent_interactions = [
                i for i in profile.interaction_history
                if i.timestamp > profile.last_updated
            ]
            
            return len(recent_interactions) >= 3  # 至少3次新交互
            
        except Exception as e:
            logger.error(f"Should update profile check error: {e}")
            return False
    
    async def _update_topic_interests(self, profile: UserProfile, analysis_results: List[Dict[str, Any]]):
        """更新话题兴趣"""
        try:
            topic_scores = defaultdict(float)
            
            for result in analysis_results:
                topics = result.get('topics', {})
                for topic, score in topics.items():
                    topic_scores[topic] += score
            
            # 归一化并更新
            if topic_scores:
                max_score = max(topic_scores.values())
                if max_score > 0:
                    for topic, score in topic_scores.items():
                        normalized_score = score / max_score
                        
                        # 指数移动平均更新
                        if topic in profile.topic_interests:
                            profile.topic_interests[topic] = (
                                profile.topic_interests[topic] * 0.7 + normalized_score * 0.3
                            )
                        else:
                            profile.topic_interests[topic] = normalized_score
            
        except Exception as e:
            logger.error(f"Update topic interests error: {e}")
    
    async def _update_behavioral_patterns(self, profile: UserProfile, interactions: List[UserInteraction]):
        """更新行为模式"""
        try:
            patterns = {}
            
            # 交互频率模式
            if len(interactions) > 1:
                time_diffs = []
                for i in range(1, len(interactions)):
                    diff = (interactions[i].timestamp - interactions[i-1].timestamp).total_seconds()
                    time_diffs.append(diff)
                
                if time_diffs:
                    patterns['avg_interaction_interval'] = np.mean(time_diffs)
                    patterns['interaction_regularity'] = 1.0 / (np.std(time_diffs) + 1)
            
            # 时间偏好模式
            hours = [interaction.timestamp.hour for interaction in interactions]
            if hours:
                patterns['preferred_hours'] = list(set(hours))
                patterns['most_active_hour'] = max(set(hours), key=hours.count)
            
            # 交互类型分布
            type_counts = defaultdict(int)
            for interaction in interactions:
                type_counts[interaction.interaction_type.value] += 1
            
            patterns['interaction_type_distribution'] = dict(type_counts)
            
            # 满意度模式
            satisfaction_scores = [
                i.satisfaction_score for i in interactions 
                if i.satisfaction_score is not None
            ]
            if satisfaction_scores:
                patterns['avg_satisfaction'] = np.mean(satisfaction_scores)
                patterns['satisfaction_trend'] = self._calculate_trend(satisfaction_scores)
            
            profile.behavioral_patterns = patterns
            
        except Exception as e:
            logger.error(f"Update behavioral patterns error: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 简单的线性趋势计算
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        return self.user_profiles.get(user_id)
    
    def get_user_recommendations(self, user_id: str) -> Dict[str, Any]:
        """获取用户个性化推荐"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return {}
            
            recommendations = {
                'communication_style': self._recommend_communication_style(profile),
                'content_suggestions': self._recommend_content(profile),
                'interaction_timing': self._recommend_timing(profile),
                'response_format': self._recommend_response_format(profile)
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Get user recommendations error: {e}")
            return {}
    
    def _recommend_communication_style(self, profile: UserProfile) -> Dict[str, Any]:
        """推荐沟通风格"""
        style_pref = profile.preferences.get(UserPreferenceType.COMMUNICATION_STYLE)
        formality_pref = profile.preferences.get(UserPreferenceType.FORMALITY_LEVEL)
        
        recommendations = {
            'tone': 'neutral',
            'formality': 'medium',
            'enthusiasm': 'medium'
        }
        
        if style_pref:
            if style_pref.value == 'friendly':
                recommendations['tone'] = 'warm'
                recommendations['enthusiasm'] = 'high'
            elif style_pref.value == 'professional':
                recommendations['tone'] = 'professional'
                recommendations['formality'] = 'high'
        
        if formality_pref:
            recommendations['formality'] = formality_pref.value
        
        return recommendations
    
    def _recommend_content(self, profile: UserProfile) -> List[str]:
        """推荐内容类型"""
        topic_interests = profile.topic_interests
        
        if not topic_interests:
            return ['general']
        
        # 按兴趣度排序
        sorted_topics = sorted(topic_interests.items(), key=lambda x: x[1], reverse=True)
        
        return [topic for topic, score in sorted_topics[:5] if score > 0.3]
    
    def _recommend_timing(self, profile: UserProfile) -> Dict[str, Any]:
        """推荐交互时机"""
        patterns = profile.behavioral_patterns
        
        recommendations = {
            'best_hours': patterns.get('preferred_hours', [9, 14, 20]),
            'avoid_hours': [0, 1, 2, 3, 4, 5],  # 深夜时段
            'response_urgency': 'medium'
        }
        
        # 基于交互频率调整紧急度
        avg_interval = patterns.get('avg_interaction_interval', 3600)
        if avg_interval < 300:  # 5分钟内
            recommendations['response_urgency'] = 'high'
        elif avg_interval > 7200:  # 2小时以上
            recommendations['response_urgency'] = 'low'
        
        return recommendations
    
    def _recommend_response_format(self, profile: UserProfile) -> Dict[str, Any]:
        """推荐回复格式"""
        length_pref = profile.preferences.get(UserPreferenceType.RESPONSE_LENGTH)
        
        recommendations = {
            'length': 'medium',
            'structure': 'paragraph',
            'use_examples': True,
            'use_bullet_points': False
        }
        
        if length_pref:
            recommendations['length'] = length_pref.value
            
            if length_pref.value == 'short':
                recommendations['structure'] = 'concise'
                recommendations['use_examples'] = False
            elif length_pref.value == 'long':
                recommendations['use_bullet_points'] = True
                recommendations['structure'] = 'detailed'
        
        return recommendations
    
    def get_profile_analytics(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像分析"""
        try:
            profile = self.get_user_profile(user_id)
            if not profile:
                return {}
            
            analytics = {
                'profile_completeness': self._calculate_profile_completeness(profile),
                'interaction_summary': self._get_interaction_summary(profile),
                'preference_confidence': self._get_preference_confidence(profile),
                'personality_summary': self._get_personality_summary(profile),
                'engagement_level': self._calculate_engagement_level(profile)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Get profile analytics error: {e}")
            return {}
    
    def _calculate_profile_completeness(self, profile: UserProfile) -> float:
        """计算画像完整度"""
        total_aspects = len(UserPreferenceType) + len(PersonalityTrait) + 2  # +2 for topics and patterns
        
        completed_aspects = 0
        completed_aspects += len(profile.preferences)
        completed_aspects += len(profile.personality_traits)
        
        if profile.topic_interests:
            completed_aspects += 1
        if profile.behavioral_patterns:
            completed_aspects += 1
        
        return completed_aspects / total_aspects
    
    def _get_interaction_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """获取交互摘要"""
        interactions = profile.interaction_history
        
        if not interactions:
            return {}
        
        return {
            'total_interactions': len(interactions),
            'interaction_types': list(set(i.interaction_type.value for i in interactions)),
            'avg_satisfaction': np.mean([i.satisfaction_score for i in interactions if i.satisfaction_score is not None]) if any(i.satisfaction_score is not None for i in interactions) else None,
            'first_interaction': interactions[0].timestamp.isoformat(),
            'last_interaction': interactions[-1].timestamp.isoformat()
        }
    
    def _get_preference_confidence(self, profile: UserProfile) -> Dict[str, float]:
        """获取偏好置信度"""
        return {
            pref_type.value: preference.confidence
            for pref_type, preference in profile.preferences.items()
        }
    
    def _get_personality_summary(self, profile: UserProfile) -> Dict[str, str]:
        """获取性格摘要"""
        personality_summary = {}
        
        for trait, score in profile.personality_traits.items():
            if score > 0.7:
                level = 'high'
            elif score < 0.3:
                level = 'low'
            else:
                level = 'medium'
            
            personality_summary[trait.value] = level
        
        return personality_summary
    
    def _calculate_engagement_level(self, profile: UserProfile) -> str:
        """计算用户参与度"""
        interactions = profile.interaction_history
        
        if not interactions:
            return 'unknown'
        
        # 基于交互频率和满意度计算参与度
        recent_interactions = [
            i for i in interactions
            if (datetime.now() - i.timestamp).days <= 7
        ]
        
        weekly_frequency = len(recent_interactions)
        
        if weekly_frequency >= 20:
            return 'very_high'
        elif weekly_frequency >= 10:
            return 'high'
        elif weekly_frequency >= 5:
            return 'medium'
        elif weekly_frequency >= 1:
            return 'low'
        else:
            return 'very_low'

# 使用示例
async def main():
    """示例用法"""
    profile_manager = UserProfileManager()
    
    # 模拟用户交互
    user_id = "user_123"
    
    interactions = [
        UserInteraction(
            user_id=user_id,
            interaction_type=InteractionType.CHAT,
            content="你好，请问AI技术的发展前景如何？",
            response="AI技术发展前景非常广阔...",
            satisfaction_score=0.9,
            duration=120.0
        ),
        UserInteraction(
            user_id=user_id,
            interaction_type=InteractionType.VOICE,
            content="我想了解机器学习的基础知识",
            response="机器学习是人工智能的一个分支...",
            satisfaction_score=0.8,
            duration=180.0
        ),
        UserInteraction(
            user_id=user_id,
            interaction_type=InteractionType.CHAT,
            content="谢谢您的详细解答，非常有帮助！",
            response="很高兴能帮到您！",
            satisfaction_score=0.95,
            duration=30.0
        )
    ]
    
    # 添加交互记录
    for interaction in interactions:
        await profile_manager.add_interaction(interaction)
    
    # 强制更新用户画像
    await profile_manager.update_user_profile(user_id)
    
    # 获取用户画像
    profile = profile_manager.get_user_profile(user_id)
    if profile:
        print("=== 用户画像 ===")
        print(f"用户ID: {profile.user_id}")
        print(f"交互次数: {len(profile.interaction_history)}")
        
        print("\n偏好设置:")
        for pref_type, preference in profile.preferences.items():
            print(f"  {pref_type.value}: {preference.value} (置信度: {preference.confidence:.2f})")
        
        print("\n性格特征:")
        for trait, score in profile.personality_traits.items():
            print(f"  {trait.value}: {score:.2f}")
        
        print("\n话题兴趣:")
        for topic, score in list(profile.topic_interests.items())[:5]:
            print(f"  {topic}: {score:.2f}")
    
    # 获取个性化推荐
    recommendations = profile_manager.get_user_recommendations(user_id)
    print("\n=== 个性化推荐 ===")
    for category, rec in recommendations.items():
        print(f"{category}: {rec}")
    
    # 获取分析统计
    analytics = profile_manager.get_profile_analytics(user_id)
    print(f"\n=== 画像分析 ===")
    print(f"画像完整度: {analytics.get('profile_completeness', 0):.2f}")
    print(f"参与度: {analytics.get('engagement_level', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(main())
