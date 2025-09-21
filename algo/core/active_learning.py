"""
主动学习和反馈系统 - v1.4.0
实现用户反馈收集、模型优化、自适应学习
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import asyncio
from collections import defaultdict
from loguru import logger


@dataclass
class FeedbackItem:
    """反馈项"""
    feedback_id: str
    conversation_id: str
    message_id: str
    user_id: str
    tenant_id: str
    feedback_type: str  # thumbs_up, thumbs_down, correction, suggestion
    rating: Optional[float] = None  # 1-5
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LearningExample:
    """学习样例"""
    input_text: str
    expected_output: str
    actual_output: Optional[str] = None
    feedback: Optional[FeedbackItem] = None
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    is_verified: bool = False


class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend
        self.feedback_buffer: List[FeedbackItem] = []
        self.feedback_stats = defaultdict(lambda: {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "corrections": 0,
            "avg_rating": 0.0
        })
    
    async def collect_feedback(
        self,
        feedback: FeedbackItem
    ) -> bool:
        """收集用户反馈"""
        try:
            # 添加到缓冲区
            self.feedback_buffer.append(feedback)
            
            # 更新统计
            stats = self.feedback_stats[feedback.tenant_id]
            stats["total"] += 1
            
            if feedback.feedback_type == "thumbs_up":
                stats["positive"] += 1
            elif feedback.feedback_type == "thumbs_down":
                stats["negative"] += 1
            elif feedback.feedback_type == "correction":
                stats["corrections"] += 1
            
            if feedback.rating:
                # 更新平均评分
                current_avg = stats["avg_rating"]
                total = stats["total"]
                stats["avg_rating"] = (current_avg * (total - 1) + feedback.rating) / total
            
            # 持久化存储
            if self.storage:
                await self.storage.save_feedback(feedback)
            
            # 如果缓冲区满了，触发批处理
            if len(self.feedback_buffer) >= 100:
                await self._process_feedback_batch()
            
            logger.info(f"收集反馈: {feedback.feedback_type} from {feedback.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"收集反馈失败: {e}")
            return False
    
    async def _process_feedback_batch(self):
        """批处理反馈"""
        if not self.feedback_buffer:
            return
        
        # 处理反馈批次
        batch = self.feedback_buffer.copy()
        self.feedback_buffer.clear()
        
        # 分析反馈模式
        patterns = self._analyze_feedback_patterns(batch)
        
        # 生成学习样例
        learning_examples = self._generate_learning_examples(batch)
        
        # 触发模型更新
        if learning_examples:
            await self._trigger_model_update(learning_examples)
        
        logger.info(f"处理了 {len(batch)} 条反馈")
    
    def _analyze_feedback_patterns(
        self,
        feedback_batch: List[FeedbackItem]
    ) -> Dict[str, Any]:
        """分析反馈模式"""
        patterns = {
            "common_issues": [],
            "improvement_areas": [],
            "positive_aspects": []
        }
        
        # 按类型分组
        by_type = defaultdict(list)
        for feedback in feedback_batch:
            by_type[feedback.feedback_type].append(feedback)
        
        # 分析负面反馈
        if by_type["thumbs_down"]:
            # 提取共同问题
            issues = defaultdict(int)
            for fb in by_type["thumbs_down"]:
                if fb.content:
                    # 简单的关键词提取
                    keywords = fb.content.lower().split()
                    for keyword in keywords:
                        issues[keyword] += 1
            
            # 找出高频问题
            patterns["common_issues"] = [
                k for k, v in sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        
        # 分析纠正反馈
        if by_type["correction"]:
            patterns["improvement_areas"] = [
                fb.metadata.get("area", "general") for fb in by_type["correction"]
            ]
        
        # 分析正面反馈
        if by_type["thumbs_up"]:
            patterns["positive_aspects"] = [
                fb.metadata.get("aspect", "general") for fb in by_type["thumbs_up"]
            ]
        
        return patterns
    
    def _generate_learning_examples(
        self,
        feedback_batch: List[FeedbackItem]
    ) -> List[LearningExample]:
        """从反馈生成学习样例"""
        examples = []
        
        for feedback in feedback_batch:
            if feedback.feedback_type == "correction" and feedback.content:
                # 从纠正反馈创建学习样例
                example = LearningExample(
                    input_text=feedback.metadata.get("original_query", ""),
                    expected_output=feedback.content,
                    actual_output=feedback.metadata.get("original_response", ""),
                    feedback=feedback,
                    confidence=0.8 if feedback.rating and feedback.rating >= 4 else 0.6,
                    is_verified=False
                )
                examples.append(example)
        
        return examples
    
    async def _trigger_model_update(self, learning_examples: List[LearningExample]):
        """触发模型更新"""
        # 这里可以调用模型微调或参数更新逻辑
        logger.info(f"触发模型更新，样例数: {len(learning_examples)}")
    
    def get_feedback_summary(self, tenant_id: str) -> Dict[str, Any]:
        """获取反馈摘要"""
        stats = self.feedback_stats.get(tenant_id, {})
        
        satisfaction_rate = (
            stats["positive"] / stats["total"] * 100
            if stats["total"] > 0 else 0
        )
        
        return {
            "total_feedback": stats["total"],
            "satisfaction_rate": satisfaction_rate,
            "average_rating": stats["avg_rating"],
            "corrections_received": stats["corrections"],
            "feedback_distribution": {
                "positive": stats["positive"],
                "negative": stats["negative"],
                "corrections": stats["corrections"]
            }
        }


class ActiveLearner:
    """主动学习器"""
    
    def __init__(self, uncertainty_threshold: float = 0.3):
        self.uncertainty_threshold = uncertainty_threshold
        self.learning_pool: List[LearningExample] = []
        self.verified_examples: List[LearningExample] = []
        self.model_performance = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    async def identify_uncertain_examples(
        self,
        predictions: List[Tuple[str, float]],
        queries: List[str]
    ) -> List[int]:
        """识别不确定的样例"""
        uncertain_indices = []
        
        for i, (prediction, confidence) in enumerate(predictions):
            if confidence < self.uncertainty_threshold:
                uncertain_indices.append(i)
                
                # 创建学习样例
                example = LearningExample(
                    input_text=queries[i],
                    expected_output="",  # 待标注
                    actual_output=prediction,
                    confidence=confidence,
                    is_verified=False
                )
                self.learning_pool.append(example)
        
        logger.info(f"识别了 {len(uncertain_indices)} 个不确定样例")
        return uncertain_indices
    
    async def request_human_annotation(
        self,
        examples: List[LearningExample]
    ) -> List[LearningExample]:
        """请求人工标注"""
        annotated_examples = []
        
        for example in examples:
            # 这里应该有实际的人工标注接口
            # 模拟标注过程
            example.is_verified = True
            example.confidence = 1.0
            annotated_examples.append(example)
        
        self.verified_examples.extend(annotated_examples)
        
        logger.info(f"获得 {len(annotated_examples)} 个人工标注样例")
        return annotated_examples
    
    async def select_training_samples(
        self,
        strategy: str = "uncertainty"
    ) -> List[LearningExample]:
        """选择训练样本"""
        if strategy == "uncertainty":
            # 选择最不确定的样本
            sorted_examples = sorted(
                self.learning_pool,
                key=lambda x: x.confidence
            )
            return sorted_examples[:min(100, len(sorted_examples))]
        
        elif strategy == "diversity":
            # 选择多样化的样本
            return self._select_diverse_samples(self.learning_pool, 100)
        
        elif strategy == "representative":
            # 选择代表性样本
            return self._select_representative_samples(self.learning_pool, 100)
        
        else:
            # 随机选择
            import random
            return random.sample(
                self.learning_pool,
                min(100, len(self.learning_pool))
            )
    
    def _select_diverse_samples(
        self,
        examples: List[LearningExample],
        n: int
    ) -> List[LearningExample]:
        """选择多样化样本"""
        if len(examples) <= n:
            return examples
        
        selected = []
        remaining = examples.copy()
        
        # 贪心选择最不相似的样本
        while len(selected) < n and remaining:
            if not selected:
                # 随机选择第一个
                import random
                selected.append(remaining.pop(random.randint(0, len(remaining)-1)))
            else:
                # 选择与已选样本最不相似的
                max_min_distance = -1
                best_idx = 0
                
                for i, example in enumerate(remaining):
                    # 计算与所有已选样本的最小距离
                    min_distance = float('inf')
                    for selected_example in selected:
                        # 简单的文本相似度
                        distance = self._text_distance(
                            example.input_text,
                            selected_example.input_text
                        )
                        min_distance = min(min_distance, distance)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_idx = i
                
                selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _select_representative_samples(
        self,
        examples: List[LearningExample],
        n: int
    ) -> List[LearningExample]:
        """选择代表性样本"""
        # 使用聚类选择代表性样本
        # 简化实现：按长度分组，每组选择一些
        groups = defaultdict(list)
        
        for example in examples:
            length_group = len(example.input_text) // 50  # 按50字符分组
            groups[length_group].append(example)
        
        selected = []
        samples_per_group = max(1, n // len(groups))
        
        for group_examples in groups.values():
            import random
            selected.extend(
                random.sample(
                    group_examples,
                    min(samples_per_group, len(group_examples))
                )
            )
        
        return selected[:n]
    
    def _text_distance(self, text1: str, text2: str) -> float:
        """计算文本距离"""
        # 简单的Jaccard距离
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return 1.0 - (len(intersection) / len(union) if union else 0.0)
    
    async def update_model(
        self,
        training_examples: List[LearningExample],
        model_updater=None
    ) -> Dict[str, float]:
        """更新模型"""
        if not training_examples:
            return self.model_performance
        
        # 准备训练数据
        X = [ex.input_text for ex in training_examples]
        y = [ex.expected_output for ex in training_examples]
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 调用模型更新器
        if model_updater:
            await model_updater.update(X_train, y_train)
            
            # 评估性能
            predictions = await model_updater.predict(X_val)
            
            # 计算指标
            # 这里简化为二分类，实际应该根据任务类型调整
            y_val_binary = [1 if y else 0 for y in y_val]
            pred_binary = [1 if p else 0 for p in predictions]
            
            self.model_performance["accuracy"] = accuracy_score(y_val_binary, pred_binary)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val_binary, pred_binary, average='binary', zero_division=0
            )
            
            self.model_performance["precision"] = precision
            self.model_performance["recall"] = recall
            self.model_performance["f1"] = f1
        
        logger.info(f"模型更新完成，性能: {self.model_performance}")
        return self.model_performance


class AdaptiveLearningSystem:
    """自适应学习系统"""
    
    def __init__(self):
        self.feedback_collector = FeedbackCollector()
        self.active_learner = ActiveLearner()
        self.learning_history: List[Dict[str, Any]] = []
        self.adaptation_rules: Dict[str, Any] = {}
        self.performance_tracker = PerformanceTracker()
    
    async def adapt_to_user(
        self,
        user_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """适应用户偏好"""
        # 分析用户交互历史
        user_profile = self._analyze_user_profile(interaction_history)
        
        # 生成适应规则
        adaptation = {
            "response_style": self._determine_response_style(user_profile),
            "detail_level": self._determine_detail_level(user_profile),
            "preferred_examples": self._determine_example_preference(user_profile),
            "language_complexity": self._determine_language_complexity(user_profile)
        }
        
        # 存储适应规则
        self.adaptation_rules[user_id] = adaptation
        
        logger.info(f"为用户 {user_id} 生成适应规则")
        return adaptation
    
    def _analyze_user_profile(
        self,
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析用户画像"""
        profile = {
            "avg_query_length": 0,
            "query_complexity": 0,
            "preferred_response_length": 0,
            "interaction_frequency": 0,
            "feedback_rate": 0,
            "satisfaction_score": 0
        }
        
        if not interaction_history:
            return profile
        
        # 计算平均查询长度
        query_lengths = [len(i.get("query", "")) for i in interaction_history]
        profile["avg_query_length"] = np.mean(query_lengths) if query_lengths else 0
        
        # 计算查询复杂度（简化：基于长度和特殊词汇）
        complex_keywords = ["如何", "为什么", "原理", "机制", "详细", "深入"]
        complexity_scores = []
        for interaction in interaction_history:
            query = interaction.get("query", "").lower()
            score = sum(1 for keyword in complex_keywords if keyword in query)
            complexity_scores.append(score)
        profile["query_complexity"] = np.mean(complexity_scores) if complexity_scores else 0
        
        # 计算偏好的响应长度
        response_lengths = [len(i.get("response", "")) for i in interaction_history]
        profile["preferred_response_length"] = np.mean(response_lengths) if response_lengths else 0
        
        # 计算交互频率
        if len(interaction_history) > 1:
            timestamps = [i.get("timestamp") for i in interaction_history if i.get("timestamp")]
            if len(timestamps) > 1:
                # 计算平均时间间隔
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                    intervals.append(interval)
                avg_interval = np.mean(intervals) if intervals else 0
                profile["interaction_frequency"] = 1 / avg_interval if avg_interval > 0 else 0
        
        # 计算反馈率和满意度
        feedback_count = sum(1 for i in interaction_history if i.get("feedback"))
        profile["feedback_rate"] = feedback_count / len(interaction_history)
        
        satisfaction_scores = [
            i.get("feedback", {}).get("rating", 0)
            for i in interaction_history
            if i.get("feedback", {}).get("rating")
        ]
        profile["satisfaction_score"] = np.mean(satisfaction_scores) if satisfaction_scores else 0
        
        return profile
    
    def _determine_response_style(self, user_profile: Dict[str, Any]) -> str:
        """确定响应风格"""
        if user_profile["query_complexity"] > 2:
            return "technical"
        elif user_profile["avg_query_length"] < 20:
            return "concise"
        else:
            return "balanced"
    
    def _determine_detail_level(self, user_profile: Dict[str, Any]) -> str:
        """确定详细程度"""
        if user_profile["preferred_response_length"] > 500:
            return "detailed"
        elif user_profile["preferred_response_length"] < 100:
            return "brief"
        else:
            return "moderate"
    
    def _determine_example_preference(self, user_profile: Dict[str, Any]) -> bool:
        """确定是否偏好示例"""
        return user_profile["query_complexity"] > 1 or user_profile["avg_query_length"] > 50
    
    def _determine_language_complexity(self, user_profile: Dict[str, Any]) -> str:
        """确定语言复杂度"""
        if user_profile["query_complexity"] > 2:
            return "advanced"
        elif user_profile["query_complexity"] < 1:
            return "simple"
        else:
            return "intermediate"
    
    async def continuous_improvement(self):
        """持续改进循环"""
        while True:
            try:
                # 收集反馈
                await self.feedback_collector._process_feedback_batch()
                
                # 选择学习样本
                training_samples = await self.active_learner.select_training_samples()
                
                # 如果有足够的样本，触发模型更新
                if len(training_samples) >= 50:
                    performance = await self.active_learner.update_model(training_samples)
                    
                    # 记录性能
                    self.performance_tracker.record(performance)
                    
                    # 检查是否需要调整策略
                    if performance["f1"] < 0.7:
                        logger.warning("模型性能下降，需要更多训练数据")
                
                # 等待下一个周期
                await asyncio.sleep(3600)  # 每小时执行一次
                
            except Exception as e:
                logger.error(f"持续改进循环错误: {e}")
                await asyncio.sleep(60)  # 错误后等待1分钟


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.metrics_over_time = defaultdict(list)
    
    def record(self, metrics: Dict[str, float]):
        """记录性能指标"""
        record = {
            "timestamp": datetime.now(),
            "metrics": metrics
        }
        self.history.append(record)
        
        # 按指标类型记录
        for key, value in metrics.items():
            self.metrics_over_time[key].append({
                "timestamp": record["timestamp"],
                "value": value
            })
    
    def get_trend(self, metric: str, window: int = 10) -> str:
        """获取指标趋势"""
        if metric not in self.metrics_over_time:
            return "unknown"
        
        values = self.metrics_over_time[metric]
        if len(values) < 2:
            return "stable"
        
        recent_values = [v["value"] for v in values[-window:]]
        
        # 计算趋势
        if len(recent_values) < 2:
            return "stable"
        
        # 简单的线性趋势
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        # 计算斜率
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.history:
            return {}
        
        latest = self.history[-1]["metrics"]
        
        summary = {
            "latest_metrics": latest,
            "trends": {},
            "alerts": []
        }
        
        # 计算趋势
        for metric in latest.keys():
            summary["trends"][metric] = self.get_trend(metric)
        
        # 生成警报
        if latest.get("accuracy", 1.0) < 0.8:
            summary["alerts"].append("准确率低于80%")
        
        if latest.get("f1", 1.0) < 0.7:
            summary["alerts"].append("F1分数低于70%")
        
        # 检查下降趋势
        for metric, trend in summary["trends"].items():
            if trend == "declining":
                summary["alerts"].append(f"{metric} 呈下降趋势")
        
        return summary
