"""
LLM智能路由 - 根据复杂度和延迟预算选择最优模型
"""
import re
import asyncio
from typing import Dict, List, Optional, Tuple
import structlog
from dataclasses import dataclass

from app.config import settings

logger = structlog.get_logger()


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: str
    max_tokens: int
    cost_per_1k_tokens: float
    avg_latency_ms: int
    capability_score: float  # 0-1, 能力评分


class LLMRouter:
    """LLM路由器"""
    
    def __init__(self):
        self.models = self._init_models()
        self.load_stats = {}  # 模型负载统计
        self.cache_hit_rates = {}  # 缓存命中率
    
    def _init_models(self) -> Dict[str, ModelConfig]:
        """初始化可用模型配置"""
        return {
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                max_tokens=4096,
                cost_per_1k_tokens=0.002,
                avg_latency_ms=800,
                capability_score=0.7
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                provider="openai",
                max_tokens=8192,
                cost_per_1k_tokens=0.0015,
                avg_latency_ms=1200,
                capability_score=0.85
            ),
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                provider="openai",
                max_tokens=8192,
                cost_per_1k_tokens=0.03,
                avg_latency_ms=2000,
                capability_score=0.95
            ),
            "claude-3-haiku": ModelConfig(
                name="claude-3-haiku-20240307",
                provider="anthropic",
                max_tokens=4096,
                cost_per_1k_tokens=0.00025,
                avg_latency_ms=600,
                capability_score=0.75
            ),
            "claude-3-sonnet": ModelConfig(
                name="claude-3-sonnet-20240229",
                provider="anthropic",
                max_tokens=4096,
                cost_per_1k_tokens=0.003,
                avg_latency_ms=1500,
                capability_score=0.9
            )
        }
    
    async def route_request(
        self,
        text: str,
        context: Optional[Dict] = None,
        latency_budget_ms: int = 700,
        cost_priority: float = 0.3,  # 0-1, 成本优先级
        quality_priority: float = 0.7  # 0-1, 质量优先级
    ) -> Tuple[str, Dict]:
        """
        路由请求到最优模型
        
        Args:
            text: 输入文本
            context: 上下文信息
            latency_budget_ms: 延迟预算
            cost_priority: 成本优先级权重
            quality_priority: 质量优先级权重
        
        Returns:
            (model_name, routing_info)
        """
        try:
            # 分析请求复杂度
            complexity_score = self._analyze_complexity(text, context)
            
            # 获取候选模型
            candidates = self._get_candidate_models(
                complexity_score, 
                latency_budget_ms
            )
            
            if not candidates:
                # 降级到最快的模型
                fallback_model = min(
                    self.models.values(), 
                    key=lambda m: m.avg_latency_ms
                )
                logger.warning(
                    "No suitable model found, using fallback",
                    fallback=fallback_model.name,
                    complexity=complexity_score,
                    budget_ms=latency_budget_ms
                )
                return fallback_model.name, {"reason": "fallback"}
            
            # 选择最优模型
            best_model = self._select_optimal_model(
                candidates,
                complexity_score,
                cost_priority,
                quality_priority
            )
            
            routing_info = {
                "complexity_score": complexity_score,
                "candidates_count": len(candidates),
                "selection_reason": "optimal",
                "estimated_latency_ms": best_model.avg_latency_ms,
                "estimated_cost": best_model.cost_per_1k_tokens
            }
            
            logger.info(
                "LLM routing decision",
                model=best_model.name,
                complexity=complexity_score,
                budget_ms=latency_budget_ms,
                **routing_info
            )
            
            return best_model.name, routing_info
        
        except Exception as e:
            logger.error("Error in LLM routing", error=str(e))
            # 默认模型
            return "gpt-3.5-turbo", {"reason": "error", "error": str(e)}
    
    def _analyze_complexity(
        self, 
        text: str, 
        context: Optional[Dict] = None
    ) -> float:
        """
        分析请求复杂度 (0-1)
        
        考虑因素:
        - 文本长度
        - 实体数量
        - 语法复杂度
        - 上下文需求
        - 特殊任务类型
        """
        score = 0.0
        
        # 基于文本长度
        text_length = len(text)
        if text_length > 500:
            score += 0.3
        elif text_length > 200:
            score += 0.2
        elif text_length > 50:
            score += 0.1
        
        # 检测复杂任务类型
        complex_patterns = [
            r'分析|推理|解释|比较|评估',  # 分析类任务
            r'代码|编程|算法|函数',      # 编程类任务
            r'翻译|转换|改写',          # 转换类任务
            r'创作|写作|故事|诗歌',      # 创作类任务
            r'计算|数学|公式',          # 计算类任务
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, text):
                score += 0.15
        
        # 检测实体和专业术语
        entity_patterns = [
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # 人名/地名
            r'\d{4}-\d{2}-\d{2}',          # 日期
            r'\$\d+|\d+元',                # 金额
            r'https?://\S+',               # URL
        ]
        
        entity_count = sum(
            len(re.findall(pattern, text)) 
            for pattern in entity_patterns
        )
        score += min(entity_count * 0.05, 0.2)
        
        # 上下文复杂度
        if context:
            if context.get('conversation_length', 0) > 10:
                score += 0.1
            if context.get('has_tools', False):
                score += 0.15
            if context.get('multimodal', False):
                score += 0.2
        
        return min(score, 1.0)
    
    def _get_candidate_models(
        self, 
        complexity_score: float, 
        latency_budget_ms: int
    ) -> List[ModelConfig]:
        """获取满足延迟要求的候选模型"""
        candidates = []
        
        for model in self.models.values():
            # 检查延迟预算
            if model.avg_latency_ms <= latency_budget_ms:
                # 检查能力匹配
                if model.capability_score >= complexity_score * 0.8:
                    candidates.append(model)
        
        return candidates
    
    def _select_optimal_model(
        self,
        candidates: List[ModelConfig],
        complexity_score: float,
        cost_priority: float,
        quality_priority: float
    ) -> ModelConfig:
        """从候选模型中选择最优模型"""
        best_model = None
        best_score = -1
        
        for model in candidates:
            # 计算综合评分
            # 质量评分 (能力匹配度)
            quality_match = min(model.capability_score / max(complexity_score, 0.1), 1.0)
            quality_score = quality_match * quality_priority
            
            # 成本评分 (成本越低越好)
            max_cost = max(m.cost_per_1k_tokens for m in candidates)
            cost_score = (1 - model.cost_per_1k_tokens / max_cost) * cost_priority
            
            # 负载评分 (当前负载越低越好)
            load_factor = self.load_stats.get(model.name, 0.0)
            load_score = (1 - load_factor) * 0.1
            
            total_score = quality_score + cost_score + load_score
            
            if total_score > best_score:
                best_score = total_score
                best_model = model
        
        return best_model or candidates[0]
    
    async def update_model_stats(
        self, 
        model_name: str, 
        latency_ms: int, 
        success: bool
    ):
        """更新模型性能统计"""
        if model_name in self.models:
            # 更新平均延迟 (指数移动平均)
            current_avg = self.models[model_name].avg_latency_ms
            alpha = 0.1
            new_avg = alpha * latency_ms + (1 - alpha) * current_avg
            self.models[model_name].avg_latency_ms = int(new_avg)
            
            # 更新负载统计
            if model_name not in self.load_stats:
                self.load_stats[model_name] = 0.0
            
            # 简单的负载衰减
            self.load_stats[model_name] = max(
                0.0, 
                self.load_stats[model_name] - 0.01
            )
            
            if not success:
                # 失败时增加负载惩罚
                self.load_stats[model_name] += 0.1
    
    def get_model_status(self) -> Dict:
        """获取所有模型状态"""
        return {
            "models": {
                name: {
                    "config": model.__dict__,
                    "load": self.load_stats.get(name, 0.0),
                    "cache_hit_rate": self.cache_hit_rates.get(name, 0.0)
                }
                for name, model in self.models.items()
            },
            "routing_strategy": settings.LLM_ROUTER
        }
