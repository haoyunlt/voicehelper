"""
增强的模型路由系统
支持动态负载均衡、成本优化、故障转移
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque

from loguru import logger


class ModelProvider(Enum):
    """模型提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    ARK = "ark"  # 豆包
    QWEN = "qwen"  # 通义千问
    BAICHUAN = "baichuan"
    LOCAL = "local"


class ModelCapability(Enum):
    """模型能力"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"


class ModelTier(Enum):
    """模型层级"""
    BASIC = "basic"      # 基础模型，成本低
    STANDARD = "standard"  # 标准模型，平衡性能和成本
    PREMIUM = "premium"   # 高端模型，性能优先
    SPECIALIZED = "specialized"  # 专业模型，特定任务


@dataclass
class ModelConfig:
    """模型配置"""
    provider: ModelProvider
    model_name: str
    capabilities: List[ModelCapability]
    tier: ModelTier
    max_tokens: int
    cost_per_1k_tokens: float
    latency_p95: float  # P95延迟（毫秒）
    quality_score: float  # 质量评分 0-1
    rate_limit_rpm: int  # 每分钟请求限制
    rate_limit_tpm: int  # 每分钟token限制
    enabled: bool = True
    priority: int = 0  # 优先级，数字越大优先级越高
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """模型指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    error_rate: float = 0.0
    last_used: Optional[datetime] = None
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class RoutingRequest:
    """路由请求"""
    task_type: ModelCapability
    content: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    priority: str = "standard"  # low, standard, high
    cost_constraint: Optional[float] = None  # 最大成本约束
    latency_constraint: Optional[float] = None  # 最大延迟约束（毫秒）
    quality_threshold: Optional[float] = None  # 最小质量阈值
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """路由结果"""
    selected_model: ModelConfig
    routing_reason: str
    estimated_cost: float
    estimated_latency: float
    confidence_score: float
    fallback_models: List[ModelConfig] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoutingStrategy(Enum):
    """路由策略"""
    COST_OPTIMIZED = "cost_optimized"      # 成本优化
    LATENCY_OPTIMIZED = "latency_optimized"  # 延迟优化
    QUALITY_OPTIMIZED = "quality_optimized"  # 质量优化
    BALANCED = "balanced"                   # 平衡策略
    ROUND_ROBIN = "round_robin"            # 轮询
    WEIGHTED_RANDOM = "weighted_random"     # 加权随机
    LOAD_BALANCED = "load_balanced"        # 负载均衡


class EnhancedModelRouter:
    """增强的模型路由器"""
    
    def __init__(self, default_strategy: RoutingStrategy = RoutingStrategy.BALANCED):
        self.models: Dict[str, ModelConfig] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.default_strategy = default_strategy
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        self.round_robin_counters: Dict[ModelCapability, int] = defaultdict(int)
        
        # 初始化预定义模型
        self._initialize_predefined_models()
        
        # 启动后台任务
        self._start_background_tasks()
    
    def _initialize_predefined_models(self):
        """初始化预定义模型"""
        predefined_models = [
            # OpenAI模型
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4-turbo",
                capabilities=[ModelCapability.CHAT, ModelCapability.REASONING, ModelCapability.CODE_GENERATION],
                tier=ModelTier.PREMIUM,
                max_tokens=128000,
                cost_per_1k_tokens=0.03,
                latency_p95=2000,
                quality_score=0.95,
                rate_limit_rpm=500,
                rate_limit_tpm=150000,
                priority=90
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                capabilities=[ModelCapability.CHAT, ModelCapability.COMPLETION],
                tier=ModelTier.STANDARD,
                max_tokens=16385,
                cost_per_1k_tokens=0.002,
                latency_p95=1500,
                quality_score=0.85,
                rate_limit_rpm=3500,
                rate_limit_tpm=90000,
                priority=70
            ),
            
            # Anthropic模型
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-opus",
                capabilities=[ModelCapability.CHAT, ModelCapability.REASONING],
                tier=ModelTier.PREMIUM,
                max_tokens=200000,
                cost_per_1k_tokens=0.075,
                latency_p95=3000,
                quality_score=0.97,
                rate_limit_rpm=100,
                rate_limit_tpm=40000,
                priority=95
            ),
            ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet",
                capabilities=[ModelCapability.CHAT, ModelCapability.REASONING],
                tier=ModelTier.STANDARD,
                max_tokens=200000,
                cost_per_1k_tokens=0.015,
                latency_p95=2500,
                quality_score=0.90,
                rate_limit_rpm=300,
                rate_limit_tpm=60000,
                priority=80
            ),
            
            # 豆包模型
            ModelConfig(
                provider=ModelProvider.ARK,
                model_name="doubao-pro-4k",
                capabilities=[ModelCapability.CHAT, ModelCapability.COMPLETION],
                tier=ModelTier.STANDARD,
                max_tokens=4096,
                cost_per_1k_tokens=0.001,
                latency_p95=1200,
                quality_score=0.82,
                rate_limit_rpm=1000,
                rate_limit_tpm=100000,
                priority=75
            ),
            
            # 通义千问
            ModelConfig(
                provider=ModelProvider.QWEN,
                model_name="qwen-turbo",
                capabilities=[ModelCapability.CHAT, ModelCapability.COMPLETION],
                tier=ModelTier.BASIC,
                max_tokens=8192,
                cost_per_1k_tokens=0.0008,
                latency_p95=1000,
                quality_score=0.78,
                rate_limit_rpm=2000,
                rate_limit_tpm=120000,
                priority=60
            ),
        ]
        
        for model in predefined_models:
            self.register_model(model)
    
    def register_model(self, model: ModelConfig):
        """注册模型"""
        model_key = f"{model.provider.value}:{model.model_name}"
        self.models[model_key] = model
        self.metrics[model_key] = ModelMetrics()
        
        # 初始化熔断器
        self.circuit_breakers[model_key] = {
            "state": "closed",  # closed, open, half_open
            "failure_count": 0,
            "last_failure_time": None,
            "failure_threshold": 5,
            "recovery_timeout": 60  # 秒
        }
        
        # 初始化限流器
        self.rate_limiters[model_key] = {
            "requests": deque(maxlen=model.rate_limit_rpm),
            "tokens": deque(maxlen=1000)  # 保存最近的token使用记录
        }
        
        logger.info(f"注册模型: {model_key}")
    
    async def route_request(self, request: RoutingRequest, strategy: Optional[RoutingStrategy] = None) -> RoutingResult:
        """路由请求到最佳模型"""
        strategy = strategy or self.default_strategy
        
        # 获取候选模型
        candidate_models = self._get_candidate_models(request)
        if not candidate_models:
            raise ValueError(f"没有可用的模型支持任务类型: {request.task_type}")
        
        # 过滤不可用的模型
        available_models = []
        for model in candidate_models:
            model_key = f"{model.provider.value}:{model.model_name}"
            if self._is_model_available(model_key, request):
                available_models.append(model)
        
        if not available_models:
            raise ValueError("所有候选模型都不可用")
        
        # 根据策略选择模型
        selected_model, routing_reason = await self._select_model_by_strategy(
            available_models, request, strategy
        )
        
        # 计算预估指标
        estimated_cost = self._estimate_cost(selected_model, request)
        estimated_latency = self._estimate_latency(selected_model, request)
        confidence_score = self._calculate_confidence(selected_model, request)
        
        # 准备备选模型
        fallback_models = [m for m in available_models if m != selected_model][:3]
        
        result = RoutingResult(
            selected_model=selected_model,
            routing_reason=routing_reason,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            confidence_score=confidence_score,
            fallback_models=fallback_models
        )
        
        logger.info(f"路由选择: {selected_model.provider.value}:{selected_model.model_name}, 原因: {routing_reason}")
        return result
    
    def _get_candidate_models(self, request: RoutingRequest) -> List[ModelConfig]:
        """获取候选模型"""
        candidates = []
        
        for model in self.models.values():
            if not model.enabled:
                continue
            
            # 检查能力匹配
            if request.task_type not in model.capabilities:
                continue
            
            # 检查token限制
            if request.max_tokens and request.max_tokens > model.max_tokens:
                continue
            
            # 检查成本约束
            if request.cost_constraint:
                estimated_cost = self._estimate_cost(model, request)
                if estimated_cost > request.cost_constraint:
                    continue
            
            # 检查延迟约束
            if request.latency_constraint and model.latency_p95 > request.latency_constraint:
                continue
            
            # 检查质量阈值
            if request.quality_threshold and model.quality_score < request.quality_threshold:
                continue
            
            candidates.append(model)
        
        return candidates
    
    def _is_model_available(self, model_key: str, request: RoutingRequest) -> bool:
        """检查模型是否可用"""
        # 检查熔断器状态
        circuit_breaker = self.circuit_breakers[model_key]
        if circuit_breaker["state"] == "open":
            # 检查是否可以尝试恢复
            if circuit_breaker["last_failure_time"]:
                time_since_failure = time.time() - circuit_breaker["last_failure_time"]
                if time_since_failure > circuit_breaker["recovery_timeout"]:
                    circuit_breaker["state"] = "half_open"
                    logger.info(f"熔断器半开: {model_key}")
                else:
                    return False
            else:
                return False
        
        # 检查限流
        model = self.models[model_key]
        rate_limiter = self.rate_limiters[model_key]
        
        current_time = time.time()
        
        # 清理过期的请求记录
        while rate_limiter["requests"] and current_time - rate_limiter["requests"][0] > 60:
            rate_limiter["requests"].popleft()
        
        # 检查RPM限制
        if len(rate_limiter["requests"]) >= model.rate_limit_rpm:
            return False
        
        return True
    
    async def _select_model_by_strategy(
        self, 
        models: List[ModelConfig], 
        request: RoutingRequest, 
        strategy: RoutingStrategy
    ) -> Tuple[ModelConfig, str]:
        """根据策略选择模型"""
        
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._select_by_cost(models, request)
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return self._select_by_latency(models, request)
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return self._select_by_quality(models, request)
        elif strategy == RoutingStrategy.BALANCED:
            return self._select_balanced(models, request)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(models, request)
        elif strategy == RoutingStrategy.WEIGHTED_RANDOM:
            return self._select_weighted_random(models, request)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return self._select_load_balanced(models, request)
        else:
            return models[0], "默认选择"
    
    def _select_by_cost(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """按成本选择"""
        best_model = min(models, key=lambda m: self._estimate_cost(m, request))
        return best_model, "成本最优"
    
    def _select_by_latency(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """按延迟选择"""
        best_model = min(models, key=lambda m: self._estimate_latency(m, request))
        return best_model, "延迟最优"
    
    def _select_by_quality(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """按质量选择"""
        best_model = max(models, key=lambda m: m.quality_score)
        return best_model, "质量最优"
    
    def _select_balanced(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """平衡选择"""
        def score_model(model: ModelConfig) -> float:
            # 综合评分：质量 * 0.4 + (1/成本) * 0.3 + (1/延迟) * 0.3
            cost = self._estimate_cost(model, request)
            latency = self._estimate_latency(model, request)
            
            cost_score = 1.0 / (cost + 0.001)  # 避免除零
            latency_score = 1.0 / (latency + 0.001)
            
            # 归一化
            cost_score = min(cost_score, 1000) / 1000
            latency_score = min(latency_score, 1.0)
            
            return model.quality_score * 0.4 + cost_score * 0.3 + latency_score * 0.3
        
        best_model = max(models, key=score_model)
        return best_model, "平衡最优"
    
    def _select_round_robin(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """轮询选择"""
        counter = self.round_robin_counters[request.task_type]
        selected_model = models[counter % len(models)]
        self.round_robin_counters[request.task_type] = counter + 1
        return selected_model, "轮询选择"
    
    def _select_weighted_random(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """加权随机选择"""
        import random
        
        weights = [model.priority + 1 for model in models]  # 避免权重为0
        selected_model = random.choices(models, weights=weights)[0]
        return selected_model, "加权随机"
    
    def _select_load_balanced(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """负载均衡选择"""
        # 选择当前负载最低的模型
        def get_load_score(model: ModelConfig) -> float:
            model_key = f"{model.provider.value}:{model.model_name}"
            metrics = self.metrics[model_key]
            rate_limiter = self.rate_limiters[model_key]
            
            # 计算当前负载：最近请求数 / 限制数
            current_requests = len(rate_limiter["requests"])
            load_ratio = current_requests / model.rate_limit_rpm
            
            # 考虑错误率
            error_penalty = metrics.error_rate * 0.5
            
            return load_ratio + error_penalty
        
        best_model = min(models, key=get_load_score)
        return best_model, "负载均衡"
    
    def _estimate_cost(self, model: ModelConfig, request: RoutingRequest) -> float:
        """估算成本"""
        # 简单估算：根据内容长度估算token数
        estimated_tokens = len(request.content) // 4  # 粗略估算
        if request.max_tokens:
            estimated_tokens += request.max_tokens
        
        return (estimated_tokens / 1000) * model.cost_per_1k_tokens
    
    def _estimate_latency(self, model: ModelConfig, request: RoutingRequest) -> float:
        """估算延迟"""
        model_key = f"{model.provider.value}:{model.model_name}"
        metrics = self.metrics[model_key]
        
        # 如果有历史数据，使用平均延迟；否则使用配置的P95延迟
        if metrics.recent_latencies:
            return sum(metrics.recent_latencies) / len(metrics.recent_latencies)
        else:
            return model.latency_p95
    
    def _calculate_confidence(self, model: ModelConfig, request: RoutingRequest) -> float:
        """计算置信度"""
        model_key = f"{model.provider.value}:{model.model_name}"
        metrics = self.metrics[model_key]
        
        # 基础置信度基于模型质量评分
        base_confidence = model.quality_score
        
        # 根据历史成功率调整
        if metrics.total_requests > 0:
            success_rate = metrics.successful_requests / metrics.total_requests
            base_confidence = base_confidence * 0.7 + success_rate * 0.3
        
        # 根据熔断器状态调整
        circuit_breaker = self.circuit_breakers[model_key]
        if circuit_breaker["state"] == "half_open":
            base_confidence *= 0.8
        elif circuit_breaker["state"] == "open":
            base_confidence = 0.0
        
        return min(base_confidence, 1.0)
    
    def record_request_result(self, model_key: str, success: bool, latency: float, tokens: int, cost: float, error: str = ""):
        """记录请求结果"""
        if model_key not in self.metrics:
            return
        
        metrics = self.metrics[model_key]
        circuit_breaker = self.circuit_breakers[model_key]
        rate_limiter = self.rate_limiters[model_key]
        
        # 更新指标
        metrics.total_requests += 1
        metrics.total_tokens += tokens
        metrics.total_cost += cost
        metrics.last_used = datetime.now()
        
        # 记录请求时间（用于限流）
        rate_limiter["requests"].append(time.time())
        
        if success:
            metrics.successful_requests += 1
            metrics.recent_latencies.append(latency)
            
            # 更新平均延迟
            if metrics.recent_latencies:
                metrics.avg_latency = sum(metrics.recent_latencies) / len(metrics.recent_latencies)
            
            # 重置熔断器
            if circuit_breaker["state"] == "half_open":
                circuit_breaker["state"] = "closed"
                circuit_breaker["failure_count"] = 0
                logger.info(f"熔断器恢复: {model_key}")
        else:
            metrics.failed_requests += 1
            metrics.recent_errors.append({
                "timestamp": time.time(),
                "error": error
            })
            
            # 更新错误率
            metrics.error_rate = metrics.failed_requests / metrics.total_requests
            
            # 更新熔断器
            circuit_breaker["failure_count"] += 1
            circuit_breaker["last_failure_time"] = time.time()
            
            if circuit_breaker["failure_count"] >= circuit_breaker["failure_threshold"]:
                circuit_breaker["state"] = "open"
                logger.warning(f"熔断器打开: {model_key}, 失败次数: {circuit_breaker['failure_count']}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        status = {}
        
        for model_key, model in self.models.items():
            metrics = self.metrics[model_key]
            circuit_breaker = self.circuit_breakers[model_key]
            
            status[model_key] = {
                "provider": model.provider.value,
                "model_name": model.model_name,
                "enabled": model.enabled,
                "tier": model.tier.value,
                "circuit_breaker_state": circuit_breaker["state"],
                "total_requests": metrics.total_requests,
                "success_rate": metrics.successful_requests / max(metrics.total_requests, 1),
                "error_rate": metrics.error_rate,
                "avg_latency": metrics.avg_latency,
                "total_cost": metrics.total_cost,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None,
            }
        
        return status
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 这里可以启动定期清理、指标计算等后台任务
        pass


# 全局模型路由器实例
model_router = EnhancedModelRouter()
