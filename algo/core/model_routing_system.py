"""
智能模型路由和切换系统
支持多AI模型统一管理、智能路由、负载均衡和故障切换
基于GitHub开源项目的最佳实践
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import numpy as np
from abc import ABC, abstractmethod
import hashlib
import random

logger = logging.getLogger(__name__)

class ModelType(Enum):
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    VISION = "vision"
    CODE_GENERATION = "code_generation"

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LOCAL = "local"

@dataclass
class ModelConfig:
    id: str
    name: str
    provider: ModelProvider
    model_type: ModelType
    endpoint: str
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    max_requests_per_minute: int = 60
    max_concurrent_requests: int = 10
    timeout: int = 30
    priority: int = 1  # 1=highest, 10=lowest
    enabled: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ModelMetrics:
    model_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    avg_cost: float = 0.0
    last_request_time: float = 0.0
    current_load: int = 0
    error_rate: float = 0.0
    availability: float = 1.0
    
    def update_request(self, success: bool, response_time: float, cost: float = 0.0):
        self.total_requests += 1
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # 更新平均响应时间
        if self.total_requests == 1:
            self.avg_response_time = response_time
            self.avg_cost = cost
        else:
            alpha = 0.1  # 指数移动平均
            self.avg_response_time = (1 - alpha) * self.avg_response_time + alpha * response_time
            self.avg_cost = (1 - alpha) * self.avg_cost + alpha * cost
        
        # 更新错误率
        self.error_rate = self.failed_requests / self.total_requests
        self.availability = self.successful_requests / self.total_requests

class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    FASTEST_RESPONSE = "fastest_response"
    LOWEST_COST = "lowest_cost"
    HIGHEST_QUALITY = "highest_quality"
    SMART_ROUTING = "smart_routing"

@dataclass
class RoutingRequest:
    model_type: ModelType
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 5  # 1=highest, 10=lowest
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RoutingResponse:
    model_id: str
    response: Any
    response_time: float
    cost: float
    tokens_used: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class ModelAdapter(ABC):
    """模型适配器基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """初始化适配器"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
    
    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def generate(self, request: RoutingRequest) -> RoutingResponse:
        """生成响应"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

class OpenAIAdapter(ModelAdapter):
    """OpenAI模型适配器"""
    
    async def generate(self, request: RoutingRequest) -> RoutingResponse:
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求数据
            data = {
                "model": self.config.name,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature
            }
            
            async with self.session.post(
                f"{self.config.endpoint}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    tokens_used = result["usage"]["total_tokens"]
                    cost = tokens_used * self.config.cost_per_1k_tokens / 1000
                    
                    return RoutingResponse(
                        model_id=self.config.id,
                        response=content,
                        response_time=response_time,
                        cost=cost,
                        tokens_used=tokens_used,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return RoutingResponse(
                        model_id=self.config.id,
                        response=None,
                        response_time=response_time,
                        cost=0.0,
                        tokens_used=0,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            response_time = time.time() - start_time
            return RoutingResponse(
                model_id=self.config.id,
                response=None,
                response_time=response_time,
                cost=0.0,
                tokens_used=0,
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            async with self.session.get(
                f"{self.config.endpoint}/models",
                headers=headers
            ) as response:
                return response.status == 200
        except:
            return False

class AnthropicAdapter(ModelAdapter):
    """Anthropic Claude模型适配器"""
    
    async def generate(self, request: RoutingRequest) -> RoutingResponse:
        start_time = time.time()
        
        try:
            headers = {
                "x-api-key": self.config.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.config.name,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
                "temperature": request.temperature or self.config.temperature
            }
            
            async with self.session.post(
                f"{self.config.endpoint}/messages",
                headers=headers,
                json=data
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    content = result["content"][0]["text"]
                    tokens_used = result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
                    cost = tokens_used * self.config.cost_per_1k_tokens / 1000
                    
                    return RoutingResponse(
                        model_id=self.config.id,
                        response=content,
                        response_time=response_time,
                        cost=cost,
                        tokens_used=tokens_used,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return RoutingResponse(
                        model_id=self.config.id,
                        response=None,
                        response_time=response_time,
                        cost=0.0,
                        tokens_used=0,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            response_time = time.time() - start_time
            return RoutingResponse(
                model_id=self.config.id,
                response=None,
                response_time=response_time,
                cost=0.0,
                tokens_used=0,
                success=False,
                error=str(e)
            )
    
    async def health_check(self) -> bool:
        # Anthropic没有专门的健康检查端点，使用简单请求测试
        try:
            test_request = RoutingRequest(
                model_type=ModelType.CHAT_COMPLETION,
                prompt="Hello",
                max_tokens=1
            )
            response = await self.generate(test_request)
            return response.success
        except:
            return False

class ModelRouter:
    """智能模型路由器"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.adapters: Dict[str, ModelAdapter] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.routing_strategy = RoutingStrategy.SMART_ROUTING
        self.round_robin_index = 0
        
        # 适配器工厂
        self.adapter_factory = {
            ModelProvider.OPENAI: OpenAIAdapter,
            ModelProvider.ANTHROPIC: AnthropicAdapter,
            # 可以添加更多适配器
        }
    
    async def initialize(self):
        """初始化路由器"""
        for adapter in self.adapters.values():
            await adapter.initialize()
        
        # 启动健康检查任务
        asyncio.create_task(self._health_check_loop())
        
        logger.info("Model router initialized")
    
    async def cleanup(self):
        """清理资源"""
        for adapter in self.adapters.values():
            await adapter.cleanup()
    
    def register_model(self, config: ModelConfig):
        """注册模型"""
        self.models[config.id] = config
        self.metrics[config.id] = ModelMetrics(model_id=config.id)
        
        # 创建适配器
        if config.provider in self.adapter_factory:
            adapter_class = self.adapter_factory[config.provider]
            self.adapters[config.id] = adapter_class(config)
        else:
            logger.warning(f"No adapter found for provider: {config.provider}")
    
    def unregister_model(self, model_id: str):
        """注销模型"""
        if model_id in self.models:
            del self.models[model_id]
        if model_id in self.adapters:
            del self.adapters[model_id]
        if model_id in self.metrics:
            del self.metrics[model_id]
    
    async def route_request(self, request: RoutingRequest) -> RoutingResponse:
        """路由请求到最适合的模型"""
        
        # 获取可用模型
        available_models = self._get_available_models(request.model_type)
        
        if not available_models:
            return RoutingResponse(
                model_id="",
                response=None,
                response_time=0.0,
                cost=0.0,
                tokens_used=0,
                success=False,
                error="No available models for this request type"
            )
        
        # 选择最佳模型
        selected_model = self._select_model(available_models, request)
        
        # 更新负载
        self.metrics[selected_model.id].current_load += 1
        
        try:
            # 执行请求
            adapter = self.adapters[selected_model.id]
            response = await adapter.generate(request)
            
            # 更新指标
            self.metrics[selected_model.id].update_request(
                success=response.success,
                response_time=response.response_time,
                cost=response.cost
            )
            
            return response
            
        except Exception as e:
            # 记录失败
            self.metrics[selected_model.id].update_request(
                success=False,
                response_time=0.0,
                cost=0.0
            )
            
            return RoutingResponse(
                model_id=selected_model.id,
                response=None,
                response_time=0.0,
                cost=0.0,
                tokens_used=0,
                success=False,
                error=str(e)
            )
        
        finally:
            # 减少负载
            self.metrics[selected_model.id].current_load -= 1
    
    def _get_available_models(self, model_type: ModelType) -> List[ModelConfig]:
        """获取可用模型列表"""
        available = []
        
        for model in self.models.values():
            if (model.enabled and 
                model.model_type == model_type and
                self.metrics[model.id].availability > 0.5):  # 可用性阈值
                available.append(model)
        
        return available
    
    def _select_model(self, available_models: List[ModelConfig], request: RoutingRequest) -> ModelConfig:
        """根据路由策略选择模型"""
        
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_models)
        
        elif self.routing_strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_models)
        
        elif self.routing_strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_models)
        
        elif self.routing_strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(available_models)
        
        elif self.routing_strategy == RoutingStrategy.LOWEST_COST:
            return self._lowest_cost_select(available_models)
        
        elif self.routing_strategy == RoutingStrategy.SMART_ROUTING:
            return self._smart_routing_select(available_models, request)
        
        else:
            # 默认使用第一个可用模型
            return available_models[0]
    
    def _round_robin_select(self, models: List[ModelConfig]) -> ModelConfig:
        """轮询选择"""
        selected = models[self.round_robin_index % len(models)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_round_robin_select(self, models: List[ModelConfig]) -> ModelConfig:
        """加权轮询选择"""
        weights = [1.0 / model.priority for model in models]
        total_weight = sum(weights)
        
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return models[i]
        
        return models[-1]
    
    def _least_connections_select(self, models: List[ModelConfig]) -> ModelConfig:
        """最少连接选择"""
        return min(models, key=lambda m: self.metrics[m.id].current_load)
    
    def _fastest_response_select(self, models: List[ModelConfig]) -> ModelConfig:
        """最快响应选择"""
        return min(models, key=lambda m: self.metrics[m.id].avg_response_time)
    
    def _lowest_cost_select(self, models: List[ModelConfig]) -> ModelConfig:
        """最低成本选择"""
        return min(models, key=lambda m: m.cost_per_1k_tokens)
    
    def _smart_routing_select(self, models: List[ModelConfig], request: RoutingRequest) -> ModelConfig:
        """智能路由选择"""
        # 综合考虑多个因素的评分算法
        
        def calculate_score(model: ModelConfig) -> float:
            metrics = self.metrics[model.id]
            
            # 基础分数
            score = 100.0
            
            # 优先级权重 (30%)
            priority_score = (11 - model.priority) / 10.0 * 30
            
            # 响应时间权重 (25%)
            max_response_time = max([self.metrics[m.id].avg_response_time for m in models])
            if max_response_time > 0:
                response_time_score = (1 - metrics.avg_response_time / max_response_time) * 25
            else:
                response_time_score = 25
            
            # 可用性权重 (20%)
            availability_score = metrics.availability * 20
            
            # 负载权重 (15%)
            max_load = max([self.metrics[m.id].current_load for m in models])
            if max_load > 0:
                load_score = (1 - metrics.current_load / max_load) * 15
            else:
                load_score = 15
            
            # 成本权重 (10%)
            max_cost = max([m.cost_per_1k_tokens for m in models])
            if max_cost > 0:
                cost_score = (1 - model.cost_per_1k_tokens / max_cost) * 10
            else:
                cost_score = 10
            
            total_score = priority_score + response_time_score + availability_score + load_score + cost_score
            
            # 请求优先级调整
            if request.priority <= 3:  # 高优先级请求
                total_score += priority_score * 0.5
            
            return total_score
        
        # 计算所有模型的分数并选择最高分
        scored_models = [(model, calculate_score(model)) for model in models]
        best_model = max(scored_models, key=lambda x: x[1])
        
        return best_model[0]
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                for model_id, adapter in self.adapters.items():
                    is_healthy = await adapter.health_check()
                    
                    if is_healthy:
                        self.models[model_id].enabled = True
                    else:
                        self.models[model_id].enabled = False
                        logger.warning(f"Model {model_id} health check failed")
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型的指标"""
        return {
            model_id: asdict(metrics)
            for model_id, metrics in self.metrics.items()
        }
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """获取模型状态"""
        status = {}
        
        for model_id, model in self.models.items():
            metrics = self.metrics[model_id]
            status[model_id] = {
                "name": model.name,
                "provider": model.provider.value,
                "type": model.model_type.value,
                "enabled": model.enabled,
                "priority": model.priority,
                "current_load": metrics.current_load,
                "availability": metrics.availability,
                "avg_response_time": metrics.avg_response_time,
                "error_rate": metrics.error_rate,
                "total_requests": metrics.total_requests
            }
        
        return status

# 使用示例和工厂函数
async def create_model_router() -> ModelRouter:
    """创建并初始化模型路由器"""
    router = ModelRouter()
    
    # 注册OpenAI模型
    openai_config = ModelConfig(
        id="gpt-4",
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        model_type=ModelType.CHAT_COMPLETION,
        endpoint="https://api.openai.com/v1",
        api_key="your-openai-api-key",
        cost_per_1k_tokens=0.03,
        priority=1
    )
    router.register_model(openai_config)
    
    # 注册Claude模型
    claude_config = ModelConfig(
        id="claude-3-sonnet",
        name="claude-3-sonnet-20240229",
        provider=ModelProvider.ANTHROPIC,
        model_type=ModelType.CHAT_COMPLETION,
        endpoint="https://api.anthropic.com/v1",
        api_key="your-anthropic-api-key",
        cost_per_1k_tokens=0.015,
        priority=2
    )
    router.register_model(claude_config)
    
    await router.initialize()
    return router

if __name__ == "__main__":
    # 测试代码
    async def test_model_routing():
        router = await create_model_router()
        
        # 测试请求
        request = RoutingRequest(
            model_type=ModelType.CHAT_COMPLETION,
            prompt="Hello, how are you?",
            max_tokens=100
        )
        
        # 路由请求
        response = await router.route_request(request)
        
        print(f"Model: {response.model_id}")
        print(f"Response: {response.response}")
        print(f"Success: {response.success}")
        print(f"Response time: {response.response_time:.3f}s")
        print(f"Cost: ${response.cost:.6f}")
        
        # 显示指标
        metrics = router.get_metrics()
        print("\nModel metrics:", json.dumps(metrics, indent=2))
        
        # 显示状态
        status = router.get_model_status()
        print("\nModel status:", json.dumps(status, indent=2))
        
        await router.cleanup()
    
    asyncio.run(test_model_routing())
