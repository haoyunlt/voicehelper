"""
多模型调用服务
支持自动切换、负载均衡和故障转移
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .multi_model_config import MultiModelConfig, ModelProvider, ModelConfig

@dataclass
class ModelStats:
    """模型统计信息"""
    total_calls: int = 0
    success_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.success_calls / self.total_calls
    
    @property
    def is_healthy(self) -> bool:
        # 如果最近5分钟内有成功调用，或者还没有调用过，认为是健康的
        if self.last_success is None:
            return self.failed_calls < 3  # 连续失败3次以下认为健康
        
        return (datetime.now() - self.last_success) < timedelta(minutes=5)

class MultiModelService:
    """多模型调用服务"""
    
    def __init__(self, config: MultiModelConfig):
        self.config = config
        self.stats: Dict[str, ModelStats] = {}
        self.circuit_breaker: Dict[str, bool] = {}  # 熔断器状态
        self.last_health_check = datetime.now()
        
        # 初始化统计信息
        for model_name in self.config.get_available_models():
            self.stats[model_name] = ModelStats()
            self.circuit_breaker[model_name] = False
    
    def _get_headers(self, model_config: ModelConfig) -> Dict[str, str]:
        """根据模型提供商获取请求头"""
        if model_config.provider == ModelProvider.DOUBAO:
            return {
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json"
            }
        elif model_config.provider == ModelProvider.GLM:
            return {
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json"
            }
        elif model_config.provider == ModelProvider.QWEN:
            return {
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-SSE": "enable"  # 通义千问流式输出
            }
        elif model_config.provider == ModelProvider.ERNIE:
            return {
                "Content-Type": "application/json"
            }
        elif model_config.provider == ModelProvider.HUNYUAN:
            return {
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json"
            }
        else:
            return {
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json"
            }
    
    def _build_request_payload(
        self, 
        model_config: ModelConfig, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """根据模型提供商构建请求载荷"""
        base_payload = {
            "model": model_config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", model_config.temperature),
            "max_tokens": kwargs.get("max_tokens", model_config.max_tokens),
        }
        
        # 添加流式输出支持
        if kwargs.get("stream", False) and model_config.supports_streaming:
            base_payload["stream"] = True
        
        # 根据提供商调整参数
        if model_config.provider == ModelProvider.QWEN:
            # 通义千问特殊参数
            base_payload["top_p"] = kwargs.get("top_p", 0.8)
            if "stream" in base_payload:
                base_payload["incremental_output"] = True
        elif model_config.provider == ModelProvider.ERNIE:
            # 文心一言特殊参数
            base_payload["top_p"] = kwargs.get("top_p", 0.8)
            base_payload["penalty_score"] = kwargs.get("penalty_score", 1.0)
        elif model_config.provider == ModelProvider.HUNYUAN:
            # 混元特殊参数
            base_payload["top_p"] = kwargs.get("top_p", 0.8)
        
        return base_payload
    
    def _get_api_endpoint(self, model_config: ModelConfig) -> str:
        """获取API端点"""
        if model_config.provider == ModelProvider.DOUBAO:
            return f"{model_config.base_url}/chat/completions"
        elif model_config.provider == ModelProvider.GLM:
            return f"{model_config.base_url}/chat/completions"
        elif model_config.provider == ModelProvider.QWEN:
            return f"{model_config.base_url}/chat/completions"
        elif model_config.provider == ModelProvider.ERNIE:
            # 文心一言需要在URL中包含access_token
            return f"{model_config.base_url}/wenxinworkshop/chat/completions?access_token={model_config.api_key}"
        elif model_config.provider == ModelProvider.HUNYUAN:
            return f"{model_config.base_url}/chat/completions"
        else:
            return f"{model_config.base_url}/chat/completions"
    
    def _select_model(self, preferred_model: Optional[str] = None) -> Optional[str]:
        """选择可用的模型"""
        # 健康检查
        self._health_check()
        
        # 如果指定了首选模型且可用，使用首选模型
        if preferred_model and preferred_model in self.config.models:
            if not self.circuit_breaker.get(preferred_model, False):
                return preferred_model
        
        # 使用主要模型
        primary = self.config.primary_model
        if primary and not self.circuit_breaker.get(primary, False):
            return primary
        
        # 使用备用模型
        for fallback in self.config.fallback_models:
            if not self.circuit_breaker.get(fallback, False):
                return fallback
        
        # 如果所有模型都被熔断，选择成功率最高的
        available_models = [
            name for name in self.config.get_available_models()
            if self.stats[name].success_rate > 0
        ]
        
        if available_models:
            return max(available_models, key=lambda x: self.stats[x].success_rate)
        
        # 最后尝试使用任何可用的模型
        return self.config.primary_model
    
    def _health_check(self):
        """健康检查和熔断器管理"""
        now = datetime.now()
        
        # 每分钟检查一次
        if (now - self.last_health_check) < timedelta(minutes=1):
            return
        
        self.last_health_check = now
        
        for model_name, stats in self.stats.items():
            # 重置熔断器（给模型恢复的机会）
            if self.circuit_breaker.get(model_name, False):
                if stats.is_healthy or stats.failed_calls == 0:
                    self.circuit_breaker[model_name] = False
                    print(f"模型 {model_name} 熔断器已重置")
            
            # 设置熔断器
            elif not stats.is_healthy and stats.success_rate < 0.5:
                self.circuit_breaker[model_name] = True
                print(f"模型 {model_name} 已被熔断，成功率: {stats.success_rate:.2%}")
    
    def _update_stats(
        self, 
        model_name: str, 
        success: bool, 
        response_time: float, 
        tokens: int = 0, 
        cost: float = 0.0, 
        error: Optional[str] = None
    ):
        """更新模型统计信息"""
        stats = self.stats[model_name]
        
        stats.total_calls += 1
        if success:
            stats.success_calls += 1
            stats.last_success = datetime.now()
        else:
            stats.failed_calls += 1
            stats.last_error = error
        
        stats.total_tokens += tokens
        stats.total_cost += cost
        
        # 更新平均响应时间
        if stats.total_calls == 1:
            stats.avg_response_time = response_time
        else:
            stats.avg_response_time = (
                (stats.avg_response_time * (stats.total_calls - 1) + response_time) 
                / stats.total_calls
            )
    
    async def call_model(
        self, 
        messages: List[Dict[str, str]], 
        preferred_model: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """调用模型（非流式）"""
        model_name = self._select_model(preferred_model)
        if not model_name:
            return None, "no_available_model"
        
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            return None, "model_config_not_found"
        
        start_time = time.time()
        
        try:
            headers = self._get_headers(model_config)
            payload = self._build_request_payload(model_config, messages, **kwargs)
            endpoint = self._get_api_endpoint(model_config)
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 30)
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # 计算token使用量和成本
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = input_tokens + output_tokens
                
                cost = model_config.cost_per_1k_input * (input_tokens / 1000) + \
                       model_config.cost_per_1k_output * (output_tokens / 1000)
                
                self._update_stats(model_name, True, response_time, total_tokens, cost)
                
                return result, model_name
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self._update_stats(model_name, False, response_time, error=error_msg)
                return None, f"api_error_{response.status_code}"
                
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            self._update_stats(model_name, False, response_time, error=error_msg)
            return None, f"exception_{type(e).__name__}"
    
    async def stream_model(
        self, 
        messages: List[Dict[str, str]], 
        preferred_model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """调用模型（流式）"""
        model_name = self._select_model(preferred_model)
        if not model_name:
            yield "error", "no_available_model"
            return
        
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            yield "error", "model_config_not_found"
            return
        
        if not model_config.supports_streaming:
            # 如果不支持流式，降级到普通调用
            result, used_model = await self.call_model(messages, preferred_model, **kwargs)
            if result and "choices" in result:
                content = result["choices"][0]["message"]["content"]
                yield "content", content
                yield "done", used_model
            else:
                yield "error", used_model
            return
        
        start_time = time.time()
        
        try:
            headers = self._get_headers(model_config)
            payload = self._build_request_payload(model_config, messages, stream=True, **kwargs)
            endpoint = self._get_api_endpoint(model_config)
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=kwargs.get("timeout", 30)
            )
            
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self._update_stats(model_name, False, time.time() - start_time, error=error_msg)
                yield "error", f"api_error_{response.status_code}"
                return
            
            total_content = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        
                        if data == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    total_content += content
                                    yield "content", content
                        except json.JSONDecodeError:
                            continue
            
            response_time = time.time() - start_time
            
            # 估算token使用量（基于内容长度）
            estimated_tokens = len(total_content) // 2  # 粗略估算
            cost = model_config.cost_per_1k_output * (estimated_tokens / 1000)
            
            self._update_stats(model_name, True, response_time, estimated_tokens, cost)
            yield "done", model_name
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            self._update_stats(model_name, False, response_time, error=error_msg)
            yield "error", f"exception_{type(e).__name__}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "models": {
                name: {
                    "total_calls": stats.total_calls,
                    "success_calls": stats.success_calls,
                    "failed_calls": stats.failed_calls,
                    "success_rate": stats.success_rate,
                    "total_tokens": stats.total_tokens,
                    "total_cost": stats.total_cost,
                    "avg_response_time": stats.avg_response_time,
                    "is_healthy": stats.is_healthy,
                    "circuit_breaker": self.circuit_breaker.get(name, False),
                    "last_error": stats.last_error
                }
                for name, stats in self.stats.items()
            },
            "config": self.config.get_model_info()
        }
    
    def reset_stats(self, model_name: Optional[str] = None):
        """重置统计信息"""
        if model_name:
            if model_name in self.stats:
                self.stats[model_name] = ModelStats()
                self.circuit_breaker[model_name] = False
        else:
            for name in self.stats:
                self.stats[name] = ModelStats()
                self.circuit_breaker[name] = False

# 全局多模型服务实例
multi_model_service = None

def get_multi_model_service() -> MultiModelService:
    """获取多模型服务实例"""
    global multi_model_service
    if multi_model_service is None:
        from .multi_model_config import multi_model_config
        multi_model_service = MultiModelService(multi_model_config)
    return multi_model_service
