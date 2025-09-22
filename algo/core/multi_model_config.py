"""
多模型配置和管理模块
支持豆包、通义千问、GLM-4、文心一言等国内大模型
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class ModelProvider(Enum):
    """模型提供商枚举"""
    DOUBAO = "doubao"           # 豆包 (字节跳动)
    QWEN = "qwen"              # 通义千问 (阿里云)
    GLM = "glm"                # GLM-4 (智谱AI)
    ERNIE = "ernie"            # 文心一言 (百度)
    HUNYUAN = "hunyuan"        # 混元 (腾讯)

@dataclass
class ModelConfig:
    """单个模型配置"""
    provider: ModelProvider
    model_name: str
    api_key: str
    base_url: str
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9
    cost_per_1k_input: float = 0.001  # 每千tokens输入成本(元)
    cost_per_1k_output: float = 0.002  # 每千tokens输出成本(元)
    context_length: int = 4096
    supports_streaming: bool = True
    supports_function_calling: bool = False
    
class MultiModelConfig:
    """多模型配置管理"""
    
    def __init__(self):
        self.models = self._load_models()
        self.primary_model = self._get_primary_model()
        self.fallback_models = self._get_fallback_models()
    
    def _load_models(self) -> Dict[str, ModelConfig]:
        """加载所有可用模型配置"""
        models = {}
        
        # 豆包大模型配置
        if os.getenv("ARK_API_KEY"):
            models["doubao-pro-4k"] = ModelConfig(
                provider=ModelProvider.DOUBAO,
                model_name="doubao-pro-4k",
                api_key=os.getenv("ARK_API_KEY", ""),
                base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.0008,
                cost_per_1k_output=0.002,
                context_length=4096,
                supports_streaming=True,
                supports_function_calling=True
            )
            
            models["doubao-lite-4k"] = ModelConfig(
                provider=ModelProvider.DOUBAO,
                model_name="doubao-lite-4k", 
                api_key=os.getenv("ARK_API_KEY", ""),
                base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.0003,
                cost_per_1k_output=0.0006,
                context_length=4096,
                supports_streaming=True,
                supports_function_calling=True
            )
        
        # 通义千问配置
        if os.getenv("QWEN_API_KEY"):
            models["qwen-turbo"] = ModelConfig(
                provider=ModelProvider.QWEN,
                model_name="qwen-turbo",
                api_key=os.getenv("QWEN_API_KEY", ""),
                base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.002,
                cost_per_1k_output=0.006,
                context_length=8192,
                supports_streaming=True,
                supports_function_calling=True
            )
            
            models["qwen-plus"] = ModelConfig(
                provider=ModelProvider.QWEN,
                model_name="qwen-plus",
                api_key=os.getenv("QWEN_API_KEY", ""),
                base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.004,
                cost_per_1k_output=0.012,
                context_length=32768,
                supports_streaming=True,
                supports_function_calling=True
            )
        
        # GLM-4 配置
        if os.getenv("GLM_API_KEY"):
            models["glm-4-flash"] = ModelConfig(
                provider=ModelProvider.GLM,
                model_name="glm-4-flash",
                api_key=os.getenv("GLM_API_KEY", ""),
                base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0001,
                context_length=128000,
                supports_streaming=True,
                supports_function_calling=True
            )
            
            models["glm-4-air"] = ModelConfig(
                provider=ModelProvider.GLM,
                model_name="glm-4-air",
                api_key=os.getenv("GLM_API_KEY", ""),
                base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.001,
                context_length=128000,
                supports_streaming=True,
                supports_function_calling=True
            )
        
        # 文心一言配置
        if os.getenv("ERNIE_API_KEY"):
            models["ernie-lite-8k"] = ModelConfig(
                provider=ModelProvider.ERNIE,
                model_name="ernie-lite-8k",
                api_key=os.getenv("ERNIE_API_KEY", ""),
                base_url=os.getenv("ERNIE_BASE_URL", "https://aip.baidubce.com/rpc/2.0/ai_custom/v1"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.0008,
                cost_per_1k_output=0.002,
                context_length=8192,
                supports_streaming=True,
                supports_function_calling=False
            )
        
        # 混元配置
        if os.getenv("HUNYUAN_API_KEY"):
            models["hunyuan-lite"] = ModelConfig(
                provider=ModelProvider.HUNYUAN,
                model_name="hunyuan-lite",
                api_key=os.getenv("HUNYUAN_API_KEY", ""),
                base_url=os.getenv("HUNYUAN_BASE_URL", "https://hunyuan.tencentcloudapi.com"),
                max_tokens=2000,
                temperature=0.7,
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                context_length=32768,
                supports_streaming=True,
                supports_function_calling=True
            )
        
        return models
    
    def _get_primary_model(self) -> Optional[str]:
        """获取主要模型"""
        # 优先级: 环境变量指定 > 豆包 > GLM > 通义千问 > 其他
        primary = os.getenv("PRIMARY_MODEL")
        if primary and primary in self.models:
            return primary
        
        # 按成本和性能优先级选择
        priority_order = [
            "doubao-pro-4k",    # 豆包 Pro - 平衡性能和成本
            "doubao-lite-4k",   # 豆包 Lite - 极致成本
            "glm-4-flash",      # GLM Flash - 最便宜
            "glm-4-air",        # GLM Air - 性能更好
            "qwen-turbo",       # 通义千问 Turbo
            "ernie-lite-8k",    # 文心 Lite
            "hunyuan-lite"      # 混元 Lite
        ]
        
        for model_name in priority_order:
            if model_name in self.models:
                return model_name
        
        # 如果都没有，返回第一个可用的
        return next(iter(self.models.keys())) if self.models else None
    
    def _get_fallback_models(self) -> List[str]:
        """获取备用模型列表"""
        fallback_order = [
            "glm-4-flash",      # 最便宜的备用
            "doubao-lite-4k",   # 豆包备用
            "qwen-turbo",       # 通义千问备用
            "ernie-lite-8k",    # 文心备用
        ]
        
        fallbacks = []
        primary = self.primary_model
        
        for model_name in fallback_order:
            if model_name in self.models and model_name != primary:
                fallbacks.append(model_name)
        
        return fallbacks
    
    def get_model_config(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """获取模型配置"""
        if model_name is None:
            model_name = self.primary_model
        
        return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """获取所有可用模型列表"""
        return list(self.models.keys())
    
    def get_models_by_provider(self, provider: ModelProvider) -> List[str]:
        """根据提供商获取模型列表"""
        return [
            name for name, config in self.models.items() 
            if config.provider == provider
        ]
    
    def get_cheapest_model(self) -> Optional[str]:
        """获取成本最低的模型"""
        if not self.models:
            return None
        
        cheapest = min(
            self.models.items(),
            key=lambda x: x[1].cost_per_1k_input + x[1].cost_per_1k_output
        )
        return cheapest[0]
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """估算调用成本"""
        config = self.get_model_config(model_name)
        if not config:
            return 0.0
        
        input_cost = (input_tokens / 1000) * config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * config.cost_per_1k_output
        
        return input_cost + output_cost
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型配置信息摘要"""
        info = {
            "total_models": len(self.models),
            "primary_model": self.primary_model,
            "fallback_models": self.fallback_models,
            "providers": list(set(config.provider.value for config in self.models.values())),
            "models": {}
        }
        
        for name, config in self.models.items():
            info["models"][name] = {
                "provider": config.provider.value,
                "context_length": config.context_length,
                "cost_per_1k": config.cost_per_1k_input + config.cost_per_1k_output,
                "supports_streaming": config.supports_streaming,
                "supports_function_calling": config.supports_function_calling
            }
        
        return info

# 全局多模型配置实例
multi_model_config = MultiModelConfig()
