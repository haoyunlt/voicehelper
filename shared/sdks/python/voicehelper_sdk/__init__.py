"""
VoiceHelper SDK - v1.9.0
智能聊天机器人系统 Python SDK

Usage:
    from voicehelper_sdk import VoiceHelperSDK, VoiceHelperError
    
    # 创建SDK实例
    sdk = VoiceHelperSDK(api_key="your_api_key")
    
    # 聊天对话
    response = await sdk.create_chat_completion([
        {"role": "user", "content": "你好"}
    ])
    
    # 语音合成
    audio_data = await sdk.synthesize_voice("你好，世界！")
    
    # 图像分析
    with open("image.jpg", "rb") as f:
        result = await sdk.analyze_image(f.read())
"""

from .client import VoiceHelperSDK
from .exceptions import VoiceHelperError, APIError, NetworkError, AuthenticationError
from .types import (
    # 基础类型
    ChatMessage,
    Attachment,
    MultimodalConfig,
    
    # 配置类型
    VoiceHelperConfig,
    VoiceSynthesisOptions,
    VoiceRecognitionOptions,
    VisionAnalysisOptions,
    
    # 响应类型
    ChatCompletionResponse,
    VoiceSynthesisResponse,
    VoiceRecognitionResponse,
    VisionAnalysisResponse,
    MultimodalFusionResponse,
    MCPServiceCallResponse,
    SystemHealth,
    SystemMetrics,
    
    # 枚举类型
    ModelType,
    MessageRole,
    AttachmentType,
    EmotionType,
    VoiceType,
    LanguageCode,
    ModalityType,
    FusionStrategy,
    ServiceCategory,
    ServiceStatus,
)

from .utils import (
    quick_chat,
    quick_tts,
    quick_vision,
    create_message,
    create_attachment,
    validate_config,
)

__version__ = "1.9.0"
__author__ = "VoiceHelper Team"
__email__ = "sdk@voicehelper.com"
__description__ = "智能聊天机器人系统 Python SDK"

__all__ = [
    # 主要类
    "VoiceHelperSDK",
    
    # 异常类
    "VoiceHelperError",
    "APIError", 
    "NetworkError",
    "AuthenticationError",
    
    # 类型定义
    "ChatMessage",
    "Attachment",
    "MultimodalConfig",
    "VoiceHelperConfig",
    "VoiceSynthesisOptions",
    "VoiceRecognitionOptions", 
    "VisionAnalysisOptions",
    "ChatCompletionResponse",
    "VoiceSynthesisResponse",
    "VoiceRecognitionResponse",
    "VisionAnalysisResponse",
    "MultimodalFusionResponse",
    "MCPServiceCallResponse",
    "SystemHealth",
    "SystemMetrics",
    
    # 枚举
    "ModelType",
    "MessageRole",
    "AttachmentType",
    "EmotionType",
    "VoiceType",
    "LanguageCode",
    "ModalityType",
    "FusionStrategy",
    "ServiceCategory",
    "ServiceStatus",
    
    # 工具函数
    "quick_chat",
    "quick_tts", 
    "quick_vision",
    "create_message",
    "create_attachment",
    "validate_config",
    
    # 版本信息
    "__version__",
]
