"""
V2架构ASR/TTS适配器
提供OpenAI、Azure等供应商的ASR/TTS适配器实现
"""

from .base import BaseAsrAdapter, BaseTtsAdapter
from .openai import OpenAIAsrAdapter, OpenAITtsAdapter
from .azure import AzureAsrAdapter, AzureTtsAdapter

__all__ = [
    "BaseAsrAdapter",
    "BaseTtsAdapter", 
    "OpenAIAsrAdapter",
    "OpenAITtsAdapter",
    "AzureAsrAdapter",
    "AzureTtsAdapter"
]
