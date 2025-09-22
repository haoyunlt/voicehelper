"""
配置模块
"""

from .bge_config import RAGConfig, BGEConfig, FAISSConfig, DocumentConfig, load_rag_config, default_rag_config

__all__ = [
    "RAGConfig",
    "BGEConfig", 
    "FAISSConfig",
    "DocumentConfig",
    "load_rag_config",
    "default_rag_config"
]
