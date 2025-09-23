"""
BGE + FAISS 配置
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os


@dataclass
class BGEConfig:
    """BGE嵌入模型配置"""
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cpu"
    normalize: bool = True
    cache_folder: Optional[str] = None
    batch_size: int = 32
    
    # 指令前缀
    query_instruction: str = "为这个问题生成表示用于检索相关文档："
    passage_instruction: str = "为这段文本生成表示用于检索相关文档："


@dataclass
class FAISSConfig:
    """FAISS索引配置"""
    index_type: str = "HNSW32,Flat"
    ef_construction: int = 200
    ef_search: int = 64
    
    # 检索参数
    top_k: int = 5
    score_threshold: float = 0.3
    
    # 存储配置
    data_dir: str = "data/faiss"
    tenant_based: bool = True


@dataclass
class DocumentConfig:
    """文档处理配置"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ['。', '！', '？', '\n\n', '\n', ' ']


@dataclass
class RAGConfig:
    """RAG系统配置"""
    bge: BGEConfig
    faiss: FAISSConfig
    document: DocumentConfig
    
    # 性能配置
    enable_metrics: bool = True
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    # 质量目标
    target_recall_at_5: float = 0.85
    target_p95_latency_ms: float = 120.0
    max_ingest_time_minutes: float = 10.0
    max_memory_gb: float = 3.0


def load_rag_config() -> RAGConfig:
    """加载RAG配置"""
    
    # BGE配置
    bge_config = BGEConfig(
        model_name=os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-zh-v1.5"),
        device=os.getenv("BGE_DEVICE", "cpu"),
        normalize=os.getenv("BGE_NORMALIZE", "true").lower() == "true",
        cache_folder=os.getenv("BGE_CACHE_FOLDER"),
        batch_size=int(os.getenv("BGE_BATCH_SIZE", "32"))
    )
    
    # FAISS配置
    faiss_config = FAISSConfig(
        index_type=os.getenv("FAISS_INDEX_TYPE", "HNSW32,Flat"),
        ef_construction=int(os.getenv("FAISS_EF_CONSTRUCTION", "200")),
        ef_search=int(os.getenv("FAISS_EF_SEARCH", "64")),
        top_k=int(os.getenv("FAISS_TOP_K", "5")),
        score_threshold=float(os.getenv("FAISS_SCORE_THRESHOLD", "0.3")),
        data_dir=os.getenv("FAISS_DATA_DIR", "data/faiss"),
        tenant_based=os.getenv("FAISS_TENANT_BASED", "true").lower() == "true"
    )
    
    # 文档配置
    document_config = DocumentConfig(
        chunk_size=int(os.getenv("DOC_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("DOC_CHUNK_OVERLAP", "50"))
    )
    
    # RAG配置
    rag_config = RAGConfig(
        bge=bge_config,
        faiss=faiss_config,
        document=document_config,
        enable_metrics=os.getenv("RAG_ENABLE_METRICS", "true").lower() == "true",
        enable_cache=os.getenv("RAG_ENABLE_CACHE", "true").lower() == "true",
        cache_ttl_seconds=int(os.getenv("RAG_CACHE_TTL", "3600"))
    )
    
    return rag_config


def get_tenant_data_dir(base_dir: str, tenant_id: str, dataset_id: str = "default") -> str:
    """获取租户数据目录"""
    return f"{base_dir}/tenants/{tenant_id}/datasets/{dataset_id}"


# 默认配置实例
default_rag_config = load_rag_config()
