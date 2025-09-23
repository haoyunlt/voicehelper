"""
RAG (Retrieval-Augmented Generation) 模块
基于 BGE + FAISS 的高效中文检索系统
"""

from .base import BaseEmbedder, BaseRetriever, RetrievalResult
from .embedder_bge import BgeEmbedder
from .retriever_faiss import FaissRetriever
from .ingest_faiss import DocumentChunker, FaissIndexBuilder, build_faiss_index, build_faiss_index_async
from .metrics import RAGMetricsCollector, get_metrics_collector, record_retrieval_metrics, record_index_load_metrics

__all__ = [
    # 基础接口
    "BaseEmbedder",
    "BaseRetriever", 
    "RetrievalResult",
    
    # BGE 嵌入
    "BgeEmbedder",
    
    # FAISS 检索
    "FaissRetriever",
    
    # 索引构建
    "DocumentChunker",
    "FaissIndexBuilder",
    "build_faiss_index",
    "build_faiss_index_async",
    
    # 指标监控
    "RAGMetricsCollector",
    "get_metrics_collector",
    "record_retrieval_metrics",
    "record_index_load_metrics"
]
