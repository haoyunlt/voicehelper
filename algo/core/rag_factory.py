"""
RAG 服务工厂
统一创建和管理 BGE + FAISS RAG 服务
"""

import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

from core.rag import BgeEmbedder, FaissRetriever, FaissIndexBuilder, DocumentChunker
from core.config.bge_config import RAGConfig, load_rag_config, get_tenant_data_dir
from core.llm import LLMService
from common.errors import VoiceHelperError


class RAGServiceFactory:
    """RAG服务工厂"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or load_rag_config()
        self._embedders: Dict[str, BgeEmbedder] = {}
        self._retrievers: Dict[str, FaissRetriever] = {}
        
    def create_embedder(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> BgeEmbedder:
        """创建BGE嵌入器"""
        model_name = model_name or self.config.bge.model_name
        device = device or self.config.bge.device
        
        cache_key = f"{model_name}_{device}"
        
        if cache_key not in self._embedders:
            embedder = BgeEmbedder(
                model_name=model_name,
                device=device,
                normalize=self.config.bge.normalize,
                cache_folder=self.config.bge.cache_folder
            )
            self._embedders[cache_key] = embedder
            logger.info(f"创建BGE嵌入器: {model_name} on {device}")
        
        return self._embedders[cache_key]
    
    def create_retriever(
        self,
        embedder: Optional[BgeEmbedder] = None,
        tenant_id: str = "default",
        dataset_id: str = "default"
    ) -> FaissRetriever:
        """创建FAISS检索器"""
        if embedder is None:
            embedder = self.create_embedder()
        
        cache_key = f"{tenant_id}_{dataset_id}"
        
        if cache_key not in self._retrievers:
            retriever = FaissRetriever(
                embedder=embedder,
                index_type=self.config.faiss.index_type,
                ef_construction=self.config.faiss.ef_construction,
                ef_search=self.config.faiss.ef_search
            )
            
            # 尝试加载已有索引
            if self.config.faiss.tenant_based:
                data_dir = get_tenant_data_dir(
                    self.config.faiss.data_dir,
                    tenant_id,
                    dataset_id
                )
            else:
                data_dir = f"{self.config.faiss.data_dir}/{dataset_id}"
            
            try:
                retriever.load_index(data_dir)
                logger.info(f"加载已有索引: {data_dir}")
            except Exception as e:
                logger.info(f"未找到已有索引，将创建新索引: {e}")
            
            self._retrievers[cache_key] = retriever
        
        return self._retrievers[cache_key]
    
    def create_index_builder(
        self,
        embedder: Optional[BgeEmbedder] = None
    ) -> FaissIndexBuilder:
        """创建索引构建器"""
        if embedder is None:
            embedder = self.create_embedder()
        
        chunker = DocumentChunker(
            chunk_size=self.config.document.chunk_size,
            chunk_overlap=self.config.document.chunk_overlap,
            separators=self.config.document.separators
        )
        
        return FaissIndexBuilder(
            embedder=embedder,
            chunker=chunker,
            index_type=self.config.faiss.index_type,
            batch_size=self.config.bge.batch_size
        )
    
    async def build_index_from_documents(
        self,
        documents: list,
        tenant_id: str = "default",
        dataset_id: str = "default",
        embedder: Optional[BgeEmbedder] = None
    ) -> Dict[str, Any]:
        """从文档构建索引"""
        try:
            # 创建索引构建器
            builder = self.create_index_builder(embedder)
            
            # 确定输出目录
            if self.config.faiss.tenant_based:
                output_dir = get_tenant_data_dir(
                    self.config.faiss.data_dir,
                    tenant_id,
                    dataset_id
                )
            else:
                output_dir = f"{self.config.faiss.data_dir}/{dataset_id}"
            
            # 构建索引
            result = await builder.build_from_documents(
                documents=documents,
                output_dir=output_dir,
                dataset_id=dataset_id
            )
            
            # 清除缓存的检索器，强制重新加载
            cache_key = f"{tenant_id}_{dataset_id}"
            if cache_key in self._retrievers:
                del self._retrievers[cache_key]
            
            logger.info(f"索引构建完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise VoiceHelperError("INDEX_BUILD_FAILED", f"Failed to build index: {e}")
    
    def get_retriever_stats(
        self,
        tenant_id: str = "default",
        dataset_id: str = "default"
    ) -> Dict[str, Any]:
        """获取检索器统计信息"""
        cache_key = f"{tenant_id}_{dataset_id}"
        
        if cache_key in self._retrievers:
            retriever = self._retrievers[cache_key]
            return retriever.get_stats()
        
        return {"error": "Retriever not found"}
    
    def clear_cache(self):
        """清除缓存"""
        self._embedders.clear()
        self._retrievers.clear()
        logger.info("RAG服务缓存已清除")


# 全局工厂实例
_global_rag_factory = RAGServiceFactory()


def get_rag_factory() -> RAGServiceFactory:
    """获取全局RAG工厂"""
    return _global_rag_factory


def create_embedder(**kwargs) -> BgeEmbedder:
    """创建嵌入器的便捷函数"""
    return _global_rag_factory.create_embedder(**kwargs)


def create_retriever(**kwargs) -> FaissRetriever:
    """创建检索器的便捷函数"""
    return _global_rag_factory.create_retriever(**kwargs)


def create_index_builder(**kwargs) -> FaissIndexBuilder:
    """创建索引构建器的便捷函数"""
    return _global_rag_factory.create_index_builder(**kwargs)


async def build_index_from_documents(**kwargs) -> Dict[str, Any]:
    """构建索引的便捷函数"""
    return await _global_rag_factory.build_index_from_documents(**kwargs)
