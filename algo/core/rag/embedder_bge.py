"""
BGE Embedder 实现
基于 BAAI/bge-large-zh-v1.5 的中文优化嵌入模型
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger

from common.errors import VoiceHelperError


class BgeEmbedder:
    """BGE 嵌入器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda",
        normalize: bool = True,
        cache_folder: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.cache_folder = cache_folder
        self._model = None
        self._dimension = None
        
    @property
    def model(self) -> SentenceTransformer:
        """延迟加载模型"""
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def dimension(self) -> int:
        """获取嵌入维度"""
        if self._dimension is None:
            _ = self.model  # 触发模型加载
        return self._dimension
    
    def load_model(self):
        """加载BGE模型"""
        try:
            logger.info(f"加载BGE模型: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"BGE模型加载完成，维度: {self._dimension}")
        except Exception as e:
            logger.error(f"BGE模型加载失败: {e}")
            raise VoiceHelperError("BGE_MODEL_LOAD_FAILED", f"Failed to load BGE model: {e}")
    
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        为查询生成嵌入向量
        BGE模型建议为查询添加特定指令前缀
        """
        try:
            # 为查询添加指令前缀
            prefixed_queries = [
                f"为这个问题生成表示用于检索相关文档：{query}" 
                for query in queries
            ]
            
            embeddings = self.model.encode(
                prefixed_queries,
                normalize_embeddings=self.normalize,
                show_progress_bar=len(queries) > 10,
                convert_to_numpy=True
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise VoiceHelperError("BGE_QUERY_EMBED_FAILED", f"Failed to embed queries: {e}")
    
    def embed_passages(self, passages: List[str]) -> np.ndarray:
        """
        为文档段落生成嵌入向量
        BGE模型建议为文档添加特定指令前缀
        """
        try:
            # 为文档添加指令前缀
            prefixed_passages = [
                f"为这段文本生成表示用于检索相关文档：{passage}" 
                for passage in passages
            ]
            
            embeddings = self.model.encode(
                prefixed_passages,
                normalize_embeddings=self.normalize,
                show_progress_bar=len(passages) > 10,
                convert_to_numpy=True
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"文档嵌入失败: {e}")
            raise VoiceHelperError("BGE_PASSAGE_EMBED_FAILED", f"Failed to embed passages: {e}")
    
    def embed_single_query(self, query: str) -> np.ndarray:
        """嵌入单个查询"""
        return self.embed_queries([query])[0]
    
    def embed_single_passage(self, passage: str) -> np.ndarray:
        """嵌入单个文档"""
        return self.embed_passages([passage])[0]
