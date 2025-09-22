"""
RAG 基础接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class BaseEmbedder(ABC):
    """嵌入器基础接口"""
    
    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """为查询生成嵌入向量"""
        pass
    
    @abstractmethod
    def embed_passages(self, passages: List[str]) -> np.ndarray:
        """为文档段落生成嵌入向量"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """获取嵌入维度"""
        pass


class BaseRetriever(ABC):
    """检索器基础接口"""
    
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """检索相关文档"""
        pass
    
    @abstractmethod
    def add_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """添加文档到索引"""
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """保存索引"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """加载索引"""
        pass


class RetrievalResult:
    """检索结果"""
    
    def __init__(
        self,
        chunk_id: str,
        content: str,
        score: float,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.source = source
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata
        }
