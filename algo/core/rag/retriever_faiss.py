"""
FAISS 检索器实现
基于 Facebook AI Similarity Search 的高效向量检索
"""

import json
import faiss
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger

from .base import BaseRetriever, BaseEmbedder, RetrievalResult
from .metrics import record_retrieval_metrics, record_index_load_metrics, MetricsTimer, get_metrics_collector
from common.errors import VoiceHelperError


class FaissRetriever(BaseRetriever):
    """FAISS 检索器"""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        index_type: str = "HNSW32,Flat",
        ef_construction: int = 200,
        ef_search: int = 64
    ):
        super().__init__(embedder)
        self.index_type = index_type
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self.index = None
        self.metadata = []
        self.index_path = None
        self.metadata_path = None
        
    def _create_index(self) -> faiss.Index:
        """创建FAISS索引"""
        try:
            dimension = self.embedder.dimension
            logger.info(f"创建FAISS索引: {self.index_type}, 维度: {dimension}")
            
            if self.index_type.startswith("HNSW"):
                # HNSW索引配置 - 使用内积相似度（适合归一化向量）
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexHNSWFlat(quantizer, dimension, 32)
                index.hnsw.efConstruction = self.ef_construction
                index.hnsw.efSearch = self.ef_search
            elif self.index_type == "Flat":
                # 平面索引 - 精确搜索
                index = faiss.IndexFlatIP(dimension)
            else:
                # 使用工厂方法创建复合索引
                index = faiss.index_factory(dimension, self.index_type)
            
            logger.info("FAISS索引创建完成")
            return index
            
        except Exception as e:
            logger.error(f"FAISS索引创建失败: {e}")
            raise VoiceHelperError("FAISS_INDEX_CREATE_FAILED", f"Failed to create FAISS index: {e}")
    
    def _load(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """加载索引和元数据"""
        if self.index is None:
            raise VoiceHelperError("FAISS_INDEX_NOT_LOADED", "Index not loaded")
        return self.index, self.metadata
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """检索相关文档"""
        start_time = time.perf_counter()
        
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("索引为空或未加载")
                return []
            
            # 生成查询向量
            query_vector = self.embedder.embed_single_query(query)
            
            # 搜索相似向量
            scores, indices = self._search_vectors(query_vector, top_k * 2)  # 获取更多结果用于过滤
            
            # 构建结果
            results = []
            for score, idx in zip(scores, indices):
                if idx == -1 or score < score_threshold:
                    continue
                
                if idx >= len(self.metadata):
                    logger.warning(f"索引超出范围: {idx} >= {len(self.metadata)}")
                    continue
                
                metadata = self.metadata[idx]
                result = {
                    "id": metadata.get("chunk_id", f"chunk_{idx}"),
                    "content": metadata.get("text", ""),
                    "score": float(score),
                    "source": metadata.get("source", ""),
                    "metadata": metadata
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            # 记录指标
            retrieval_time_ms = (time.perf_counter() - start_time) * 1000
            record_retrieval_metrics(
                query=query,
                retrieval_time_ms=retrieval_time_ms,
                results=results,
                retriever_type="faiss"
            )
            
            logger.info(f"检索完成，查询: {query[:50]}..., 结果数: {len(results)}, 耗时: {retrieval_time_ms:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise VoiceHelperError("FAISS_RETRIEVE_FAILED", f"Failed to retrieve: {e}")
    
    def _search_vectors(
        self, 
        query_vector: np.ndarray, 
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """搜索相似向量"""
        try:
            # 设置搜索参数
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = self.ef_search
            
            # 确保查询向量格式正确
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # 执行搜索
            scores, indices = self.index.search(query_vector, top_k)
            
            return scores[0], indices[0]
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise VoiceHelperError("FAISS_SEARCH_FAILED", f"Failed to search vectors: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """添加文档到索引"""
        try:
            if not documents:
                return {"added": 0, "total": 0}
            
            # 提取文本内容
            texts = []
            metadata = []
            
            for doc in documents:
                if isinstance(doc, dict) and "text" in doc:
                    texts.append(doc["text"])
                    metadata.append(doc)
                elif isinstance(doc, str):
                    texts.append(doc)
                    metadata.append({"text": doc})
                else:
                    logger.warning(f"跳过无效文档: {doc}")
                    continue
            
            if not texts:
                return {"added": 0, "total": 0}
            
            # 生成嵌入向量
            embeddings = self.embedder.embed_passages(texts)
            
            # 创建索引（如果不存在）
            if self.index is None:
                self.index = self._create_index()
            
            # 添加向量到索引
            self.index.add(embeddings)
            
            # 保存元数据
            self.metadata.extend(metadata)
            
            result = {
                "added": len(texts),
                "total": self.index.ntotal,
                "metadata_count": len(self.metadata)
            }
            
            logger.info(f"文档添加完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"文档添加失败: {e}")
            raise VoiceHelperError("FAISS_ADD_DOCUMENTS_FAILED", f"Failed to add documents: {e}")
    
    def save_index(self, path: str) -> None:
        """保存索引"""
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS索引
            index_path = str(path_obj / "index.faiss")
            metadata_path = str(path_obj / "meta.json")
            
            if self.index is not None:
                faiss.write_index(self.index, index_path)
            
            # 保存元数据
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            self.index_path = index_path
            self.metadata_path = metadata_path
            
            logger.info(f"索引已保存: {index_path}, 元数据: {metadata_path}")
            
        except Exception as e:
            logger.error(f"索引保存失败: {e}")
            raise VoiceHelperError("FAISS_SAVE_FAILED", f"Failed to save index: {e}")
    
    def load_index(self, path: str) -> None:
        """加载索引"""
        start_time = time.perf_counter()
        
        try:
            path_obj = Path(path)
            index_path = str(path_obj / "index.faiss")
            metadata_path = str(path_obj / "meta.json")
            
            # 检查文件是否存在
            if not Path(index_path).exists():
                logger.warning(f"索引文件不存在: {index_path}")
                return
            
            if not Path(metadata_path).exists():
                logger.warning(f"元数据文件不存在: {metadata_path}")
                return
            
            # 加载FAISS索引
            self.index = faiss.read_index(index_path)
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.index_path = index_path
            self.metadata_path = metadata_path
            
            # 记录加载指标
            load_time_ms = (time.perf_counter() - start_time) * 1000
            index_size_mb = Path(index_path).stat().st_size / (1024 * 1024)
            
            record_index_load_metrics(
                load_time_ms=load_time_ms,
                index_size_mb=index_size_mb,
                vector_count=self.index.ntotal,
                dimension=self.embedder.dimension,
                index_type=self.index_type
            )
            
            logger.info(f"索引已加载: {index_path}, 向量数: {self.index.ntotal}, 元数据数: {len(self.metadata)}, 耗时: {load_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            raise VoiceHelperError("FAISS_LOAD_FAILED", f"Failed to load index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal if self.index else 0,
            "metadata_count": len(self.metadata),
            "dimension": self.embedder.dimension,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search
        }
