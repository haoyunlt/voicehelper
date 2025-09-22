"""
基于V2架构的BGE+FAISS检索器
继承BaseRetriever，集成BGE嵌入和FAISS向量检索
"""

import os
import json
import time
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger

from ..base.runnable import BaseRetriever
from ..base.protocols import StreamCallback

# 简化的指标记录
class MetricsTimer:
    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000

def record_retrieval_metrics(query: str, result_count: int, latency_ms: float, avg_score: float):
    """记录检索指标"""
    logger.info(f"检索指标: query_len={len(query)}, results={result_count}, latency={latency_ms:.1f}ms, avg_score={avg_score:.3f}")


class BGEFaissRetriever(BaseRetriever):
    """BGE+FAISS检索器 - V2架构实现"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        index_type: str = "HNSW32,Flat", 
        ef_construction: int = 200,
        ef_search: int = 64,
        tenant_id: str = "default",
        dataset_id: str = "default",
        data_dir: str = "data/faiss",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # BGE模型配置
        self.model_name = model_name
        self.model = None
        self.dimension = 1024  # BGE-large默认维度
        
        # FAISS索引配置
        self.index_type = index_type
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # 租户和数据集
        self.tenant_id = tenant_id
        self.dataset_id = dataset_id
        self.data_dir = Path(data_dir)
        
        # 索引和元数据
        self.index = None
        self.metadata = []
        
        # 路径配置
        self._setup_paths()
    
    def _setup_paths(self):
        """设置文件路径"""
        self.tenant_dir = self.data_dir / "tenants" / self.tenant_id / "datasets" / self.dataset_id
        self.index_path = self.tenant_dir / "index.faiss"
        self.metadata_path = self.tenant_dir / "meta.json"
        
        # 确保目录存在
        self.tenant_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_bge_model(self):
        """加载BGE模型"""
        if self.model is None:
            try:
                logger.info(f"加载BGE模型: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"BGE模型加载完成，维度: {self.dimension}")
            except Exception as e:
                logger.error(f"BGE模型加载失败: {e}")
                raise
    
    def _encode_texts(self, texts: List[str], instruction: str = "") -> np.ndarray:
        """编码文本为向量"""
        self._load_bge_model()
        
        try:
            # 添加指令前缀（BGE模型特性）
            if instruction:
                texts = [f"{instruction}{text}" for text in texts]
            
            # 编码并归一化
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise
    
    def _create_faiss_index(self) -> faiss.Index:
        """创建FAISS索引"""
        try:
            logger.info(f"创建FAISS索引: {self.index_type}, 维度: {self.dimension}")
            
            if self.index_type.startswith("HNSW"):
                # HNSW索引 - 使用内积相似度（适合归一化向量）
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexHNSWFlat(quantizer, self.dimension, 32)
                index.hnsw.efConstruction = self.ef_construction
                index.hnsw.efSearch = self.ef_search
                logger.info(f"HNSW索引创建完成: ef_construction={self.ef_construction}, ef_search={self.ef_search}")
                
            elif self.index_type == "Flat":
                # 平坦索引 - 精确搜索
                index = faiss.IndexFlatIP(self.dimension)
                logger.info("Flat索引创建完成")
                
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
            
            return index
            
        except Exception as e:
            logger.error(f"FAISS索引创建失败: {e}")
            raise
    
    def build_index(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        构建索引
        
        Args:
            documents: 文档列表，每个文档包含id、content等字段
            
        Returns:
            构建结果统计
        """
        try:
            logger.info(f"开始构建索引，文档数量: {len(documents)}")
            start_time = time.time()
            
            # 提取文档内容
            texts = []
            metadata = []
            
            for doc in documents:
                content = doc.get("content", "")
                if content.strip():
                    texts.append(content)
                    metadata.append({
                        "id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "url": doc.get("url", ""),
                        "metadata": doc.get("metadata", {})
                    })
            
            if not texts:
                raise ValueError("没有有效的文档内容")
            
            # 生成嵌入向量
            logger.info("生成文档嵌入向量...")
            embeddings = self._encode_texts(texts, instruction="为这个句子生成表示以用于检索相关文章：")
            
            # 创建索引
            self.index = self._create_faiss_index()
            
            # 添加向量到索引
            logger.info("添加向量到FAISS索引...")
            self.index.add(embeddings)
            self.metadata = metadata
            
            # 保存索引和元数据
            self._save_index()
            
            build_time = time.time() - start_time
            result = {
                "status": "success",
                "doc_count": len(documents),
                "vector_count": len(texts),
                "dimension": self.dimension,
                "build_time": build_time,
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path)
            }
            
            logger.info(f"索引构建完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise
    
    def _save_index(self):
        """保存索引和元数据"""
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"FAISS索引已保存: {self.index_path}")
            
            # 保存元数据
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": self.metadata,
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "model_name": self.model_name,
                    "vector_count": len(self.metadata)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"元数据已保存: {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise
    
    def _load_index(self):
        """加载索引和元数据"""
        try:
            if not self.index_path.exists() or not self.metadata_path.exists():
                logger.warning(f"索引文件不存在: {self.index_path}")
                return False
            
            # 加载FAISS索引
            self.index = faiss.read_index(str(self.index_path))
            
            # 设置搜索参数
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = self.ef_search
            
            # 加载元数据
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata = data["metadata"]
                self.dimension = data["dimension"]
            
            logger.info(f"索引加载完成: {len(self.metadata)} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            return False
    
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        检索文档 - 实现BaseRetriever抽象方法
        
        Args:
            query: 查询文本
            **kwargs: 检索参数
            
        Returns:
            检索结果列表
        """
        top_k = kwargs.get('top_k', self.top_k)
        score_threshold = kwargs.get('score_threshold', 0.0)
        
        try:
            # 确保索引已加载
            if self.index is None:
                if not self._load_index():
                    return []
            
            # 编码查询
            query_embedding = self._encode_texts([query], instruction="为这个句子生成表示以用于检索相关文章：")
            
            # 执行检索
            with MetricsTimer() as timer:
                scores, indices = self.index.search(query_embedding, top_k)
            
            # 处理结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1 or score < score_threshold:
                    continue
                
                metadata = self.metadata[idx]
                results.append({
                    "content": metadata.get("title", "") + " " + metadata.get("content", ""),
                    "score": float(score),
                    "source": metadata.get("source", ""),
                    "url": metadata.get("url", ""),
                    "id": metadata.get("id", ""),
                    "metadata": metadata.get("metadata", {})
                })
            
            # 记录指标
            record_retrieval_metrics(
                query=query,
                result_count=len(results),
                latency_ms=timer.elapsed_ms,
                avg_score=np.mean([r["score"] for r in results]) if results else 0.0
            )
            
            logger.info(f"检索完成: query='{query[:50]}...', results={len(results)}, latency={timer.elapsed_ms:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        if self.index is None:
            self._load_index()
        
        return {
            "model_name": self.model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "vector_count": len(self.metadata) if self.metadata else 0,
            "tenant_id": self.tenant_id,
            "dataset_id": self.dataset_id,
            "index_exists": self.index is not None
        }
