"""
BGE + FAISS RAG实现
基于BGE嵌入模型和FAISS向量数据库的检索增强生成系统
"""

import os
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

from core.llm import LLMService
from common.errors import VoiceHelperError


class BGEEmbeddingService:
    """BGE嵌入服务"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.model = None
        self.dimension = 1024  # BGE-large的维度
        
    def load_model(self):
        """加载BGE模型"""
        try:
            logger.info(f"加载BGE模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"BGE模型加载完成，维度: {self.dimension}")
        except Exception as e:
            logger.error(f"BGE模型加载失败: {e}")
            raise VoiceHelperError("EMBEDDING_MODEL_LOAD_FAILED", f"Failed to load BGE model: {e}")
    
    def encode(self, texts: List[str], instruction: str = "") -> np.ndarray:
        """编码文本为向量"""
        if self.model is None:
            self.load_model()
        
        try:
            # 添加指令前缀（BGE模型特性）
            if instruction:
                texts = [f"{instruction}{text}" for text in texts]
            
            # 编码并归一化
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            raise VoiceHelperError("EMBEDDING_ENCODE_FAILED", f"Failed to encode texts: {e}")


class FAISSVectorStore:
    """FAISS向量存储"""
    
    def __init__(self, dimension: int, index_type: str = "HNSW32,Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.index_path = None
        self.metadata_path = None
        
    def create_index(self, ef_construction: int = 200):
        """创建FAISS索引"""
        try:
            logger.info(f"创建FAISS索引: {self.index_type}, 维度: {self.dimension}")
            
            if self.index_type.startswith("HNSW"):
                # HNSW索引配置
                quantizer = faiss.IndexFlatIP(self.dimension)  # 内积相似度
                self.index = faiss.IndexHNSWFlat(quantizer, self.dimension, 32)
                self.index.hnsw.efConstruction = ef_construction
                self.index.hnsw.efSearch = 64
            else:
                # 平面索引
                self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.info("FAISS索引创建完成")
            
        except Exception as e:
            logger.error(f"FAISS索引创建失败: {e}")
            raise VoiceHelperError("FAISS_INDEX_CREATE_FAILED", f"Failed to create FAISS index: {e}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """添加向量到索引"""
        if self.index is None:
            self.create_index()
        
        try:
            # 确保向量是float32类型
            vectors = vectors.astype(np.float32)
            
            # 添加到索引
            self.index.add(vectors)
            
            # 保存元数据
            self.metadata.extend(metadata)
            
            logger.info(f"添加 {len(vectors)} 个向量到索引，总数: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"向量添加失败: {e}")
            raise VoiceHelperError("FAISS_ADD_VECTORS_FAILED", f"Failed to add vectors: {e}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, ef_search: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """搜索相似向量"""
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        try:
            # 设置搜索参数
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = ef_search
            
            # 确保查询向量格式正确
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # 搜索
            scores, indices = self.index.search(query_vector, top_k)
            
            return scores[0], indices[0]
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise VoiceHelperError("FAISS_SEARCH_FAILED", f"Failed to search vectors: {e}")
    
    def save(self, index_path: str, metadata_path: str):
        """保存索引和元数据"""
        try:
            # 保存FAISS索引
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
    
    def load(self, index_path: str, metadata_path: str):
        """加载索引和元数据"""
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(index_path)
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.index_path = index_path
            self.metadata_path = metadata_path
            
            logger.info(f"索引已加载: {index_path}, 向量数: {self.index.ntotal}, 元数据数: {len(self.metadata)}")
            
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            raise VoiceHelperError("FAISS_LOAD_FAILED", f"Failed to load index: {e}")


class BGEFAISSRAGService:
    """BGE + FAISS RAG服务"""
    
    def __init__(
        self,
        llm_service: LLMService,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        index_type: str = "HNSW32,Flat",
        data_dir: str = "data"
    ):
        self.llm_service = llm_service
        self.embedding_service = BGEEmbeddingService(embedding_model)
        self.vector_store = None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.index_type = index_type
        
        # 检索配置
        self.retrieval_config = {
            "top_k": 5,
            "ef_search": 64,
            "score_threshold": 0.3,
            "query_instruction": "为这个问题检索相关文档：",
            "passage_instruction": "",
        }
        
    async def initialize(self):
        """初始化服务"""
        try:
            # 加载嵌入模型
            self.embedding_service.load_model()
            
            # 初始化向量存储
            self.vector_store = FAISSVectorStore(
                dimension=self.embedding_service.dimension,
                index_type=self.index_type
            )
            
            # 尝试加载已有索引
            await self.load_index()
            
            logger.info("BGE+FAISS RAG服务初始化完成")
            
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {e}")
            raise VoiceHelperError("RAG_INIT_FAILED", f"Failed to initialize RAG service: {e}")
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """文档入库"""
        try:
            logger.info(f"开始文档入库，文档数: {len(documents)}")
            
            # 文档分块
            chunks = []
            metadata = []
            
            for doc_idx, doc in enumerate(documents):
                doc_chunks = self._chunk_document(doc, chunk_size, chunk_overlap)
                chunks.extend([chunk['content'] for chunk in doc_chunks])
                metadata.extend([{
                    **chunk['metadata'],
                    'doc_id': doc.get('id', f'doc_{doc_idx}'),
                    'source': doc.get('source', 'unknown'),
                    'title': doc.get('title', ''),
                } for chunk in doc_chunks])
            
            logger.info(f"文档分块完成，块数: {len(chunks)}")
            
            # 批量编码
            all_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = self.embedding_service.encode(
                    batch_chunks,
                    instruction=self.retrieval_config["passage_instruction"]
                )
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            logger.info(f"文档编码完成，向量形状: {embeddings.shape}")
            
            # 添加到向量存储
            if self.vector_store.index is None:
                self.vector_store.create_index()
            
            self.vector_store.add_vectors(embeddings, metadata)
            
            # 保存索引
            await self.save_index()
            
            result = {
                "status": "success",
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "vectors_added": len(embeddings),
                "total_vectors": self.vector_store.index.ntotal
            }
            
            logger.info(f"文档入库完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"文档入库失败: {e}")
            raise VoiceHelperError("DOCUMENT_INGEST_FAILED", f"Failed to ingest documents: {e}")
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """检索相关文档"""
        try:
            if self.vector_store is None or self.vector_store.index is None:
                logger.warning("向量索引未初始化")
                return []
            
            # 使用配置的默认值
            top_k = top_k or self.retrieval_config["top_k"]
            score_threshold = score_threshold or self.retrieval_config["score_threshold"]
            
            # 编码查询
            query_embedding = self.embedding_service.encode(
                [query],
                instruction=self.retrieval_config["query_instruction"]
            )[0]
            
            # 搜索相似向量
            scores, indices = self.vector_store.search(
                query_embedding,
                top_k=top_k * 2,  # 获取更多结果用于过滤
                ef_search=self.retrieval_config["ef_search"]
            )
            
            # 构建结果
            results = []
            for score, idx in zip(scores, indices):
                if idx == -1 or score < score_threshold:
                    continue
                
                metadata = self.vector_store.metadata[idx]
                
                # 应用元数据过滤
                if filter_metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                results.append({
                    "id": metadata.get("chunk_id", f"chunk_{idx}"),
                    "content": metadata.get("content", ""),
                    "score": float(score),
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "metadata": metadata
                })
                
                if len(results) >= top_k:
                    break
            
            logger.info(f"检索完成，查询: {query[:50]}..., 结果数: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            raise VoiceHelperError("DOCUMENT_RETRIEVE_FAILED", f"Failed to retrieve documents: {e}")
    
    async def query(
        self,
        question: str,
        context_limit: int = 4000,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """RAG问答"""
        try:
            # 检索相关文档
            retrieved_docs = await self.retrieve(question)
            
            if not retrieved_docs:
                logger.warning(f"未找到相关文档: {question}")
                return {
                    "answer": "抱歉，我没有找到相关的信息来回答您的问题。",
                    "references": [],
                    "context_used": ""
                }
            
            # 构建上下文
            context_parts = []
            context_length = 0
            used_docs = []
            
            for doc in retrieved_docs:
                content = doc["content"]
                if context_length + len(content) > context_limit:
                    break
                
                context_parts.append(f"文档{len(context_parts)+1}：{content}")
                context_length += len(content)
                used_docs.append(doc)
            
            context = "\n\n".join(context_parts)
            
            # 构建提示词
            prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请说明无法回答。

文档内容：
{context}

问题：{question}

请基于上述文档内容回答问题，并保持回答的准确性和相关性。"""
            
            # 调用LLM
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = {
                "answer": response.get("content", ""),
                "references": [
                    {
                        "id": doc["id"],
                        "source": doc["source"],
                        "title": doc["title"],
                        "score": doc["score"],
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                    }
                    for doc in used_docs
                ],
                "context_used": context[:500] + "..." if len(context) > 500 else context,
                "total_references": len(retrieved_docs)
            }
            
            logger.info(f"RAG问答完成，问题: {question[:50]}..., 引用数: {len(used_docs)}")
            return result
            
        except Exception as e:
            logger.error(f"RAG问答失败: {e}")
            raise VoiceHelperError("RAG_QUERY_FAILED", f"Failed to process RAG query: {e}")
    
    def _chunk_document(
        self,
        document: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """文档分块"""
        content = document.get("content", "")
        if not content:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk_content = content[start:end]
            
            # 尝试在句号处断开
            if end < len(content):
                last_period = chunk_content.rfind('。')
                if last_period > chunk_size * 0.7:  # 至少保留70%的内容
                    end = start + last_period + 1
                    chunk_content = content[start:end]
            
            chunks.append({
                "content": chunk_content.strip(),
                "metadata": {
                    "chunk_id": f"{document.get('id', 'doc')}_{chunk_id}",
                    "chunk_index": chunk_id,
                    "start_pos": start,
                    "end_pos": end,
                    "length": len(chunk_content)
                }
            })
            
            start = end - chunk_overlap
            chunk_id += 1
        
        return chunks
    
    async def save_index(self, dataset_id: str = "default"):
        """保存索引"""
        if self.vector_store is None:
            return
        
        index_path = self.data_dir / f"{dataset_id}.faiss"
        metadata_path = self.data_dir / f"{dataset_id}_meta.json"
        
        self.vector_store.save(str(index_path), str(metadata_path))
    
    async def load_index(self, dataset_id: str = "default"):
        """加载索引"""
        index_path = self.data_dir / f"{dataset_id}.faiss"
        metadata_path = self.data_dir / f"{dataset_id}_meta.json"
        
        if index_path.exists() and metadata_path.exists():
            if self.vector_store is None:
                self.vector_store = FAISSVectorStore(
                    dimension=self.embedding_service.dimension,
                    index_type=self.index_type
                )
            
            self.vector_store.load(str(index_path), str(metadata_path))
            logger.info(f"索引加载成功: {dataset_id}")
        else:
            logger.info(f"索引文件不存在，将创建新索引: {dataset_id}")
    
    async def reload_index(self, dataset_id: str = "default"):
        """热重载索引"""
        try:
            await self.load_index(dataset_id)
            logger.info(f"索引热重载完成: {dataset_id}")
            return {"status": "success", "dataset_id": dataset_id}
        except Exception as e:
            logger.error(f"索引热重载失败: {e}")
            raise VoiceHelperError("INDEX_RELOAD_FAILED", f"Failed to reload index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.vector_store is None or self.vector_store.index is None:
            return {
                "total_vectors": 0,
                "index_type": self.index_type,
                "dimension": self.embedding_service.dimension,
                "model_name": self.embedding_service.model_name
            }
        
        return {
            "total_vectors": self.vector_store.index.ntotal,
            "index_type": self.index_type,
            "dimension": self.embedding_service.dimension,
            "model_name": self.embedding_service.model_name,
            "metadata_count": len(self.vector_store.metadata)
        }
