import asyncio
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    rank: int

class EnhancedFAISSRAG:
    """增强版FAISS RAG系统"""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        index_type: str = "HNSW",
        dimension: int = 1024,
        data_dir: str = "data/rag"
    ):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.index_type = index_type
        self.dimension = dimension
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS索引
        self.index = None
        self.documents: List[Document] = []
        self.id_to_index: Dict[str, int] = {}
        
        # 线程池用于并行处理
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 缓存
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.query_cache: Dict[str, List[RetrievalResult]] = {}
        
    async def initialize(self):
        """初始化RAG系统"""
        try:
            # 加载嵌入模型
            logger.info(f"加载嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # 尝试加载已有索引
            await self.load_index()
            
            logger.info("FAISS RAG系统初始化完成")
            
        except Exception as e:
            logger.error(f"RAG系统初始化失败: {e}")
            raise
    
    def _create_faiss_index(self) -> faiss.Index:
        """创建FAISS索引"""
        if self.index_type == "HNSW":
            # HNSW索引，适合高维向量和大规模数据
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        elif self.index_type == "IVF":
            # IVF索引，适合超大规模数据
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            # 默认使用Flat索引
            index = faiss.IndexFlatL2(self.dimension)
            
        return index
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """添加文档到索引"""
        try:
            start_time = time.time()
            
            # 准备文档对象
            doc_objects = []
            texts_to_embed = []
            
            for doc_data in documents:
                doc = Document(
                    id=doc_data.get("id", f"doc_{len(self.documents)}"),
                    content=doc_data.get("content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                doc_objects.append(doc)
                texts_to_embed.append(doc.content)
            
            # 批量生成嵌入
            logger.info(f"为{len(texts_to_embed)}个文档生成嵌入")
            embeddings = await self._generate_embeddings_batch(texts_to_embed)
            
            # 创建或更新索引
            if self.index is None:
                self.index = self._create_faiss_index()
            
            # 添加向量到索引
            embeddings_array = np.array(embeddings).astype('float32')
            start_index = len(self.documents)
            
            if hasattr(self.index, 'train') and not self.index.is_trained:
                self.index.train(embeddings_array)
            
            self.index.add(embeddings_array)
            
            # 更新文档存储
            for i, (doc, embedding) in enumerate(zip(doc_objects, embeddings)):
                doc.embedding = embedding
                self.documents.append(doc)
                self.id_to_index[doc.id] = start_index + i
            
            # 保存索引
            await self.save_index()
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "documents_added": len(documents),
                "total_documents": len(self.documents),
                "processing_time": processing_time,
                "index_size": self.index.ntotal if self.index else 0
            }
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "documents_added": 0
            }
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """搜索相关文档"""
        try:
            # 检查缓存
            cache_key = f"{query}:{top_k}:{score_threshold}"
            if cache_key in self.query_cache:
                logger.debug("使用缓存结果")
                return self.query_cache[cache_key]
            
            if self.index is None or len(self.documents) == 0:
                logger.warning("索引为空或未初始化")
                return []
            
            # 生成查询嵌入
            query_embedding = await self._generate_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # 执行搜索
            search_k = min(top_k * 2, len(self.documents))  # 搜索更多结果用于过滤
            scores, indices = self.index.search(query_vector, search_k)
            
            # 构建结果
            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                    
                if score < score_threshold:
                    continue
                
                if idx >= len(self.documents):
                    logger.warning(f"索引超出范围: {idx}")
                    continue
                
                document = self.documents[idx]
                
                # 应用过滤器
                if filters and not self._apply_filters(document, filters):
                    continue
                
                results.append(RetrievalResult(
                    document=document,
                    score=float(score),
                    rank=rank
                ))
                
                if len(results) >= top_k:
                    break
            
            # 缓存结果
            self.query_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """混合搜索：语义搜索 + 关键词搜索"""
        try:
            # 语义搜索
            semantic_results = await self.search(query, top_k * 2)
            
            # 关键词搜索（简单实现）
            keyword_results = await self._keyword_search(query, top_k * 2)
            
            # 结果融合
            combined_results = self._combine_search_results(
                semantic_results, 
                keyword_results, 
                semantic_weight, 
                keyword_weight
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return await self.search(query, top_k)  # 降级到语义搜索
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """生成单个文本的嵌入"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            self.embedding_model.encode,
            text
        )
        
        self.embedding_cache[text] = embedding
        return embedding
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量生成嵌入"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self.embedding_model.encode,
            texts
        )
        return embeddings.tolist()
    
    async def _keyword_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """关键词搜索（简单TF-IDF实现）"""
        # 这里可以集成更复杂的关键词搜索算法，如BM25
        query_terms = query.lower().split()
        results = []
        
        for idx, doc in enumerate(self.documents):
            content_lower = doc.content.lower()
            score = sum(content_lower.count(term) for term in query_terms)
            
            if score > 0:
                results.append(RetrievalResult(
                    document=doc,
                    score=float(score),
                    rank=idx
                ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _combine_search_results(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[RetrievalResult]:
        """合并搜索结果"""
        # 创建文档ID到结果的映射
        semantic_map = {result.document.id: result for result in semantic_results}
        keyword_map = {result.document.id: result for result in keyword_results}
        
        # 获取所有文档ID
        all_doc_ids = set(semantic_map.keys()) | set(keyword_map.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            semantic_score = semantic_map.get(doc_id, RetrievalResult(None, 0.0, 999)).score
            keyword_score = keyword_map.get(doc_id, RetrievalResult(None, 0.0, 999)).score
            
            # 归一化分数
            if semantic_results:
                semantic_score = semantic_score / max(1.0, max(r.score for r in semantic_results))
            if keyword_results:
                keyword_score = keyword_score / max(1.0, max(r.score for r in keyword_results))
            
            # 计算组合分数
            combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score
            
            # 获取文档对象
            document = semantic_map.get(doc_id, keyword_map.get(doc_id)).document
            
            combined_results.append(RetrievalResult(
                document=document,
                score=combined_score,
                rank=0  # 重新排序后设置
            ))
        
        # 按组合分数排序
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # 更新排名
        for rank, result in enumerate(combined_results):
            result.rank = rank
        
        return combined_results
    
    def _apply_filters(self, document: Document, filters: Dict[str, Any]) -> bool:
        """应用过滤器"""
        for key, value in filters.items():
            if key not in document.metadata:
                return False
            if document.metadata[key] != value:
                return False
        return True
    
    async def save_index(self):
        """保存索引和文档"""
        try:
            if self.index is not None:
                # 保存FAISS索引
                index_path = self.data_dir / "faiss.index"
                faiss.write_index(self.index, str(index_path))
                
                # 保存文档数据
                docs_path = self.data_dir / "documents.pkl"
                with open(docs_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'id_to_index': self.id_to_index
                    }, f)
                
                # 保存元数据
                metadata_path = self.data_dir / "metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'embedding_model': self.embedding_model_name,
                        'index_type': self.index_type,
                        'dimension': self.dimension,
                        'total_documents': len(self.documents),
                        'created_at': time.time()
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"索引已保存到 {self.data_dir}")
                
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    async def load_index(self):
        """加载索引和文档"""
        try:
            index_path = self.data_dir / "faiss.index"
            docs_path = self.data_dir / "documents.pkl"
            metadata_path = self.data_dir / "metadata.json"
            
            if not all(p.exists() for p in [index_path, docs_path, metadata_path]):
                logger.info("未找到已有索引，将创建新索引")
                return
            
            # 加载FAISS索引
            self.index = faiss.read_index(str(index_path))
            
            # 加载文档数据
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.id_to_index = data['id_to_index']
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                logger.info(f"加载索引: {metadata['total_documents']} 个文档")
            
            logger.info(f"索引已从 {self.data_dir} 加载")
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            self.index = None
            self.documents = []
            self.id_to_index = {}
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取RAG系统统计信息"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "cache_size": len(self.embedding_cache),
            "query_cache_size": len(self.query_cache)
        }

# 使用示例
async def main():
    # 初始化RAG系统
    rag = EnhancedFAISSRAG(
        embedding_model="BAAI/bge-large-zh-v1.5",
        index_type="HNSW"
    )
    
    await rag.initialize()
    
    # 添加文档
    documents = [
        {
            "id": "doc1",
            "content": "VoiceHelper是一个智能语音助手平台",
            "metadata": {"category": "product", "language": "zh"}
        },
        {
            "id": "doc2", 
            "content": "支持实时语音识别和语音合成功能",
            "metadata": {"category": "feature", "language": "zh"}
        }
    ]
    
    result = await rag.add_documents(documents)
    print(f"添加文档结果: {result}")
    
    # 搜索
    search_results = await rag.search("语音助手功能", top_k=3)
    for result in search_results:
        print(f"文档: {result.document.content}")
        print(f"分数: {result.score}")
        print(f"排名: {result.rank}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main())
