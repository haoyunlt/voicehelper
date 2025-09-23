"""
FAISS 索引构建管线
支持文档分块、批量嵌入、索引构建等功能
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

from .embedder_bge import BgeEmbedder
from .retriever_faiss import FaissRetriever
from common.errors import VoiceHelperError


class DocumentChunker:
    """文档分块器"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ['。', '！', '？', '\n\n', '\n', ' ']
    
    def chunk_document(
        self,
        document: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """分块单个文档"""
        content = document.get("content", "")
        if not content:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # 如果没有超出文档长度，尝试在分隔符处断开
            if end < len(content):
                chunk_content = content[start:end]
                
                # 寻找最佳分割点
                best_split = end
                for separator in self.separators:
                    last_sep = chunk_content.rfind(separator)
                    if last_sep > self.chunk_size * 0.7:  # 至少保留70%的内容
                        best_split = start + last_sep + len(separator)
                        break
                
                end = best_split
            
            chunk_content = content[start:end].strip()
            if not chunk_content:
                start = end
                continue
            
            chunk_data = {
                "text": chunk_content,
                "chunk_id": f"{document.get('id', 'doc')}_{chunk_id}",
                "chunk_index": chunk_id,
                "start_pos": start,
                "end_pos": end,
                "length": len(chunk_content),
                "source": document.get("source", ""),
                "title": document.get("title", ""),
                "doc_id": document.get("id", ""),
                **document.get("metadata", {})
            }
            
            chunks.append(chunk_data)
            
            # 计算下一个起始位置（考虑重叠）
            start = max(start + 1, end - self.chunk_overlap)
            chunk_id += 1
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量分块文档"""
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"文档分块失败: {doc.get('id', 'unknown')}, 错误: {e}")
                continue
        
        logger.info(f"文档分块完成，原文档数: {len(documents)}, 分块数: {len(all_chunks)}")
        return all_chunks


def build_faiss_index(
    passages: List[Dict[str, Any]],
    embedder: BgeEmbedder,
    index_out: str,
    meta_out: str,
    index_type: str = "HNSW32,Flat",
    batch_size: int = 32,
    ef_construction: int = 200
) -> Dict[str, Any]:
    """
    构建FAISS索引
    
    Args:
        passages: 文档段落列表，每个元素包含 'text' 字段
        embedder: BGE嵌入器
        index_out: 索引输出路径
        meta_out: 元数据输出路径
        index_type: FAISS索引类型
        batch_size: 批处理大小
        ef_construction: HNSW构建参数
    
    Returns:
        构建结果统计
    """
    try:
        logger.info(f"开始构建FAISS索引，文档数: {len(passages)}")
        
        # 提取文本
        texts = [p.get("text", "") for p in passages if p.get("text")]
        if not texts:
            raise VoiceHelperError("NO_VALID_TEXTS", "No valid texts found in passages")
        
        # 批量生成嵌入
        logger.info("开始生成嵌入向量...")
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedder.embed_passages(batch_texts)
            all_embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"已处理 {i + len(batch_texts)}/{len(texts)} 个文档")
        
        # 合并所有嵌入
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"嵌入生成完成，形状: {embeddings.shape}")
        
        # 创建FAISS索引
        logger.info(f"创建FAISS索引: {index_type}")
        dimension = embeddings.shape[1]
        
        if index_type.startswith("HNSW"):
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexHNSWFlat(quantizer, dimension, 32)
            index.hnsw.efConstruction = ef_construction
        else:
            index = faiss.index_factory(dimension, index_type)
        
        # 添加向量到索引
        logger.info("添加向量到索引...")
        index.add(embeddings)
        
        # 保存索引
        logger.info(f"保存索引到: {index_out}")
        Path(index_out).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, index_out)
        
        # 保存元数据
        logger.info(f"保存元数据到: {meta_out}")
        with open(meta_out, 'w', encoding='utf-8') as f:
            json.dump(passages, f, ensure_ascii=False, indent=2)
        
        result = {
            "status": "success",
            "total_documents": len(passages),
            "total_vectors": index.ntotal,
            "dimension": dimension,
            "index_type": index_type,
            "index_path": index_out,
            "metadata_path": meta_out
        }
        
        logger.info(f"FAISS索引构建完成: {result}")
        return result
        
    except Exception as e:
        logger.error(f"FAISS索引构建失败: {e}")
        raise VoiceHelperError("FAISS_INDEX_BUILD_FAILED", f"Failed to build FAISS index: {e}")


async def build_faiss_index_async(
    passages: List[Dict[str, Any]],
    embedder: BgeEmbedder,
    index_out: str,
    meta_out: str,
    **kwargs
) -> Dict[str, Any]:
    """异步构建FAISS索引"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        build_faiss_index,
        passages,
        embedder,
        index_out,
        meta_out,
        **kwargs
    )


class FaissIndexBuilder:
    """FAISS索引构建器"""
    
    def __init__(
        self,
        embedder: BgeEmbedder,
        chunker: Optional[DocumentChunker] = None,
        index_type: str = "HNSW32,Flat",
        batch_size: int = 32
    ):
        self.embedder = embedder
        self.chunker = chunker or DocumentChunker()
        self.index_type = index_type
        self.batch_size = batch_size
    
    async def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        output_dir: str,
        dataset_id: str = "default"
    ) -> Dict[str, Any]:
        """从文档构建索引"""
        try:
            # 文档分块
            logger.info("开始文档分块...")
            chunks = self.chunker.chunk_documents(documents)
            
            if not chunks:
                raise VoiceHelperError("NO_CHUNKS_GENERATED", "No chunks generated from documents")
            
            # 构建索引路径
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            index_path = str(output_path / f"{dataset_id}.faiss")
            meta_path = str(output_path / f"{dataset_id}_meta.json")
            
            # 构建索引
            result = await build_faiss_index_async(
                passages=chunks,
                embedder=self.embedder,
                index_out=index_path,
                meta_out=meta_path,
                index_type=self.index_type,
                batch_size=self.batch_size
            )
            
            result.update({
                "dataset_id": dataset_id,
                "chunks_created": len(chunks),
                "original_documents": len(documents)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"从文档构建索引失败: {e}")
            raise VoiceHelperError("BUILD_INDEX_FROM_DOCS_FAILED", f"Failed to build index from documents: {e}")
    
    def create_retriever(
        self,
        index_path: str,
        ef_search: int = 64
    ) -> FaissRetriever:
        """创建检索器并加载索引"""
        retriever = FaissRetriever(
            embedder=self.embedder,
            index_type=self.index_type,
            ef_search=ef_search
        )
        
        retriever.load_index(index_path)
        return retriever
