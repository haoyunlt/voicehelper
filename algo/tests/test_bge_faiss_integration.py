"""
BGE + FAISS 集成测试
测试完整的BGE+FAISS RAG流程
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

from core.rag import BgeEmbedder, FaissRetriever, DocumentChunker, FaissIndexBuilder
from core.rag_factory import RAGServiceFactory
from core.config.bge_config import RAGConfig, BGEConfig, FAISSConfig, DocumentConfig
from core.rag.metrics import get_metrics_collector


@pytest.fixture
def sample_documents():
    """示例文档"""
    return [
        {
            "id": "doc_1",
            "title": "Python编程基础",
            "content": "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。它广泛应用于Web开发、数据科学、人工智能等领域。",
            "source": "python_guide.pdf",
            "metadata": {"category": "programming", "difficulty": "beginner"}
        },
        {
            "id": "doc_2",
            "title": "机器学习入门",
            "content": "机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习模式。常见的机器学习算法包括线性回归、决策树、神经网络等。",
            "source": "ml_intro.pdf",
            "metadata": {"category": "ai", "difficulty": "intermediate"}
        },
        {
            "id": "doc_3",
            "title": "数据结构与算法",
            "content": "数据结构是计算机存储、组织数据的方式。常见的数据结构包括数组、链表、栈、队列、树、图等。算法是解决问题的步骤和方法。",
            "source": "ds_algo.pdf",
            "metadata": {"category": "computer_science", "difficulty": "intermediate"}
        }
    ]


@pytest.fixture
def temp_config():
    """临时配置"""
    temp_dir = tempfile.mkdtemp()
    
    config = RAGConfig(
        bge=BGEConfig(
            model_name="BAAI/bge-small-zh-v1.5",  # 使用小模型加快测试
            device="cpu",  # 使用CPU避免GPU依赖
            batch_size=2
        ),
        faiss=FAISSConfig(
            index_type="Flat",  # 使用简单索引
            data_dir=temp_dir,
            tenant_based=False
        ),
        document=DocumentConfig(
            chunk_size=100,
            chunk_overlap=20
        )
    )
    
    yield config
    
    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBGEEmbedder:
    """BGE嵌入器测试"""
    
    def test_embedder_creation(self):
        """测试嵌入器创建"""
        embedder = BgeEmbedder(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"
        )
        assert embedder.model_name == "BAAI/bge-small-zh-v1.5"
        assert embedder.device == "cpu"
    
    def test_embed_queries(self):
        """测试查询嵌入"""
        embedder = BgeEmbedder(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"
        )
        
        queries = ["什么是Python？", "机器学习如何工作？"]
        embeddings = embedder.embed_queries(queries)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == embedder.dimension
        assert embeddings.dtype.name == 'float32'
    
    def test_embed_passages(self):
        """测试文档嵌入"""
        embedder = BgeEmbedder(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"
        )
        
        passages = ["Python是一种编程语言", "机器学习是AI的分支"]
        embeddings = embedder.embed_passages(passages)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == embedder.dimension
        assert embeddings.dtype.name == 'float32'


class TestDocumentChunker:
    """文档分块器测试"""
    
    def test_chunk_document(self, sample_documents):
        """测试文档分块"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        chunks = chunker.chunk_document(sample_documents[0])
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)
    
    def test_chunk_documents(self, sample_documents):
        """测试批量文档分块"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        all_chunks = chunker.chunk_documents(sample_documents)
        
        assert len(all_chunks) > len(sample_documents)  # 应该产生更多分块
        assert all("text" in chunk for chunk in all_chunks)


class TestFaissRetriever:
    """FAISS检索器测试"""
    
    @pytest.mark.asyncio
    async def test_retriever_workflow(self, sample_documents, temp_config):
        """测试完整的检索工作流"""
        # 创建嵌入器
        embedder = BgeEmbedder(
            model_name=temp_config.bge.model_name,
            device=temp_config.bge.device
        )
        
        # 创建检索器
        retriever = FaissRetriever(
            embedder=embedder,
            index_type=temp_config.faiss.index_type
        )
        
        # 添加文档
        chunker = DocumentChunker(
            chunk_size=temp_config.document.chunk_size,
            chunk_overlap=temp_config.document.chunk_overlap
        )
        chunks = chunker.chunk_documents(sample_documents)
        
        result = retriever.add_documents(chunks)
        assert result["added"] > 0
        assert result["total"] > 0
        
        # 执行检索
        query = "什么是Python编程？"
        results = retriever.retrieve(query, top_k=3)
        
        assert len(results) > 0
        assert all("score" in result for result in results)
        assert all("content" in result for result in results)
        
        # 检查结果是否按分数排序
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)


class TestRAGServiceFactory:
    """RAG服务工厂测试"""
    
    @pytest.mark.asyncio
    async def test_factory_workflow(self, sample_documents, temp_config):
        """测试工厂完整工作流"""
        factory = RAGServiceFactory(config=temp_config)
        
        # 构建索引
        result = await factory.build_index_from_documents(
            documents=sample_documents,
            tenant_id="test_tenant",
            dataset_id="test_dataset"
        )
        
        assert result["status"] == "success"
        assert result["total_documents"] == len(sample_documents)
        assert result["chunks_created"] > 0
        
        # 创建检索器
        retriever = factory.create_retriever(
            tenant_id="test_tenant",
            dataset_id="test_dataset"
        )
        
        # 执行检索
        query = "机器学习算法有哪些？"
        results = retriever.retrieve(query, top_k=2)
        
        assert len(results) > 0
        
        # 获取统计信息
        stats = factory.get_retriever_stats(
            tenant_id="test_tenant",
            dataset_id="test_dataset"
        )
        
        assert "total_vectors" in stats
        assert stats["total_vectors"] > 0


class TestMetrics:
    """指标测试"""
    
    def test_metrics_collection(self):
        """测试指标收集"""
        collector = get_metrics_collector()
        
        # 记录检索指标
        collector.record_retrieval(
            query="测试查询",
            retrieval_time_ms=50.0,
            results=[{"score": 0.8}, {"score": 0.6}],
            retriever_type="faiss"
        )
        
        # 获取统计
        stats = collector.get_retrieval_stats(window_minutes=1)
        
        assert stats["total_queries"] >= 1
        assert stats["avg_latency_ms"] > 0
        assert stats["avg_results_count"] > 0
    
    def test_index_metrics(self):
        """测试索引指标"""
        collector = get_metrics_collector()
        
        # 记录索引加载指标
        collector.record_index_load(
            load_time_ms=100.0,
            index_size_mb=5.0,
            vector_count=100,
            dimension=512,
            index_type="HNSW32,Flat"
        )
        
        # 获取统计
        stats = collector.get_index_stats()
        
        assert stats["latest_load_time_ms"] == 100.0
        assert stats["vector_count"] == 100
        assert stats["dimension"] == 512


@pytest.mark.integration
class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_rag_pipeline(self, sample_documents, temp_config):
        """测试完整的RAG管道"""
        factory = RAGServiceFactory(config=temp_config)
        
        # 1. 构建索引
        build_result = await factory.build_index_from_documents(
            documents=sample_documents,
            tenant_id="integration_test",
            dataset_id="complete_pipeline"
        )
        
        assert build_result["status"] == "success"
        
        # 2. 创建检索器
        retriever = factory.create_retriever(
            tenant_id="integration_test",
            dataset_id="complete_pipeline"
        )
        
        # 3. 执行多个查询
        test_queries = [
            "Python编程语言的特点是什么？",
            "机器学习有哪些算法？",
            "数据结构包括哪些类型？"
        ]
        
        all_results = []
        for query in test_queries:
            results = retriever.retrieve(query, top_k=2)
            all_results.extend(results)
            
            # 验证结果质量
            assert len(results) > 0
            assert all(result["score"] > 0 for result in results)
        
        # 4. 检查指标
        metrics_collector = get_metrics_collector()
        retrieval_stats = metrics_collector.get_retrieval_stats(window_minutes=1)
        
        assert retrieval_stats["total_queries"] >= len(test_queries)
        assert retrieval_stats["avg_latency_ms"] > 0
        
        # 5. 验证检索器统计
        retriever_stats = factory.get_retriever_stats(
            tenant_id="integration_test",
            dataset_id="complete_pipeline"
        )
        
        assert retriever_stats["total_vectors"] > 0
        assert retriever_stats["index_type"] == temp_config.faiss.index_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
