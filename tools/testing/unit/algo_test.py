"""
算法服务单元测试
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from algo.core.bge_faiss_rag import BGEEmbeddingService, FAISSVectorStore, BGEFAISSRAGService
from algo.core.langgraph_agent import LangGraphAgent, AgentContext, AgentState


class TestBGEEmbeddingService:
    """BGE嵌入服务测试"""
    
    @pytest.fixture
    def embedding_service(self):
        service = BGEEmbeddingService("BAAI/bge-large-zh-v1.5")
        # Mock模型加载
        service.model = Mock()
        service.model.encode = Mock(return_value=np.random.rand(2, 1024).astype(np.float32))
        service.model.get_sentence_embedding_dimension = Mock(return_value=1024)
        service.dimension = 1024
        return service
    
    def test_encode_texts(self, embedding_service):
        """测试文本编码"""
        texts = ["这是一个测试", "这是另一个测试"]
        embeddings = embedding_service.encode(texts)
        
        assert embeddings.shape == (2, 1024)
        assert embeddings.dtype == np.float32
        embedding_service.model.encode.assert_called_once()
    
    def test_encode_with_instruction(self, embedding_service):
        """测试带指令的编码"""
        texts = ["测试文本"]
        instruction = "为这个问题检索相关文档："
        
        embedding_service.encode(texts, instruction=instruction)
        
        # 验证调用参数包含指令前缀
        call_args = embedding_service.model.encode.call_args[0][0]
        assert call_args[0] == f"{instruction}{texts[0]}"


class TestFAISSVectorStore:
    """FAISS向量存储测试"""
    
    @pytest.fixture
    def vector_store(self):
        return FAISSVectorStore(dimension=1024)
    
    def test_create_index(self, vector_store):
        """测试索引创建"""
        vector_store.create_index()
        
        assert vector_store.index is not None
        assert vector_store.index.d == 1024
    
    def test_add_vectors(self, vector_store):
        """测试向量添加"""
        vectors = np.random.rand(10, 1024).astype(np.float32)
        metadata = [{"id": i, "content": f"文档{i}"} for i in range(10)]
        
        vector_store.add_vectors(vectors, metadata)
        
        assert vector_store.index.ntotal == 10
        assert len(vector_store.metadata) == 10
    
    def test_search_vectors(self, vector_store):
        """测试向量搜索"""
        # 添加一些向量
        vectors = np.random.rand(10, 1024).astype(np.float32)
        metadata = [{"id": i, "content": f"文档{i}"} for i in range(10)]
        vector_store.add_vectors(vectors, metadata)
        
        # 搜索
        query_vector = np.random.rand(1024).astype(np.float32)
        scores, indices = vector_store.search(query_vector, top_k=5)
        
        assert len(scores) == 5
        assert len(indices) == 5
        assert all(idx < 10 for idx in indices if idx != -1)


class TestBGEFAISSRAGService:
    """BGE+FAISS RAG服务测试"""
    
    @pytest.fixture
    def rag_service(self):
        llm_service = Mock()
        llm_service.generate = AsyncMock(return_value={"content": "这是测试回答"})
        
        service = BGEFAISSRAGService(llm_service)
        
        # Mock嵌入服务
        service.embedding_service = Mock()
        service.embedding_service.dimension = 1024
        service.embedding_service.encode = Mock(
            return_value=np.random.rand(1, 1024).astype(np.float32)
        )
        
        # Mock向量存储
        service.vector_store = Mock()
        service.vector_store.search = Mock(return_value=(
            np.array([0.9, 0.8, 0.7]),
            np.array([0, 1, 2])
        ))
        service.vector_store.metadata = [
            {"chunk_id": "chunk_0", "content": "相关文档1", "source": "doc1.txt"},
            {"chunk_id": "chunk_1", "content": "相关文档2", "source": "doc2.txt"},
            {"chunk_id": "chunk_2", "content": "相关文档3", "source": "doc3.txt"},
        ]
        
        return service
    
    @pytest.mark.asyncio
    async def test_retrieve(self, rag_service):
        """测试文档检索"""
        query = "测试查询"
        results = await rag_service.retrieve(query, top_k=3)
        
        assert len(results) == 3
        assert all("content" in result for result in results)
        assert all("score" in result for result in results)
        assert results[0]["score"] >= results[1]["score"]  # 按分数排序
    
    @pytest.mark.asyncio
    async def test_query(self, rag_service):
        """测试RAG问答"""
        question = "这是一个测试问题"
        result = await rag_service.query(question)
        
        assert "answer" in result
        assert "references" in result
        assert result["answer"] == "这是测试回答"
        assert len(result["references"]) > 0
        
        # 验证LLM调用
        rag_service.llm_service.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ingest_documents(self, rag_service):
        """测试文档入库"""
        documents = [
            {
                "id": "doc1",
                "title": "测试文档1",
                "content": "这是测试文档的内容，用于验证文档入库功能。" * 20,
                "source": "test1.txt"
            }
        ]
        
        # Mock向量存储方法
        rag_service.vector_store.create_index = Mock()
        rag_service.vector_store.add_vectors = Mock()
        rag_service.vector_store.index = Mock()
        rag_service.vector_store.index.ntotal = 10
        
        # Mock嵌入服务返回
        rag_service.embedding_service.encode = Mock(
            return_value=np.random.rand(5, 1024).astype(np.float32)
        )
        
        result = await rag_service.ingest_documents(documents)
        
        assert result["status"] == "success"
        assert result["documents_processed"] == 1
        assert result["chunks_created"] > 0


class TestLangGraphAgent:
    """LangGraph Agent测试"""
    
    @pytest.fixture
    def agent(self):
        llm_service = Mock()
        llm_service.generate = AsyncMock(return_value={"content": "这是Agent的回答"})
        
        rag_service = Mock()
        rag_service.retrieve = AsyncMock(return_value=[
            {"id": "doc1", "content": "相关文档", "score": 0.9}
        ])
        
        return LangGraphAgent(
            llm_service=llm_service,
            rag_service=rag_service,
            tools=[],
            enable_planning=True,
            enable_memory=False
        )
    
    @pytest.mark.asyncio
    async def test_process_message(self, agent):
        """测试消息处理"""
        events = []
        async for event in agent.process_message(
            message="测试消息",
            conversation_id="test-conv",
            user_id="test-user"
        ):
            events.append(event)
        
        assert len(events) > 0
        assert any(event.event_type == "agent_start" for event in events)
        assert any(event.event_type == "agent_complete" for event in events)
    
    def test_agent_context_creation(self):
        """测试Agent上下文创建"""
        context = AgentContext(
            conversation_id="test-conv",
            user_id="test-user",
            session_id="test-session"
        )
        
        assert context.conversation_id == "test-conv"
        assert context.user_id == "test-user"
        assert context.session_id == "test-session"
        assert context.state == AgentState.IDLE
        assert len(context.messages) == 0


# 测试运行配置
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
