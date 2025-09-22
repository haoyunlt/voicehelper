"""
算法服务单元测试
测试覆盖：RAG引擎、语音处理、多模态融合、批处理服务
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import numpy as np

# 导入被测试的模块
from algo.core.retrieve import RetrieveService
from algo.core.voice import VoiceService
from algo.core.multimodal_fusion import MultimodalFusionEngine
from algo.services.batch_service import LLMBatchService
from algo.core.models import QueryRequest, VoiceQueryRequest, IngestRequest


class TestRetrieveService:
    """RAG检索服务测试"""
    
    @pytest.fixture
    def mock_embeddings(self):
        """模拟嵌入模型"""
        embeddings = Mock()
        embeddings.embed_query.return_value = np.random.rand(768).tolist()
        return embeddings
    
    @pytest.fixture
    def mock_milvus(self):
        """模拟Milvus向量数据库"""
        milvus = Mock()
        milvus.search.return_value = [
            [
                Mock(
                    id="doc_1",
                    distance=0.85,
                    entity=Mock(
                        get=Mock(side_effect=lambda key: {
                            "content": "这是测试文档内容",
                            "source": "test_doc.pdf",
                            "metadata": {"page": 1}
                        }.get(key))
                    )
                ),
                Mock(
                    id="doc_2", 
                    distance=0.78,
                    entity=Mock(
                        get=Mock(side_effect=lambda key: {
                            "content": "另一个测试文档",
                            "source": "test_doc2.pdf",
                            "metadata": {"page": 2}
                        }.get(key))
                    )
                )
            ]
        ]
        return milvus
    
    @pytest.fixture
    def mock_reranker(self):
        """模拟重排序器"""
        reranker = Mock()
        reranker.predict.return_value = [0.9, 0.7]
        return reranker
    
    @pytest.fixture
    def retrieve_service(self, mock_embeddings, mock_milvus, mock_reranker):
        """创建检索服务实例"""
        service = RetrieveService()
        service.embeddings = mock_embeddings
        service.milvus = mock_milvus
        service.reranker = mock_reranker
        return service
    
    @pytest.mark.asyncio
    async def test_retrieve_documents_success(self, retrieve_service):
        """测试文档检索成功"""
        query = "测试查询"
        top_k = 5
        
        # 调用检索方法
        documents = await retrieve_service._retrieve_documents(query, top_k, None)
        
        # 验证结果
        assert len(documents) == 2
        assert documents[0].content == "这是测试文档内容"
        assert documents[0].source == "test_doc.pdf"
        assert documents[0].score == 0.85
    
    @pytest.mark.asyncio
    async def test_rerank_documents(self, retrieve_service):
        """测试文档重排序"""
        from algo.core.retrieve import Document
        
        documents = [
            Document(
                chunk_id="doc_1",
                content="文档1内容",
                source="doc1.pdf",
                score=0.7,
                metadata={}
            ),
            Document(
                chunk_id="doc_2",
                content="文档2内容", 
                source="doc2.pdf",
                score=0.8,
                metadata={}
            )
        ]
        
        # 调用重排序
        reranked = await retrieve_service._rerank_documents("查询", documents)
        
        # 验证重排序结果
        assert len(reranked) == 2
        assert reranked[0].score == 0.9  # 重排序后的分数
        assert reranked[1].score == 0.7
    
    @pytest.mark.asyncio
    async def test_stream_query_success(self, retrieve_service):
        """测试流式查询成功"""
        # 模拟LLM客户端
        mock_llm_responses = [
            '{"type": "refs", "refs": [{"source": "test.pdf", "content": "测试内容"}]}',
            '{"type": "message", "content": "这是"}',
            '{"type": "message", "content": "测试回答"}',
            '{"type": "end"}'
        ]
        
        with patch.object(retrieve_service, '_stream_llm_response') as mock_stream:
            mock_stream.return_value = iter(mock_llm_responses)
            
            request = QueryRequest(
                messages=[{"role": "user", "content": "测试问题"}],
                top_k=5,
                temperature=0.7
            )
            
            # 收集流式响应
            responses = []
            async for response in retrieve_service.stream_query(request):
                responses.append(response)
            
            # 验证响应
            assert len(responses) >= 3
            assert any("refs" in resp for resp in responses)
            assert any("message" in resp for resp in responses)
    
    def test_build_prompt(self, retrieve_service):
        """测试提示词构建"""
        messages = [
            {"role": "user", "content": "用户问题"}
        ]
        
        from algo.core.retrieve import Document
        references = [
            Document(
                chunk_id="ref_1",
                content="参考文档内容",
                source="ref.pdf",
                score=0.9,
                metadata={"page": 1}
            )
        ]
        
        # 构建提示词
        prompt = retrieve_service._build_prompt(messages, references)
        
        # 验证提示词结构
        assert isinstance(prompt, list)
        assert len(prompt) >= 2  # 至少包含系统消息和用户消息
        assert any(msg["role"] == "system" for msg in prompt)
        assert any(msg["role"] == "user" for msg in prompt)


class TestVoiceService:
    """语音服务测试"""
    
    @pytest.fixture
    def mock_asr_model(self):
        """模拟ASR模型"""
        asr = Mock()
        asr.transcribe.return_value = Mock(
            text="这是语音转文字结果",
            confidence=0.95
        )
        return asr
    
    @pytest.fixture
    def mock_tts_model(self):
        """模拟TTS模型"""
        tts = Mock()
        tts.synthesize.return_value = b"fake_audio_data"
        return tts
    
    @pytest.fixture
    def mock_emotion_analyzer(self):
        """模拟情感分析器"""
        analyzer = Mock()
        analyzer.analyze.return_value = Mock(
            primary_emotion="happy",
            confidence=0.85,
            all_emotions={
                "happy": 0.85,
                "neutral": 0.10,
                "sad": 0.05
            }
        )
        return analyzer
    
    @pytest.fixture
    def voice_service(self, mock_asr_model, mock_tts_model, mock_emotion_analyzer):
        """创建语音服务实例"""
        retrieve_service = Mock()
        service = VoiceService(retrieve_service)
        service.asr_model = mock_asr_model
        service.tts_model = mock_tts_model
        service.emotion_analyzer = mock_emotion_analyzer
        return service
    
    @pytest.mark.asyncio
    async def test_process_voice_input_success(self, voice_service):
        """测试语音输入处理成功"""
        audio_data = b"fake_audio_data"
        
        # 调用语音处理
        result = await voice_service.process_voice_input(audio_data)
        
        # 验证结果
        assert result.transcript == "这是语音转文字结果"
        assert result.confidence == 0.95
        assert result.emotion.primary_emotion == "happy"
        assert result.emotion.confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_success(self, voice_service):
        """测试语音合成成功"""
        text = "要合成的文本"
        voice_config = Mock(
            voice_id="zh-CN-XiaoxiaoNeural",
            speed=1.0,
            pitch=0.0,
            emotion="neutral"
        )
        
        # 调用语音合成
        audio_data = await voice_service.synthesize_speech(text, voice_config)
        
        # 验证结果
        assert audio_data == b"fake_audio_data"
        voice_service.tts_model.synthesize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_voice_query_complete_flow(self, voice_service):
        """测试完整语音查询流程"""
        request = VoiceQueryRequest(
            audio_data=b"fake_audio",
            session_id="session_123",
            language="zh-CN"
        )
        
        # 模拟RAG查询响应
        voice_service.retrieve_service.stream_query.return_value = iter([
            '{"type": "message", "content": "语音查询回答"}'
        ])
        
        # 收集响应
        responses = []
        async for response in voice_service.process_voice_query(request):
            responses.append(response)
        
        # 验证响应包含转录、情感和回答
        response_types = [json.loads(resp)["type"] for resp in responses if resp.startswith("{")]
        assert "transcript" in response_types
        assert "emotion" in response_types


class TestMultimodalFusionEngine:
    """多模态融合引擎测试"""
    
    @pytest.fixture
    def mock_encoders(self):
        """模拟各模态编码器"""
        text_encoder = Mock()
        text_encoder.encode.return_value = np.random.rand(512)
        
        image_encoder = Mock()
        image_encoder.encode.return_value = np.random.rand(512)
        
        audio_encoder = Mock()
        audio_encoder.encode.return_value = np.random.rand(512)
        
        return text_encoder, image_encoder, audio_encoder
    
    @pytest.fixture
    def fusion_engine(self, mock_encoders):
        """创建多模态融合引擎实例"""
        text_encoder, image_encoder, audio_encoder = mock_encoders
        
        engine = MultimodalFusionEngine()
        engine.text_encoder = text_encoder
        engine.image_encoder = image_encoder
        engine.audio_encoder = audio_encoder
        
        # 模拟注意力机制和融合模型
        engine.attention_mechanism = Mock()
        engine.attention_mechanism.compute_attention.return_value = np.random.rand(10, 10)
        
        engine.fusion_transformer = Mock()
        engine.fusion_transformer.fuse.return_value = np.random.rand(1024)
        
        return engine
    
    @pytest.mark.asyncio
    async def test_fuse_text_and_image(self, fusion_engine):
        """测试文本和图像融合"""
        from algo.core.multimodal_fusion import MultimodalInput
        
        inputs = MultimodalInput(
            text="这是文本输入",
            image=b"fake_image_data"
        )
        
        # 调用融合处理
        result = await fusion_engine.fuse_modalities(inputs)
        
        # 验证结果
        assert result.modalities == ["text", "image"]
        assert result.unified_representation is not None
        assert len(result.unified_representation) == 1024
        assert "text" in result.modality_weights
        assert "image" in result.modality_weights
    
    @pytest.mark.asyncio
    async def test_fuse_all_modalities(self, fusion_engine):
        """测试所有模态融合"""
        from algo.core.multimodal_fusion import MultimodalInput
        
        inputs = MultimodalInput(
            text="文本输入",
            image=b"image_data",
            audio=b"audio_data"
        )
        
        # 调用融合处理
        result = await fusion_engine.fuse_modalities(inputs)
        
        # 验证结果
        assert len(result.modalities) == 3
        assert all(modality in result.modality_weights for modality in ["text", "image", "audio"])
        assert result.confidence > 0


class TestLLMBatchService:
    """LLM批处理服务测试"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """模拟LLM客户端"""
        client = Mock()
        client.batch_chat_completion.return_value = [
            Mock(
                content="批处理回答1",
                model="test-model",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                finish_reason="stop",
                request_id="req_1",
                processing_time=0.5
            ),
            Mock(
                content="批处理回答2",
                model="test-model", 
                usage={"prompt_tokens": 15, "completion_tokens": 25},
                finish_reason="stop",
                request_id="req_2",
                processing_time=0.6
            )
        ]
        return client
    
    @pytest.fixture
    def batch_service(self, mock_llm_client):
        """创建批处理服务实例"""
        service = LLMBatchService()
        service.llm_client = mock_llm_client
        return service
    
    @pytest.mark.asyncio
    async def test_batch_processing_success(self, batch_service):
        """测试批处理成功"""
        # 准备批处理请求
        requests = [
            {
                "messages": [{"role": "user", "content": "问题1"}],
                "model": "test-model",
                "temperature": 0.7
            },
            {
                "messages": [{"role": "user", "content": "问题2"}],
                "model": "test-model",
                "temperature": 0.7
            }
        ]
        
        # 启动批处理服务
        await batch_service.start()
        
        # 提交批处理任务
        tasks = []
        for req in requests:
            task = asyncio.create_task(
                batch_service.chat_completion(**req)
            )
            tasks.append(task)
        
        # 等待结果
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == 2
        assert all(result.content.startswith("批处理回答") for result in results)
        assert all(result.processing_time > 0 for result in results)
        
        # 停止服务
        await batch_service.stop()
    
    @pytest.mark.asyncio
    async def test_batch_queue_management(self, batch_service):
        """测试批处理队列管理"""
        await batch_service.start()
        
        # 获取初始统计
        stats = await batch_service.get_service_stats()
        initial_processed = stats.get("total_processed", 0)
        
        # 提交单个请求
        result = await batch_service.chat_completion(
            messages=[{"role": "user", "content": "测试"}],
            model="test-model"
        )
        
        # 验证统计更新
        new_stats = await batch_service.get_service_stats()
        assert new_stats["total_processed"] > initial_processed
        
        await batch_service.stop()


class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    @pytest.mark.asyncio
    async def test_retrieve_service_performance(self):
        """测试检索服务性能"""
        # 创建模拟服务
        service = RetrieveService()
        
        # 模拟依赖
        service.embeddings = Mock()
        service.embeddings.embed_query.return_value = np.random.rand(768).tolist()
        
        service.milvus = Mock()
        service.milvus.search.return_value = [[]]
        
        # 性能测试
        import time
        start_time = time.time()
        
        await service._retrieve_documents("测试查询", 5, None)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 验证性能要求（< 50ms）
        assert response_time < 0.05, f"检索响应时间 {response_time:.3f}s 超过50ms阈值"
    
    @pytest.mark.asyncio
    async def test_voice_processing_performance(self):
        """测试语音处理性能"""
        service = VoiceService(Mock())
        
        # 模拟ASR模型
        service.asr_model = Mock()
        service.asr_model.transcribe.return_value = Mock(
            text="测试转录",
            confidence=0.9
        )
        
        # 模拟情感分析器
        service.emotion_analyzer = Mock()
        service.emotion_analyzer.analyze.return_value = Mock(
            primary_emotion="neutral",
            confidence=0.8
        )
        
        # 性能测试
        import time
        start_time = time.time()
        
        await service.process_voice_input(b"fake_audio_data")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 验证性能要求（< 150ms）
        assert response_time < 0.15, f"语音处理响应时间 {response_time:.3f}s 超过150ms阈值"


# 测试配置和工具
@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        "performance_thresholds": {
            "retrieve_response_time": 0.05,  # 50ms
            "voice_processing_time": 0.15,   # 150ms
            "batch_processing_time": 1.0,    # 1s
            "multimodal_fusion_time": 0.2    # 200ms
        },
        "test_data": {
            "sample_audio": b"fake_audio_data_16khz_pcm",
            "sample_image": b"fake_image_data_jpg",
            "sample_text": "这是测试文本内容"
        }
    }


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short", "-x"])
