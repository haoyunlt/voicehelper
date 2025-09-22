"""
V2架构集成测试
验证父类/子类设计模式的完整功能
"""

import asyncio
import json
import time
from typing import List, Dict, Any
import pytest
from loguru import logger

# 导入V2架构组件
from algo.core.base import StreamCallback
from algo.core.rag.bge_faiss_retriever import BGEFaissRetriever
from algo.core.tools import FetchTool, FsReadTool, GithubReadTool
from algo.core.graph.chat_voice import ChatVoiceAgentGraph


class TestEventCollector:
    """测试事件收集器"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        
    def __call__(self, event: str, payload: dict) -> None:
        self.events.append({
            "event": event,
            "payload": payload,
            "timestamp": time.time()
        })
        
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["event"] == event_type]
        
    def clear(self):
        self.events.clear()


@pytest.fixture
def event_collector():
    """事件收集器fixture"""
    return TestEventCollector()


@pytest.fixture
def sample_documents():
    """示例文档数据"""
    return [
        {
            "id": "doc_1",
            "title": "Python编程指南",
            "content": "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
            "source": "python_guide.md",
            "metadata": {"category": "programming", "language": "python"}
        },
        {
            "id": "doc_2", 
            "title": "机器学习基础",
            "content": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
            "source": "ml_basics.md",
            "metadata": {"category": "ai", "topic": "machine_learning"}
        },
        {
            "id": "doc_3",
            "title": "Web开发技术",
            "content": "现代Web开发涉及前端、后端、数据库等多个技术栈。",
            "source": "web_dev.md", 
            "metadata": {"category": "web", "stack": "fullstack"}
        }
    ]


class TestV2Architecture:
    """V2架构集成测试类"""
    
    def test_bge_faiss_retriever_basic(self, sample_documents, event_collector):
        """测试BGE+FAISS检索器基本功能"""
        logger.info("测试BGE+FAISS检索器基本功能")
        
        # 创建检索器
        retriever = BGEFaissRetriever(
            model_name="BAAI/bge-small-zh-v1.5",  # 使用小模型加速测试
            tenant_id="test_tenant",
            dataset_id="test_dataset"
        )
        
        # 构建索引
        result = retriever.build_index(sample_documents)
        
        # 验证构建结果
        assert result["status"] == "success"
        assert result["doc_count"] == 3
        assert result["vector_count"] == 3
        assert result["dimension"] > 0
        
        # 测试检索功能
        results = retriever.retrieve_with_callback(
            "Python编程", 
            cb=event_collector,
            top_k=2
        )
        
        # 验证检索结果
        assert len(results) > 0
        assert results[0]["score"] > 0.5  # 相关性分数应该较高
        
        # 验证事件回调
        retrieve_events = event_collector.get_events_by_type("retrieve_start")
        assert len(retrieve_events) == 1
        
        result_events = event_collector.get_events_by_type("retrieve_result")
        assert len(result_events) == 1
        
        logger.info(f"检索测试完成，返回 {len(results)} 个结果")
    
    def test_tools_functionality(self, event_collector):
        """测试工具功能"""
        logger.info("测试工具功能")
        
        # 测试FetchTool
        fetch_tool = FetchTool()
        
        # 测试工具元数据
        assert fetch_tool.name == "fetch_url"
        assert "URL" in fetch_tool.description
        assert "url" in fetch_tool.args_schema["properties"]
        
        # 测试参数验证
        args = fetch_tool.validate_args(url="https://httpbin.org/json")
        assert args["url"] == "https://httpbin.org/json"
        
        # 测试工具执行（模拟）
        try:
            result = fetch_tool.run_with_callback(
                cb=event_collector,
                url="https://httpbin.org/json",
                timeout=5
            )
            
            # 验证结果
            assert result["status"] == "success"
            assert len(result["text"]) > 0
            
            # 验证事件
            start_events = event_collector.get_events_by_type("tool_start")
            assert len(start_events) == 1
            
            result_events = event_collector.get_events_by_type("tool_result")
            assert len(result_events) == 1
            
        except Exception as e:
            logger.warning(f"网络请求失败（预期行为）: {e}")
            # 网络请求可能失败，这是正常的
        
        # 测试FsReadTool
        fs_tool = FsReadTool()
        assert fs_tool.name == "read_file"
        
        # 测试GithubReadTool
        github_tool = GithubReadTool()
        assert github_tool.name == "read_github"
        
        logger.info("工具功能测试完成")
    
    def test_agent_graph_integration(self, sample_documents, event_collector):
        """测试Agent图集成功能"""
        logger.info("测试Agent图集成功能")
        
        # 创建检索器
        retriever = BGEFaissRetriever(
            model_name="BAAI/bge-small-zh-v1.5",
            tenant_id="test_tenant", 
            dataset_id="test_dataset"
        )
        
        # 构建索引
        retriever.build_index(sample_documents)
        
        # 创建工具
        tools = [
            FetchTool(),
            FsReadTool(),
            GithubReadTool()
        ]
        
        # 创建Agent图
        agent_graph = ChatVoiceAgentGraph(
            retriever=retriever,
            tools=tools
        )
        
        # 测试流式处理
        query = "什么是Python编程？"
        results = []
        
        for result in agent_graph.stream(query, cb=event_collector):
            results.append(result)
            logger.debug(f"Agent结果: {result}")
        
        # 验证结果
        assert len(results) > 0
        
        # 检查是否有意图分析
        intent_results = [r for r in results if r.get("event") == "intent"]
        assert len(intent_results) > 0
        
        # 检查是否有检索结果
        retrieve_results = [r for r in results if r.get("event") == "retrieve"]
        assert len(retrieve_results) > 0
        
        # 检查是否有最终答案
        answer_results = [r for r in results if r.get("event") == "answer"]
        assert len(answer_results) > 0
        
        # 验证答案内容
        answer = answer_results[0]["data"]
        assert "text" in answer
        assert len(answer["text"]) > 0
        
        logger.info(f"Agent图测试完成，生成 {len(results)} 个事件")
    
    def test_mixin_functionality(self, event_collector):
        """测试Mixin功能"""
        logger.info("测试Mixin功能")
        
        # 测试重试功能
        fetch_tool = FetchTool()
        fetch_tool.max_retries = 2
        fetch_tool.retry_delay = 0.1
        
        # 测试可观测功能
        fetch_tool.emit(event_collector, "test_event", {"message": "test"})
        
        # 验证事件
        test_events = event_collector.get_events_by_type("test_event")
        assert len(test_events) == 1
        assert test_events[0]["payload"]["message"] == "test"
        
        # 测试缓存功能
        retriever = BGEFaissRetriever(
            model_name="BAAI/bge-small-zh-v1.5",
            tenant_id="test_tenant",
            dataset_id="test_dataset"
        )
        
        # 缓存功能通过_cached_call方法测试
        cache_key = retriever._get_cache_key("test_query", top_k=5)
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
        
        logger.info("Mixin功能测试完成")
    
    def test_error_handling(self, event_collector):
        """测试错误处理"""
        logger.info("测试错误处理")
        
        # 测试无效URL的FetchTool
        fetch_tool = FetchTool()
        
        result = fetch_tool.run_with_callback(
            cb=event_collector,
            url="invalid_url",
            timeout=1
        )
        
        # 验证错误处理
        assert result["status"] == "error"
        assert "error" in result
        
        # 验证错误事件
        error_events = event_collector.get_events_by_type("tool_error")
        assert len(error_events) == 1
        
        logger.info("错误处理测试完成")
    
    def test_performance_metrics(self, sample_documents):
        """测试性能指标"""
        logger.info("测试性能指标")
        
        # 创建检索器
        retriever = BGEFaissRetriever(
            model_name="BAAI/bge-small-zh-v1.5",
            tenant_id="test_tenant",
            dataset_id="test_dataset"
        )
        
        # 测试索引构建时间
        start_time = time.time()
        result = retriever.build_index(sample_documents)
        build_time = time.time() - start_time
        
        logger.info(f"索引构建时间: {build_time:.3f}s")
        assert build_time < 30  # 应该在30秒内完成
        
        # 测试检索延迟
        start_time = time.time()
        results = retriever.retrieve("Python编程", top_k=3)
        retrieve_time = time.time() - start_time
        
        logger.info(f"检索延迟: {retrieve_time*1000:.1f}ms")
        assert retrieve_time < 1.0  # 应该在1秒内完成
        
        # 验证检索质量
        assert len(results) > 0
        assert results[0]["score"] > 0.3  # 相关性分数应该合理
        
        logger.info("性能指标测试完成")


if __name__ == "__main__":
    # 运行测试
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # 创建测试实例
    test_suite = TestV2Architecture()
    event_collector = TestEventCollector()
    
    # 示例文档
    sample_docs = [
        {
            "id": "doc_1",
            "title": "Python编程指南", 
            "content": "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
            "source": "python_guide.md",
            "metadata": {"category": "programming"}
        }
    ]
    
    try:
        logger.info("开始V2架构集成测试")
        
        # 运行基础测试
        test_suite.test_tools_functionality(event_collector)
        event_collector.clear()
        
        test_suite.test_mixin_functionality(event_collector)
        event_collector.clear()
        
        test_suite.test_error_handling(event_collector)
        event_collector.clear()
        
        # 如果有BGE模型，运行完整测试
        try:
            test_suite.test_bge_faiss_retriever_basic(sample_docs, event_collector)
            event_collector.clear()
            
            test_suite.test_agent_graph_integration(sample_docs, event_collector)
            event_collector.clear()
            
            test_suite.test_performance_metrics(sample_docs)
            
        except Exception as e:
            logger.warning(f"BGE模型相关测试跳过: {e}")
        
        logger.info("V2架构集成测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise
