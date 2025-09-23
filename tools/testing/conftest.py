"""
pytest 全局配置文件
定义全局fixtures、测试配置和工具函数
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from typing import Dict, Any, Generator
import logging
import json
from unittest.mock import Mock, AsyncMock


# 测试环境配置
@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """测试配置"""
    return {
        "backend_url": os.getenv("TEST_BACKEND_URL", "http://localhost:8080"),
        "algo_url": os.getenv("TEST_ALGO_URL", "http://localhost:8000"),
        "frontend_url": os.getenv("TEST_FRONTEND_URL", "http://localhost:3000"),
        "websocket_url": os.getenv("TEST_WS_URL", "ws://localhost:8080"),
        "test_timeout": int(os.getenv("TEST_TIMEOUT", "30")),
        "test_data_dir": os.getenv("TEST_DATA_DIR", "tests/data"),
        "performance_thresholds": {
            "api_response_time": float(os.getenv("TEST_API_RESPONSE_TIME", "2.0")),
            "websocket_connect_time": float(os.getenv("TEST_WS_CONNECT_TIME", "5.0")),
            "success_rate": float(os.getenv("TEST_SUCCESS_RATE", "95.0")),
        }
    }


# 临时目录
@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


# 测试数据
@pytest.fixture(scope="session")
def test_data() -> Dict[str, Any]:
    """测试数据"""
    return {
        "sample_documents": [
            {
                "name": "test_doc_1.txt",
                "content": "这是第一个测试文档，包含关于VoiceHelper系统的基本信息。",
                "metadata": {"category": "system", "type": "basic"}
            },
            {
                "name": "test_doc_2.txt", 
                "content": "这是第二个测试文档，描述了系统的高级功能和特性。",
                "metadata": {"category": "features", "type": "advanced"}
            }
        ],
        "sample_queries": [
            "VoiceHelper是什么？",
            "系统有哪些功能？",
            "如何使用语音功能？",
            "技术架构如何？"
        ],
        "sample_conversations": [
            {
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "您好！我是VoiceHelper智能助手，有什么可以帮助您的吗？"}
                ]
            }
        ],
        "sample_audio": {
            "format": "pcm16",
            "sample_rate": 16000,
            "data": b"fake_audio_data" * 1000  # 模拟音频数据
        }
    }


# 模拟服务
@pytest.fixture
def mock_llm_client():
    """模拟LLM客户端"""
    client = Mock()
    client.chat_completion = AsyncMock(return_value={
        "content": "这是模拟的LLM回答",
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    })
    client.create_embeddings = AsyncMock(return_value=[[0.1] * 768])
    return client


@pytest.fixture
def mock_vector_store():
    """模拟向量数据库"""
    store = Mock()
    store.search = Mock(return_value=[
        Mock(
            id="doc_1",
            distance=0.85,
            entity=Mock(get=Mock(return_value="测试文档内容"))
        )
    ])
    store.insert = Mock(return_value=True)
    return store


@pytest.fixture
def mock_redis_client():
    """模拟Redis客户端"""
    client = Mock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    client.exists = AsyncMock(return_value=False)
    return client


# 数据库fixtures
@pytest.fixture
def mock_database():
    """模拟数据库连接"""
    db = Mock()
    db.execute = AsyncMock(return_value=Mock(fetchall=Mock(return_value=[])))
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


# HTTP客户端
@pytest.fixture
async def http_client():
    """HTTP客户端"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        yield session


# WebSocket客户端
@pytest.fixture
async def websocket_client():
    """WebSocket客户端"""
    import websockets
    
    class MockWebSocketClient:
        def __init__(self):
            self.connected = False
            self.messages = []
        
        async def connect(self, uri, **kwargs):
            self.connected = True
            return self
        
        async def send(self, message):
            self.messages.append(message)
        
        async def recv(self):
            return '{"type": "ack", "message": "test response"}'
        
        async def close(self):
            self.connected = False
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.close()
    
    return MockWebSocketClient()


# 性能监控
@pytest.fixture
def performance_monitor():
    """性能监控器"""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = []
        
        def record_metric(self, name: str, value: float, unit: str = "ms"):
            self.metrics.append({
                "name": name,
                "value": value,
                "unit": unit,
                "timestamp": asyncio.get_event_loop().time()
            })
        
        def get_metrics(self):
            return self.metrics
        
        def clear_metrics(self):
            self.metrics.clear()
    
    return PerformanceMonitor()


# 日志配置
@pytest.fixture(autouse=True)
def configure_logging():
    """配置测试日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 抑制某些库的日志
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


# 环境变量设置
@pytest.fixture(autouse=True)
def set_test_env():
    """设置测试环境变量"""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"
    yield
    # 清理
    os.environ.pop("TESTING", None)


# 异步事件循环
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# 测试标记处理
def pytest_configure(config):
    """pytest配置"""
    # 注册自定义标记
    config.addinivalue_line(
        "markers", "slow: 标记测试为慢速测试"
    )
    config.addinivalue_line(
        "markers", "external: 标记测试需要外部服务"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 为没有标记的测试添加默认标记
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# 测试报告钩子
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """生成测试报告"""
    outcome = yield
    report = outcome.get_result()
    
    # 添加额外信息到报告
    if report.when == "call":
        # 记录测试执行时间
        setattr(item, "test_duration", report.duration)
        
        # 记录测试结果
        if report.outcome == "passed":
            setattr(item, "test_result", "PASS")
        elif report.outcome == "failed":
            setattr(item, "test_result", "FAIL")
        elif report.outcome == "skipped":
            setattr(item, "test_result", "SKIP")


# 测试会话钩子
def pytest_sessionstart(session):
    """测试会话开始"""
    print("\n" + "="*60)
    print("VoiceHelper 测试套件启动")
    print("="*60)


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束"""
    print("\n" + "="*60)
    print("VoiceHelper 测试套件完成")
    print(f"退出状态: {exitstatus}")
    print("="*60)


# 工具函数
class TestUtils:
    """测试工具类"""
    
    @staticmethod
    def create_mock_response(status_code: int = 200, data: Dict = None):
        """创建模拟HTTP响应"""
        response = Mock()
        response.status_code = status_code
        response.json.return_value = data or {}
        response.text = json.dumps(data or {})
        return response
    
    @staticmethod
    def create_test_file(content: str, filename: str = "test.txt") -> str:
        """创建测试文件"""
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
        """等待条件满足"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False


@pytest.fixture
def test_utils():
    """测试工具fixture"""
    return TestUtils
