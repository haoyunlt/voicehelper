"""
测试框架验证
验证pytest和相关测试工具是否正常工作
"""

import pytest
import asyncio
import json
import sys
from pathlib import Path


class TestFrameworkValidation:
    """测试框架验证类"""
    
    def test_basic_assertion(self):
        """基础断言测试"""
        assert True
        assert 1 + 1 == 2
        assert "hello" in "hello world"
    
    def test_string_operations(self):
        """字符串操作测试"""
        text = "VoiceHelper"
        assert text.lower() == "voicehelper"
        assert len(text) == 11
        assert text.startswith("Voice")
    
    def test_list_operations(self):
        """列表操作测试"""
        items = [1, 2, 3, 4, 5]
        assert len(items) == 5
        assert 3 in items
        assert items[0] == 1
        assert items[-1] == 5
    
    def test_dict_operations(self):
        """字典操作测试"""
        config = {
            "service": "voicehelper",
            "version": "1.9.0",
            "features": ["chat", "voice", "rag"]
        }
        assert config["service"] == "voicehelper"
        assert "chat" in config["features"]
        assert len(config["features"]) == 3
    
    def test_json_operations(self):
        """JSON操作测试"""
        data = {"message": "hello", "status": "success"}
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["message"] == "hello"
        assert parsed["status"] == "success"
    
    def test_file_path_operations(self):
        """文件路径操作测试"""
        current_file = Path(__file__)
        assert current_file.exists()
        assert current_file.suffix == ".py"
        assert "test_framework_validation" in current_file.name
    
    def test_exception_handling(self):
        """异常处理测试"""
        with pytest.raises(ValueError):
            int("not_a_number")
        
        with pytest.raises(KeyError):
            empty_dict = {}
            _ = empty_dict["nonexistent_key"]
        
        with pytest.raises(ZeroDivisionError):
            _ = 1 / 0


class TestAsyncFramework:
    """异步测试框架验证"""
    
    @pytest.mark.asyncio
    async def test_basic_async(self):
        """基础异步测试"""
        await asyncio.sleep(0.01)  # 10ms延迟
        assert True
    
    @pytest.mark.asyncio
    async def test_async_operations(self):
        """异步操作测试"""
        async def async_add(a, b):
            await asyncio.sleep(0.01)
            return a + b
        
        result = await async_add(2, 3)
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_async_exception(self):
        """异步异常测试"""
        async def async_error():
            await asyncio.sleep(0.01)
            raise ValueError("Async error")
        
        with pytest.raises(ValueError, match="Async error"):
            await async_error()


class TestPytestFeatures:
    """Pytest功能特性测试"""
    
    @pytest.fixture
    def sample_data(self):
        """示例数据fixture"""
        return {
            "users": ["alice", "bob", "charlie"],
            "config": {"debug": True, "timeout": 30}
        }
    
    def test_fixture_usage(self, sample_data):
        """测试fixture使用"""
        assert len(sample_data["users"]) == 3
        assert sample_data["config"]["debug"] is True
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
    ])
    def test_parametrized(self, input_val, expected):
        """参数化测试"""
        assert input_val * 2 == expected
    
    @pytest.mark.skipif(sys.version_info < (3, 8), reason="需要Python 3.8+")
    def test_conditional_skip(self):
        """条件跳过测试"""
        # 这个测试只在Python 3.8+上运行
        assert sys.version_info >= (3, 8)
    
    @pytest.mark.xfail(reason="已知会失败的测试")
    def test_expected_failure(self):
        """预期失败测试"""
        assert False  # 这个测试预期会失败


def test_environment_info():
    """环境信息测试"""
    print(f"\nPython版本: {sys.version}")
    print(f"平台: {sys.platform}")
    print(f"当前工作目录: {Path.cwd()}")
    
    # 检查关键模块
    modules_to_check = [
        "pytest", "asyncio", "json", "pathlib"
    ]
    
    for module_name in modules_to_check:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name}: {getattr(module, '__version__', 'N/A')}")
        except ImportError:
            print(f"❌ {module_name}: 未安装")
    
    assert True  # 总是通过，只是为了显示信息


if __name__ == "__main__":
    # 直接运行此文件时的行为
    pytest.main([__file__, "-v", "-s"])
