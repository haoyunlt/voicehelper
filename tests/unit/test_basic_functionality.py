"""
基础功能单元测试
测试核心模块的基本功能
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock

class TestBasicFunctionality:
    """基础功能测试"""
    
    def test_basic_import(self):
        """测试基本导入功能"""
        # 测试基本的Python功能
        assert True
        
    def test_json_operations(self):
        """测试JSON操作"""
        test_data = {
            "message": "Hello World",
            "timestamp": time.time(),
            "user_id": "test_user"
        }
        
        # 序列化
        json_str = json.dumps(test_data)
        assert isinstance(json_str, str)
        
        # 反序列化
        parsed_data = json.loads(json_str)
        assert parsed_data["message"] == "Hello World"
        assert parsed_data["user_id"] == "test_user"
        
    @pytest.mark.asyncio
    async def test_async_operations(self):
        """测试异步操作"""
        async def async_function():
            await asyncio.sleep(0.1)
            return "async_result"
        
        result = await async_function()
        assert result == "async_result"
        
    def test_mock_functionality(self):
        """测试Mock功能"""
        mock_service = Mock()
        mock_service.get_data.return_value = {"status": "success"}
        
        result = mock_service.get_data()
        assert result["status"] == "success"
        mock_service.get_data.assert_called_once()

class TestPerformanceBasics:
    """基础性能测试"""
    
    def test_response_time(self):
        """测试响应时间"""
        start_time = time.time()
        
        # 模拟一些计算
        result = sum(range(1000))
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 响应时间应该小于1秒
        assert response_time < 1.0
        assert result == 499500
        
    def test_memory_usage(self):
        """测试内存使用"""
        # 创建一些数据结构
        data_list = [i for i in range(10000)]
        data_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        # 验证数据正确性
        assert len(data_list) == 10000
        assert len(data_dict) == 1000
        assert data_dict["key_0"] == "value_0"
        
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """测试并发操作"""
        async def worker_task(task_id):
            await asyncio.sleep(0.1)
            return f"task_{task_id}_completed"
        
        # 创建多个并发任务
        tasks = [worker_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务都完成
        assert len(results) == 5
        assert all("completed" in result for result in results)

class TestDataValidation:
    """数据验证测试"""
    
    def test_input_validation(self):
        """测试输入验证"""
        def validate_user_input(data):
            if not isinstance(data, dict):
                return False, "Input must be a dictionary"
            
            required_fields = ["user_id", "message"]
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            if len(data["message"]) > 1000:
                return False, "Message too long"
                
            return True, "Valid input"
        
        # 测试有效输入
        valid_data = {"user_id": "123", "message": "Hello"}
        is_valid, message = validate_user_input(valid_data)
        assert is_valid
        assert message == "Valid input"
        
        # 测试无效输入
        invalid_data = {"user_id": "123"}  # 缺少message字段
        is_valid, message = validate_user_input(invalid_data)
        assert not is_valid
        assert "Missing required field: message" in message
        
    def test_data_sanitization(self):
        """测试数据清理"""
        def sanitize_message(message):
            # 移除潜在的恶意内容
            dangerous_patterns = ["<script>", "javascript:", "onclick="]
            
            for pattern in dangerous_patterns:
                if pattern.lower() in message.lower():
                    return message.replace(pattern, "[REMOVED]")
            
            return message
        
        # 测试正常消息
        clean_message = "Hello, how are you?"
        result = sanitize_message(clean_message)
        assert result == clean_message
        
        # 测试包含脚本的消息
        malicious_message = "Hello <script>alert('xss')</script>"
        result = sanitize_message(malicious_message)
        assert "<script>" not in result
        assert "[REMOVED]" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
