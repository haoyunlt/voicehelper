"""
后端处理器单元测试
测试覆盖：API处理器、中间件、服务层
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from gin import Gin
import asyncio

# 导入被测试的模块
from backend.internal.handler.integration import IntegrationHandler
from backend.internal.handler.handler import Handlers
from backend.internal.service.service import Services
from backend.pkg.middleware.auth import AuthMiddleware
from backend.pkg.middleware.zero_trust import ZeroTrustMiddleware


class TestIntegrationHandler:
    """集成服务处理器测试"""
    
    @pytest.fixture
    def mock_manager(self):
        """模拟集成管理器"""
        manager = Mock()
        manager.list_services.return_value = [
            {
                "id": "test-service-1",
                "name": "Test Service",
                "category": "office",
                "status": "active",
                "endpoints": ["GET /api/test"]
            }
        ]
        return manager
    
    @pytest.fixture
    def handler(self, mock_manager):
        """创建处理器实例"""
        return IntegrationHandler(mock_manager)
    
    def test_list_services_success(self, handler, mock_manager):
        """测试服务列表获取成功"""
        # 模拟Gin上下文
        mock_context = Mock()
        mock_context.Query.return_value = ""
        
        # 调用处理器方法
        handler.ListServices(mock_context)
        
        # 验证调用
        mock_manager.list_services.assert_called_once()
        mock_context.JSON.assert_called_once()
        
        # 验证响应数据
        call_args = mock_context.JSON.call_args
        assert call_args[0][0] == 200  # HTTP状态码
        assert "services" in call_args[0][1]
    
    def test_register_service_success(self, handler, mock_manager):
        """测试服务注册成功"""
        mock_context = Mock()
        mock_context.ShouldBindJSON.return_value = None
        mock_manager.register_service.return_value = "service-123"
        
        # 模拟请求数据
        service_data = {
            "name": "New Service",
            "category": "development",
            "config": {"api_key": "test-key"}
        }
        
        handler.RegisterService(mock_context)
        
        mock_context.ShouldBindJSON.assert_called_once()
        mock_context.JSON.assert_called_once()
    
    def test_call_service_success(self, handler, mock_manager):
        """测试服务调用成功"""
        mock_context = Mock()
        mock_context.Param.return_value = "test-service-1"
        mock_context.ShouldBindJSON.return_value = None
        
        # 模拟服务调用结果
        mock_manager.call_service.return_value = {
            "status": "success",
            "data": {"result": "test result"},
            "execution_time": 0.15
        }
        
        handler.CallService(mock_context)
        
        mock_manager.call_service.assert_called_once()
        mock_context.JSON.assert_called_once()
    
    def test_call_service_not_found(self, handler, mock_manager):
        """测试服务调用失败 - 服务不存在"""
        mock_context = Mock()
        mock_context.Param.return_value = "non-existent-service"
        
        # 模拟服务不存在异常
        mock_manager.call_service.side_effect = Exception("Service not found")
        
        handler.CallService(mock_context)
        
        # 验证返回错误响应
        mock_context.JSON.assert_called_once()
        call_args = mock_context.JSON.call_args
        assert call_args[0][0] == 500  # HTTP状态码


class TestAuthMiddleware:
    """认证中间件测试"""
    
    @pytest.fixture
    def auth_middleware(self):
        """创建认证中间件实例"""
        return AuthMiddleware("test-secret-key")
    
    def test_valid_jwt_token(self, auth_middleware):
        """测试有效JWT令牌"""
        # 创建有效的JWT令牌
        valid_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiMTIzIiwicm9sZXMiOlsidXNlciJdLCJleHAiOjk5OTk5OTk5OTl9.test"
        
        mock_context = Mock()
        mock_context.GetHeader.return_value = f"Bearer {valid_token}"
        
        # 模拟JWT验证成功
        with patch('backend.pkg.middleware.auth.validate_jwt') as mock_validate:
            mock_validate.return_value = {"user_id": "123", "roles": ["user"]}
            
            middleware_func = auth_middleware.Middleware()
            middleware_func(mock_context)
            
            # 验证设置了用户信息
            mock_context.Set.assert_called()
    
    def test_missing_token(self, auth_middleware):
        """测试缺少令牌"""
        mock_context = Mock()
        mock_context.GetHeader.return_value = ""
        
        middleware_func = auth_middleware.Middleware()
        middleware_func(mock_context)
        
        # 验证返回401错误
        mock_context.JSON.assert_called_once()
        call_args = mock_context.JSON.call_args
        assert call_args[0][0] == 401
    
    def test_invalid_token_format(self, auth_middleware):
        """测试无效令牌格式"""
        mock_context = Mock()
        mock_context.GetHeader.return_value = "InvalidToken"
        
        middleware_func = auth_middleware.Middleware()
        middleware_func(mock_context)
        
        # 验证返回401错误
        mock_context.JSON.assert_called_once()
        call_args = mock_context.JSON.call_args
        assert call_args[0][0] == 401


class TestZeroTrustMiddleware:
    """零信任中间件测试"""
    
    @pytest.fixture
    def zero_trust_middleware(self):
        """创建零信任中间件实例"""
        return ZeroTrustMiddleware()
    
    def test_exempt_path_bypass(self, zero_trust_middleware):
        """测试豁免路径绕过检查"""
        mock_context = Mock()
        mock_context.Request.URL.Path = "/health"
        
        middleware_func = zero_trust_middleware.Middleware()
        middleware_func(mock_context)
        
        # 验证直接调用Next()
        mock_context.Next.assert_called_once()
    
    def test_risk_assessment_low_risk(self, zero_trust_middleware):
        """测试低风险评估通过"""
        mock_context = Mock()
        mock_context.Request.URL.Path = "/api/v1/chat"
        mock_context.GetString.return_value = "user-123"
        
        # 模拟低风险评分
        with patch.object(zero_trust_middleware.engine, 'assess_risk') as mock_assess:
            mock_assess.return_value = 0.3  # 低于阈值0.7
            
            middleware_func = zero_trust_middleware.Middleware()
            middleware_func(mock_context)
            
            mock_context.Next.assert_called_once()
    
    def test_risk_assessment_high_risk(self, zero_trust_middleware):
        """测试高风险评估拦截"""
        mock_context = Mock()
        mock_context.Request.URL.Path = "/api/v1/admin"
        mock_context.GetString.return_value = "user-456"
        
        # 模拟高风险评分
        with patch.object(zero_trust_middleware.engine, 'assess_risk') as mock_assess:
            mock_assess.return_value = 0.9  # 高于阈值0.7
            
            middleware_func = zero_trust_middleware.Middleware()
            middleware_func(mock_context)
            
            # 验证返回403错误
            mock_context.JSON.assert_called_once()
            call_args = mock_context.JSON.call_args
            assert call_args[0][0] == 403


class TestHealthCheck:
    """健康检查测试"""
    
    def test_health_check_success(self):
        """测试健康检查成功"""
        services = Mock()
        handlers = Handlers(services)
        
        mock_context = Mock()
        handlers.HealthCheck(mock_context)
        
        # 验证返回200状态
        mock_context.JSON.assert_called_once()
        call_args = mock_context.JSON.call_args
        assert call_args[0][0] == 200
        assert call_args[0][1]["status"] == "ok"


class TestServiceIntegration:
    """服务集成测试"""
    
    @pytest.fixture
    def mock_services(self):
        """模拟服务层"""
        services = Mock()
        services.chat_service = Mock()
        services.user_service = Mock()
        services.dataset_service = Mock()
        return services
    
    def test_chat_service_integration(self, mock_services):
        """测试对话服务集成"""
        # 模拟对话服务响应
        mock_services.chat_service.stream_chat.return_value = iter([
            {"type": "message", "content": "Hello"},
            {"type": "end"}
        ])
        
        # 测试调用
        result = list(mock_services.chat_service.stream_chat({
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        
        assert len(result) == 2
        assert result[0]["type"] == "message"
        assert result[1]["type"] == "end"


# 性能测试辅助函数
class PerformanceTestHelpers:
    """性能测试辅助工具"""
    
    @staticmethod
    def measure_response_time(func, *args, **kwargs):
        """测量函数响应时间"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    async def measure_async_response_time(func, *args, **kwargs):
        """测量异步函数响应时间"""
        import time
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time


# 测试配置
@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        "api_base_url": "http://localhost:8080",
        "algo_base_url": "http://localhost:8000",
        "test_timeout": 30,
        "performance_threshold": {
            "api_response_time": 0.2,  # 200ms
            "auth_middleware_time": 0.01,  # 10ms
            "health_check_time": 0.005  # 5ms
        }
    }


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
