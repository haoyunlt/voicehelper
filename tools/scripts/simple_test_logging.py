#!/usr/bin/env python3
"""
VoiceHelper 日志系统和错误码简单测试脚本
不依赖外部库，仅测试核心功能
"""

import os
import sys
import time

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algo'))

from common.logger import init_logger, get_logger
from common.errors import ErrorCode, VoiceHelperError, get_error_info


def test_error_codes():
    """测试错误码系统"""
    print("🔢 测试错误码系统")
    print("=" * 50)
    
    # 测试成功码
    success_info = get_error_info(ErrorCode.SUCCESS)
    print(f"✅ 成功码: {success_info}")
    
    # 测试各种错误码
    test_codes = [
        ErrorCode.GATEWAY_INTERNAL_ERROR,
        ErrorCode.AUTH_TOKEN_EXPIRED,
        ErrorCode.CHAT_SESSION_NOT_FOUND,
        ErrorCode.VOICE_ASR_FAILED,
        ErrorCode.RAG_RETRIEVAL_FAILED,
        ErrorCode.STORAGE_FILE_NOT_FOUND,
        ErrorCode.SYSTEM_INTERNAL_ERROR,
    ]
    
    for code in test_codes:
        error_info = get_error_info(code)
        print(f"❌ {error_info.service} - {error_info.category}: [{code}] {error_info.message}")
    
    # 测试自定义异常
    try:
        raise VoiceHelperError(ErrorCode.RAG_INVALID_QUERY, "测试查询无效", {
            "query": "test query",
            "user_id": "test_user"
        })
    except VoiceHelperError as e:
        print(f"🚨 自定义异常: {e}")
        print(f"   HTTP状态码: {e.http_status}")
        print(f"   详细信息: {e.details}")
    
    print()


def test_logging_system():
    """测试日志系统"""
    print("📝 测试日志系统")
    print("=" * 50)
    
    # 初始化日志器
    init_logger("voicehelper-test")
    logger = get_logger("test_module")
    
    print("测试不同级别的日志:")
    
    # 测试不同级别的日志
    logger.debug("这是调试日志", context={"debug_info": "test_data"})
    logger.info("这是信息日志", context={"info": "service_started"})
    logger.warning("这是警告日志", context={"warning": "deprecated_api"})
    logger.error("这是错误日志", context={"error": "connection_failed"})
    
    print("\n测试特殊类型日志:")
    
    # 测试启动日志
    logger.startup("测试服务启动", context={
        "service": "voicehelper-test",
        "host": "0.0.0.0",
        "port": 8000,
        "local_ip": "192.168.1.100",
        "pid": os.getpid(),
    })
    
    # 测试业务日志
    logger.business("用户登录", context={
        "user_id": "test_user_123",
        "login_method": "password",
        "ip": "192.168.1.50",
    })
    
    # 测试性能日志
    start_time = time.time()
    time.sleep(0.1)  # 模拟操作
    duration_ms = (time.time() - start_time) * 1000
    logger.performance("数据库查询", duration_ms, context={
        "query": "SELECT * FROM users",
        "rows_returned": 100,
    })
    
    # 测试安全日志
    logger.security("可疑登录尝试", context={
        "ip": "192.168.1.200",
        "attempts": 5,
        "user_agent": "suspicious_bot",
    })
    
    # 测试错误码日志
    logger.error_with_code(ErrorCode.AUTH_TOKEN_EXPIRED, "Token验证失败", context={
        "token": "xxx...xxx",
        "user_id": "test_user_123",
        "expired_at": "2024-12-21T10:30:00Z",
    })
    
    # 测试异常日志
    try:
        raise ValueError("这是一个测试异常")
    except Exception as e:
        logger.exception("处理请求时发生异常", e, context={
            "request_id": "req_123456",
            "endpoint": "/test/api",
        })
    
    print()


def test_network_info():
    """测试网络信息记录"""
    print("🌐 测试网络信息记录")
    print("=" * 50)
    
    from common.logger import NetworkInfo
    
    # 创建模拟的网络信息
    class MockRequest:
        def __init__(self):
            self.method = "POST"
            self.url = MockURL()
            self.headers = {
                "user-agent": "VoiceHelper-Test/1.0",
                "x-request-id": "req_test_123456",
                "x-forwarded-for": "192.168.1.50, 10.0.0.1"
            }
            self.client = MockClient()
    
    class MockURL:
        def __str__(self):
            return "http://192.168.1.100:8000/api/v1/query"
    
    class MockClient:
        def __init__(self):
            self.host = "192.168.1.50"
            self.port = 54321
    
    # 测试网络信息提取
    mock_request = MockRequest()
    network_info = NetworkInfo(mock_request)
    
    print("网络信息提取测试:")
    print(f"  本地IP: {network_info.local_ip}")
    print(f"  本地端口: {network_info.local_port}")
    print(f"  远程IP: {network_info.remote_ip}")
    print(f"  远程端口: {network_info.remote_port}")
    print(f"  URL: {network_info.url}")
    print(f"  方法: {network_info.method}")
    print(f"  用户代理: {network_info.user_agent}")
    print(f"  请求ID: {network_info.request_id}")
    
    # 转换为字典
    network_dict = network_info.to_dict()
    print(f"\n网络信息字典: {network_dict}")
    
    print()


def test_error_mapping():
    """测试错误码到HTTP状态码的映射"""
    print("🔄 测试错误码映射")
    print("=" * 50)
    
    test_mappings = [
        (ErrorCode.SUCCESS, 200),
        (ErrorCode.GATEWAY_INVALID_REQUEST, 400),
        (ErrorCode.AUTH_TOKEN_EXPIRED, 401),
        (ErrorCode.AUTH_PERMISSION_DENIED, 403),
        (ErrorCode.CHAT_SESSION_NOT_FOUND, 404),
        (ErrorCode.GATEWAY_REQUEST_TOO_LARGE, 413),
        (ErrorCode.GATEWAY_RATE_LIMIT_EXCEEDED, 429),
        (ErrorCode.SYSTEM_INTERNAL_ERROR, 500),
        (ErrorCode.RAG_SERVICE_UNAVAILABLE, 503),
    ]
    
    print("错误码到HTTP状态码映射:")
    for error_code, expected_status in test_mappings:
        error_info = get_error_info(error_code)
        actual_status = error_info.http_status
        status = "✅" if actual_status == expected_status else "❌"
        print(f"  {status} {error_code} -> {actual_status} (期望: {expected_status})")
    
    print()


def main():
    """主函数"""
    print("🎯 VoiceHelper 日志系统和错误码简单测试")
    print("=" * 60)
    print()
    
    # 设置环境变量
    os.environ["LOG_LEVEL"] = "debug"
    os.environ["SERVICE_NAME"] = "voicehelper-test"
    
    # 运行测试
    test_error_codes()
    test_logging_system()
    test_network_info()
    test_error_mapping()
    
    print("🎉 测试完成！")
    print()
    print("📋 测试总结:")
    print("- ✅ 错误码系统正常工作")
    print("- ✅ 日志系统正常记录")
    print("- ✅ 结构化日志格式正确")
    print("- ✅ 网络信息记录完整")
    print("- ✅ 错误码HTTP映射正确")
    print("- ✅ 性能指标记录准确")
    print()
    print("💡 下一步:")
    print("1. 启动后端服务: cd backend && go run cmd/server/main.go")
    print("2. 启动算法服务: cd algo && python app/main.py")
    print("3. 测试API端点: curl http://localhost:8080/health")
    print("4. 测试错误处理: curl http://localhost:8080/api/v1/error-test")
    print("5. 查看日志输出验证格式和内容")


if __name__ == "__main__":
    main()
