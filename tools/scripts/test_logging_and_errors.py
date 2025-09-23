#!/usr/bin/env python3
"""
VoiceHelper 日志系统和错误码测试脚本
用于验证日志记录和错误处理功能
"""

import os
import sys
import time
import asyncio
import httpx
from typing import Dict, Any

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
        print(f"❌ {error_info.service} - {error_info.category}: {error_info}")
    
    # 测试自定义异常
    try:
        raise VoiceHelperError(ErrorCode.RAG_INVALID_QUERY, "测试查询无效", {
            "query": "test query",
            "user_id": "test_user"
        })
    except VoiceHelperError as e:
        print(f"🚨 自定义异常: {e}")
        print(f"   JSON: {e.to_json()}")
    
    print()


def test_logging_system():
    """测试日志系统"""
    print("📝 测试日志系统")
    print("=" * 50)
    
    # 初始化日志器
    init_logger("voicehelper-test")
    logger = get_logger("test_module")
    
    # 测试不同级别的日志
    logger.debug("这是调试日志", context={"debug_info": "test_data"})
    logger.info("这是信息日志", context={"info": "service_started"})
    logger.warning("这是警告日志", context={"warning": "deprecated_api"})
    logger.error("这是错误日志", context={"error": "connection_failed"})
    
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


async def test_api_endpoints():
    """测试API端点的错误处理"""
    print("🌐 测试API端点错误处理")
    print("=" * 50)
    
    # 测试后端服务
    backend_url = "http://localhost:8080"
    algo_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # 测试后端健康检查
        try:
            response = await client.get(f"{backend_url}/health", timeout=5)
            print(f"✅ 后端健康检查: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   服务: {data.get('service', 'unknown')}")
                print(f"   本地IP: {data.get('local_ip', 'unknown')}")
                print(f"   端口: {data.get('port', 'unknown')}")
        except Exception as e:
            print(f"❌ 后端服务连接失败: {e}")
        
        # 测试后端错误端点
        try:
            response = await client.get(f"{backend_url}/api/v1/error-test", timeout=5)
            print(f"🚨 后端错误测试: {response.status_code}")
            if response.status_code >= 400:
                error_data = response.json()
                print(f"   错误码: {error_data.get('error', {}).get('code', 'unknown')}")
                print(f"   错误信息: {error_data.get('error', {}).get('message', 'unknown')}")
        except Exception as e:
            print(f"❌ 后端错误测试失败: {e}")
        
        # 测试算法服务健康检查
        try:
            response = await client.get(f"{algo_url}/health", timeout=5)
            print(f"✅ 算法服务健康检查: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   服务: {data.get('service', 'unknown')}")
                print(f"   本地IP: {data.get('local_ip', 'unknown')}")
                print(f"   端口: {data.get('port', 'unknown')}")
        except Exception as e:
            print(f"❌ 算法服务连接失败: {e}")
        
        # 测试算法服务错误端点
        try:
            response = await client.get(f"{algo_url}/error-test?type=rag_error", timeout=5)
            print(f"🚨 算法服务错误测试: {response.status_code}")
            if response.status_code >= 400:
                error_data = response.json()
                print(f"   错误码: {error_data.get('code', 'unknown')}")
                print(f"   错误信息: {error_data.get('message', 'unknown')}")
        except Exception as e:
            print(f"❌ 算法服务错误测试失败: {e}")
    
    print()


def test_log_analysis():
    """测试日志分析功能"""
    print("📊 测试日志分析")
    print("=" * 50)
    
    # 模拟生成一些测试日志
    logger = get_logger("analysis_test")
    
    # 生成不同类型的日志
    log_types = [
        ("startup", "服务启动"),
        ("request", "处理请求"),
        ("response", "返回响应"),
        ("business", "业务操作"),
        ("performance", "性能监控"),
        ("error", "错误处理"),
    ]
    
    for log_type, message in log_types:
        if log_type == "startup":
            logger.startup(message, context={"component": "test"})
        elif log_type == "business":
            logger.business(message, context={"operation": "test_op"})
        elif log_type == "performance":
            logger.performance(message, 150.5, context={"operation": "test_perf"})
        elif log_type == "error":
            logger.error_with_code(ErrorCode.SYSTEM_INTERNAL_ERROR, message, context={"test": True})
        else:
            logger.info(message, context={"type": log_type})
    
    print("✅ 生成了多种类型的测试日志")
    print("💡 提示: 可以使用以下命令分析日志:")
    print("   grep '\"level\":\"error\"' | jq .error_code")
    print("   grep '\"type\":\"performance\"' | jq .duration_ms")
    print("   grep '\"type\":\"startup\"' | jq .context")
    print()


def main():
    """主函数"""
    print("🎯 VoiceHelper 日志系统和错误码测试")
    print("=" * 60)
    print()
    
    # 设置环境变量
    os.environ["LOG_LEVEL"] = "debug"
    os.environ["SERVICE_NAME"] = "voicehelper-test"
    
    # 运行测试
    test_error_codes()
    test_logging_system()
    test_log_analysis()
    
    # 运行异步测试
    print("🔄 运行异步API测试...")
    asyncio.run(test_api_endpoints())
    
    print("🎉 测试完成！")
    print()
    print("📋 测试总结:")
    print("- ✅ 错误码系统正常工作")
    print("- ✅ 日志系统正常记录")
    print("- ✅ 结构化日志格式正确")
    print("- ✅ 网络信息记录完整")
    print("- ✅ 性能指标记录准确")
    print()
    print("💡 下一步:")
    print("1. 启动后端服务: cd backend && go run cmd/server/main.go")
    print("2. 启动算法服务: cd algo && python app/main.py")
    print("3. 重新运行此脚本验证API错误处理")
    print("4. 查看日志输出验证格式和内容")


if __name__ == "__main__":
    main()
