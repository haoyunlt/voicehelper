#!/usr/bin/env python3
"""
VoiceHelper v1.23.0 性能测试
测试实时语音打断检测、多语言支持、增强安全认证、移动端优化
"""

import asyncio
import time
import json
import sys
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# 导入v1.23.0核心模块
try:
    from algo.core.realtime_interrupt_detection import (
        RealtimeInterruptSystem, InterruptType, ConversationState,
        start_interrupt_session, process_interrupt_input, recover_from_interrupt,
        get_interrupt_system_stats
    )
    from algo.core.multilingual_support_system import (
        MultilingualSupportSystem, SupportedLanguage, process_multilingual_text,
        set_user_language, get_supported_languages, get_multilingual_stats
    )
    from algo.core.enhanced_security_auth import (
        EnhancedSecuritySystem, AuthenticationMethod, SecurityLevel,
        authenticate_user, validate_session, get_security_stats
    )
    from algo.core.mobile_optimization_system import (
        MobileOptimizationSystem, DeviceType, NetworkType, TouchGesture,
        DeviceInfo, register_mobile_device, process_mobile_request, get_mobile_stats
    )
    print("✅ 成功导入v1.23.0核心模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

@dataclass
class PerformanceResult:
    """性能测试结果"""
    test_name: str
    duration: float
    success: bool
    metrics: Dict[str, Any]
    timestamp: str

class V123PerformanceTest:
    """v1.23.0性能测试套件"""
    
    def __init__(self):
        self.interrupt_system = RealtimeInterruptSystem()
        self.multilingual_system = MultilingualSupportSystem()
        self.security_system = EnhancedSecuritySystem()
        self.mobile_system = MobileOptimizationSystem()
        self.results = []
        
    async def test_realtime_interrupt_detection(self) -> PerformanceResult:
        """测试实时语音打断检测"""
        print("\n🎯 测试实时语音打断检测...")
        
        # 开始会话
        session_id = "test_interrupt_session"
        user_id = "test_user_001"
        context = await start_interrupt_session(session_id, user_id)
        
        # 测试打断检测
        test_cases = [
            {"audio_data": b"fake_audio_1", "text_input": "停止，我需要打断", "expected_interrupt": True},
            {"audio_data": b"fake_audio_2", "text_input": "继续对话", "expected_interrupt": False},
            {"audio_data": b"fake_audio_3", "text_input": "紧急停止", "expected_interrupt": True},
            {"audio_data": b"fake_audio_4", "text_input": "正常交流", "expected_interrupt": False},
            {"audio_data": b"fake_audio_5", "text_input": "暂停一下", "expected_interrupt": True}
        ]
        
        start_time = time.time()
        interrupt_count = 0
        correct_detections = 0
        
        for i, test_case in enumerate(test_cases):
            response = await process_interrupt_input(
                session_id, 
                test_case["audio_data"], 
                test_case["text_input"]
            )
            
            if response.should_interrupt == test_case["expected_interrupt"]:
                correct_detections += 1
            
            if response.should_interrupt:
                interrupt_count += 1
                # 测试恢复
                recovered = await recover_from_interrupt(session_id)
                if not recovered:
                    print(f"  恢复失败: 测试用例 {i+1}")
        
        detection_time = time.time() - start_time
        detection_accuracy = correct_detections / len(test_cases)
        
        # 获取系统统计
        stats = get_interrupt_system_stats()
        
        # 验证结果
        success = (detection_accuracy >= 0.6 and 
                  detection_time < 5.0 and
                  interrupt_count >= 1)
        
        print(f"  打断检测准确率: {detection_accuracy:.2%}")
        print(f"  检测时间: {detection_time:.2f}s")
        print(f"  检测到打断数: {interrupt_count}")
        
        return PerformanceResult(
            test_name="realtime_interrupt_detection",
            duration=detection_time,
            success=success,
            metrics={
                "test_cases": len(test_cases),
                "correct_detections": correct_detections,
                "detection_accuracy": detection_accuracy,
                "interrupt_count": interrupt_count,
                "detection_time": detection_time,
                "system_stats": stats
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_multilingual_support(self) -> PerformanceResult:
        """测试多语言支持"""
        print("\n🌐 测试多语言支持...")
        
        # 测试多语言文本
        test_texts = [
            ("你好，世界！", "zh-CN"),
            ("Hello, world!", "en-US"),
            ("こんにちは、世界！", "ja-JP"),
            ("안녕하세요, 세계!", "ko-KR"),
            ("Hola, mundo!", "es-ES"),
            ("Bonjour, monde!", "fr-FR"),
            ("Hallo, Welt!", "de-DE"),
            ("你好，世界！", "zh-TW")
        ]
        
        start_time = time.time()
        detection_results = []
        
        for text, expected_lang in test_texts:
            result = await process_multilingual_text(text, f"session_{expected_lang}", "test_user")
            detection_results.append({
                "text": text,
                "expected": expected_lang,
                "detected": result["detected_language"],
                "confidence": result["detection_confidence"],
                "correct": result["detected_language"] == expected_lang
            })
        
        detection_time = time.time() - start_time
        correct_detections = sum(1 for r in detection_results if r["correct"])
        detection_accuracy = correct_detections / len(detection_results)
        
        # 测试语言设置
        set_user_language("test_user", SupportedLanguage.ENGLISH)
        supported_languages = get_supported_languages()
        
        # 获取统计
        stats = get_multilingual_stats()
        
        # 验证结果
        success = (detection_accuracy >= 0.5 and 
                  detection_time < 3.0 and
                  len(supported_languages) >= 8)
        
        print(f"  语言检测准确率: {detection_accuracy:.2%}")
        print(f"  检测时间: {detection_time:.2f}s")
        print(f"  支持语言数: {len(supported_languages)}")
        
        return PerformanceResult(
            test_name="multilingual_support",
            duration=detection_time,
            success=success,
            metrics={
                "test_texts": len(test_texts),
                "correct_detections": correct_detections,
                "detection_accuracy": detection_accuracy,
                "detection_time": detection_time,
                "supported_languages": len(supported_languages),
                "detection_results": detection_results,
                "system_stats": stats
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_enhanced_security_auth(self) -> PerformanceResult:
        """测试增强安全认证"""
        print("\n🔒 测试增强安全认证...")
        
        # 注册测试用户
        self.security_system.mfa_authenticator.register_user(
            "test_user_001", "password123",
            totp_secret="test_secret_001",
            biometric_data={"type": "fingerprint", "template": b"fake_template", "quality": 0.9}
        )
        
        # 测试认证
        test_credentials = [
            {
                "user_id": "test_user_001",
                "credentials": {"password": "password123", "totp_code": "123456"},
                "methods": [AuthenticationMethod.PASSWORD, AuthenticationMethod.TOTP],
                "expected_success": True
            },
            {
                "user_id": "test_user_001",
                "credentials": {"password": "wrong_password"},
                "methods": [AuthenticationMethod.PASSWORD],
                "expected_success": False
            },
            {
                "user_id": "test_user_001",
                "credentials": {"biometric_data": {"type": "fingerprint", "template": b"fake_template"}},
                "methods": [AuthenticationMethod.BIOMETRIC],
                "expected_success": True
            }
        ]
        
        start_time = time.time()
        auth_results = []
        successful_auths = 0
        
        for test_case in test_credentials:
            result = await authenticate_user(
                test_case["user_id"],
                test_case["credentials"],
                test_case["methods"]
            )
            
            auth_results.append({
                "user_id": test_case["user_id"],
                "success": result.success,
                "expected": test_case["expected_success"],
                "security_level": result.security_level.value,
                "methods_used": [m.value for m in result.authentication_methods]
            })
            
            if result.success == test_case["expected_success"]:
                successful_auths += 1
            
            # 测试会话验证
            if result.success:
                session_valid = await validate_session(result.session_token)
                if not session_valid:
                    print(f"  会话验证失败: {test_case['user_id']}")
        
        auth_time = time.time() - start_time
        auth_accuracy = successful_auths / len(test_credentials)
        
        # 获取统计
        stats = get_security_stats()
        
        # 验证结果
        success = (auth_accuracy >= 0.5 and 
                  auth_time < 5.0 and
                  successful_auths >= 1)
        
        print(f"  认证准确率: {auth_accuracy:.2%}")
        print(f"  认证时间: {auth_time:.2f}s")
        print(f"  成功认证数: {successful_auths}")
        
        return PerformanceResult(
            test_name="enhanced_security_auth",
            duration=auth_time,
            success=success,
            metrics={
                "test_cases": len(test_credentials),
                "successful_auths": successful_auths,
                "auth_accuracy": auth_accuracy,
                "auth_time": auth_time,
                "auth_results": auth_results,
                "system_stats": stats
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_mobile_optimization(self) -> PerformanceResult:
        """测试移动端优化"""
        print("\n📱 测试移动端优化...")
        
        # 创建测试设备
        test_devices = [
            DeviceInfo(
                device_id="mobile_001",
                device_type=DeviceType.MOBILE,
                screen_width=375,
                screen_height=667,
                pixel_ratio=2.0,
                os_version="iOS 15.0",
                browser_version="Safari 15.0",
                memory_total=4096,
                memory_available=2048,
                cpu_cores=6,
                network_type=NetworkType.MOBILE_5G,
                battery_level=0.8,
                is_charging=False
            ),
            DeviceInfo(
                device_id="tablet_001",
                device_type=DeviceType.TABLET,
                screen_width=768,
                screen_height=1024,
                pixel_ratio=2.0,
                os_version="Android 12.0",
                browser_version="Chrome 95.0",
                memory_total=8192,
                memory_available=4096,
                cpu_cores=8,
                network_type=NetworkType.WIFI,
                battery_level=0.6,
                is_charging=True
            )
        ]
        
        start_time = time.time()
        device_results = []
        
        for device in test_devices:
            # 注册设备
            registration_result = await register_mobile_device(device)
            device_results.append({
                "device_id": device.device_id,
                "device_type": device.device_type.value,
                "performance_level": registration_result.get("performance_level", "unknown"),
                "optimization_enabled": True
            })
            
            # 测试移动端请求处理
            request_data = {
                "content": "测试内容",
                "images": [{"url": "test.jpg", "size": 1024}],
                "touch_events": [{
                    "event_id": f"touch_{device.device_id}",
                    "gesture": TouchGesture.TAP,
                    "coordinates": (100, 200),
                    "timestamp": time.time(),
                    "pressure": 0.5,
                    "duration": 0.2
                }]
            }
            
            response = await process_mobile_request(device.device_id, request_data)
            if not response.get("device_optimized", False):
                print(f"  设备优化失败: {device.device_id}")
        
        optimization_time = time.time() - start_time
        
        # 获取统计
        stats = get_mobile_stats()
        
        # 验证结果
        success = (len(device_results) >= 2 and 
                  optimization_time < 3.0 and
                  all(r["optimization_enabled"] for r in device_results))
        
        print(f"  注册设备数: {len(device_results)}")
        print(f"  优化时间: {optimization_time:.2f}s")
        print(f"  设备类型: {[r['device_type'] for r in device_results]}")
        
        return PerformanceResult(
            test_name="mobile_optimization",
            duration=optimization_time,
            success=success,
            metrics={
                "test_devices": len(test_devices),
                "device_results": device_results,
                "optimization_time": optimization_time,
                "system_stats": stats
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_system_integration(self) -> PerformanceResult:
        """测试系统集成"""
        print("\n🔗 测试系统集成...")
        
        # 测试多系统协同工作
        start_time = time.time()
        
        # 1. 多语言 + 打断检测
        multilingual_text = "你好，我需要打断一下"
        session_id = "integration_test_session"
        user_id = "integration_test_user"
        
        # 开始会话
        await start_interrupt_session(session_id, user_id)
        
        # 处理多语言文本
        lang_result = await process_multilingual_text(multilingual_text, session_id, user_id)
        
        # 处理打断
        interrupt_result = await process_interrupt_input(session_id, b"fake_audio", multilingual_text)
        
        # 2. 安全认证 + 移动端优化
        device_info = DeviceInfo(
            device_id="integration_device",
            device_type=DeviceType.MOBILE,
            screen_width=375,
            screen_height=667,
            pixel_ratio=2.0,
            os_version="iOS 15.0",
            browser_version="Safari 15.0",
            memory_total=4096,
            memory_available=2048,
            cpu_cores=6,
            network_type=NetworkType.WIFI,
            battery_level=0.8,
            is_charging=False
        )
        
        # 注册设备
        device_result = await register_mobile_device(device_info)
        
        # 认证用户
        auth_result = await authenticate_user(
            "integration_user",
            {"password": "test_password"},
            [AuthenticationMethod.PASSWORD]
        )
        
        integration_time = time.time() - start_time
        
        # 验证集成结果
        integration_success = (
            lang_result["detected_language"] == "zh-CN" and
            device_result["performance_level"] in ["low", "medium", "high", "ultra"]
        )
        
        print(f"  集成测试时间: {integration_time:.2f}s")
        print(f"  多语言检测: {lang_result['detected_language']}")
        print(f"  打断检测: {interrupt_result.should_interrupt}")
        print(f"  设备优化: {device_result['performance_level']}")
        print(f"  安全认证: {auth_result.success}")
        
        return PerformanceResult(
            test_name="system_integration",
            duration=integration_time,
            success=integration_success,
            metrics={
                "integration_time": integration_time,
                "multilingual_detected": lang_result["detected_language"],
                "interrupt_detected": interrupt_result.should_interrupt,
                "device_optimized": device_result.get("performance_level", "unknown"),
                "auth_success": auth_result.success
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有性能测试"""
        print("🚀 开始VoiceHelper v1.23.0性能测试")
        print("=" * 50)
        
        start_time = time.time()
        
        # 运行各项测试
        tests = [
            self.test_realtime_interrupt_detection(),
            self.test_multilingual_support(),
            self.test_enhanced_security_auth(),
            self.test_mobile_optimization(),
            self.test_system_integration()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # 处理结果
        test_results = {}
        passed_tests = 0
        total_tests = len(results)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ 测试 {i+1} 失败: {result}")
                test_results[f"test_{i+1}"] = {
                    "success": False,
                    "error": str(result)
                }
            else:
                test_results[result.test_name] = asdict(result)
                if result.success:
                    passed_tests += 1
                    print(f"✅ {result.test_name}: 通过")
                else:
                    print(f"❌ {result.test_name}: 失败")
        
        # 计算总体评分
        overall_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "version": "v1.23.0",
            "overall_score": overall_score,
            "grade": self._get_grade(overall_score),
            "test_results": test_results,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{overall_score:.1f}%"
            }
        }
        
        total_duration = time.time() - start_time
        
        print("\n" + "=" * 50)
        print(f"🎯 v1.23.0测试完成！")
        print(f"总体评分: {overall_score:.1f}/100")
        print(f"测试状态: {self._get_grade(overall_score)}")
        print(f"通过测试: {passed_tests}/{total_tests}")
        print(f"总耗时: {total_duration:.1f}秒")
        
        return report
    
    def _get_grade(self, score: float) -> str:
        """根据分数获取等级"""
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 80:
            return "A (良好)"
        elif score >= 70:
            return "B (合格)"
        elif score >= 60:
            return "C (及格)"
        else:
            return "D (不及格)"

async def main():
    """主函数"""
    tester = V123PerformanceTest()
    report = await tester.run_all_tests()
    
    # 保存测试报告
    report_file = f"v1_23_0_performance_results_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 测试报告已保存: {report_file}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())
