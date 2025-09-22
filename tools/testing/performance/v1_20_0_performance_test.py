#!/usr/bin/env python3
"""
VoiceHelper v1.20.0 性能测试
测试语音延迟优化、情感识别和批处理调度器性能
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

# 导入v1.20.0核心模块
try:
    from algo.core.enhanced_voice_optimizer import EnhancedVoiceOptimizer, VoiceResponse
    from algo.core.advanced_emotion_recognition import AdvancedEmotionRecognition, EmotionAnalysisResult
    from algo.core.simple_batch_scheduler import SimpleBatchScheduler, ProcessRequest, RequestType, RequestPriority
    print("✅ 成功导入v1.20.0核心模块")
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

class V120PerformanceTest:
    """v1.20.0性能测试套件"""
    
    def __init__(self):
        self.voice_optimizer = EnhancedVoiceOptimizer()
        self.emotion_recognizer = AdvancedEmotionRecognition()
        self.batch_scheduler = SimpleBatchScheduler()
        self.results = []
        
    async def test_voice_latency_optimization(self) -> PerformanceResult:
        """测试语音延迟优化"""
        print("\n🎤 测试语音延迟优化...")
        
        test_cases = [
            {"audio_length": 1, "expected_latency": 100},   # 1秒音频，期望100ms
            {"audio_length": 3, "expected_latency": 150},   # 3秒音频，期望150ms
            {"audio_length": 5, "expected_latency": 200},   # 5秒音频，期望200ms
            {"audio_length": 10, "expected_latency": 300},  # 10秒音频，期望300ms
        ]
        
        total_latency = 0
        passed_tests = 0
        latency_results = []
        
        for case in test_cases:
            # 生成测试音频数据
            audio_data = self._generate_test_audio(case["audio_length"])
            
            # 测试语音优化
            start_time = time.time()
            try:
                result = await self.voice_optimizer.optimize_voice_pipeline(audio_data)
                latency = (time.time() - start_time) * 1000  # 转换为毫秒
                
                latency_results.append({
                    "audio_length": case["audio_length"],
                    "expected_latency": case["expected_latency"],
                    "actual_latency": latency,
                    "passed": latency <= case["expected_latency"]
                })
                
                if latency <= case["expected_latency"]:
                    passed_tests += 1
                    print(f"  ✅ {case['audio_length']}秒音频: {latency:.1f}ms (目标{case['expected_latency']}ms)")
                else:
                    print(f"  ❌ {case['audio_length']}秒音频: {latency:.1f}ms (目标{case['expected_latency']}ms)")
                
                total_latency += latency
                
            except Exception as e:
                print(f"  ❌ 测试失败: {e}")
                latency_results.append({
                    "audio_length": case["audio_length"],
                    "expected_latency": case["expected_latency"],
                    "actual_latency": float('inf'),
                    "passed": False
                })
        
        avg_latency = total_latency / len(test_cases) if test_cases else 0
        success_rate = (passed_tests / len(test_cases)) * 100 if test_cases else 0
        
        return PerformanceResult(
            test_name="voice_latency_optimization",
            duration=time.time(),
            success=success_rate >= 75,  # 75%通过率
            metrics={
                "average_latency": avg_latency,
                "success_rate": success_rate,
                "passed_tests": passed_tests,
                "total_tests": len(test_cases),
                "latency_results": latency_results
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_emotion_recognition_accuracy(self) -> PerformanceResult:
        """测试情感识别准确率"""
        print("\n🧠 测试情感识别准确率...")
        
        # 测试样本
        test_samples = [
            {"text": "我今天很开心！", "expected_emotion": "happy"},
            {"text": "这让我很沮丧", "expected_emotion": "sad"},
            {"text": "我生气了！", "expected_emotion": "angry"},
            {"text": "这很正常", "expected_emotion": "neutral"},
            {"text": "太棒了！", "expected_emotion": "excited"},
            {"text": "我很平静", "expected_emotion": "calm"},
            {"text": "这让我很困惑", "expected_emotion": "confused"},
            {"text": "我很失望", "expected_emotion": "sad"},
            {"text": "太令人兴奋了！", "expected_emotion": "excited"},
            {"text": "我感到很放松", "expected_emotion": "calm"}
        ]
        
        correct_predictions = 0
        total_processing_time = 0
        confidence_scores = []
        
        for sample in test_samples:
            try:
                start_time = time.time()
                result = await self.emotion_recognizer.analyze_text_only(
                    text=sample["text"],
                    user_id="test_user"
                )
                processing_time = (time.time() - start_time) * 1000
                total_processing_time += processing_time
                
                # 检查预测结果
                predicted_emotion = result.primary_emotion
                confidence = result.confidence
                confidence_scores.append(confidence)
                
                if predicted_emotion == sample["expected_emotion"]:
                    correct_predictions += 1
                    print(f"  ✅ '{sample['text']}' -> {predicted_emotion} (置信度: {confidence:.2f})")
                else:
                    print(f"  ❌ '{sample['text']}' -> {predicted_emotion} (期望: {sample['expected_emotion']}, 置信度: {confidence:.2f})")
                
            except Exception as e:
                print(f"  ❌ 情感识别失败: {e}")
        
        accuracy = (correct_predictions / len(test_samples)) * 100 if test_samples else 0
        avg_processing_time = total_processing_time / len(test_samples) if test_samples else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return PerformanceResult(
            test_name="emotion_recognition_accuracy",
            duration=time.time(),
            success=accuracy >= 40,  # 当前目标40%（演示版本）
            metrics={
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_samples": len(test_samples),
                "avg_processing_time": avg_processing_time,
                "avg_confidence": avg_confidence,
                "confidence_scores": confidence_scores
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_batch_processing_throughput(self) -> PerformanceResult:
        """测试批处理吞吐量"""
        print("\n⚡ 测试批处理吞吐量...")
        
        # 启动批处理调度器
        await self.batch_scheduler.start()
        
        # 测试不同批大小的性能
        batch_sizes = [10, 25, 50, 100, 200]
        throughput_results = []
        
        for batch_size in batch_sizes:
            print(f"  测试批大小: {batch_size}")
            
            # 创建测试请求
            requests = []
            for i in range(batch_size):
                request = ProcessRequest(
                    id=f"test_{i}",
                    type=RequestType.TEXT_GENERATION,
                    data=f"test_data_{i}",
                    user_id="test_user",
                    priority=RequestPriority.NORMAL
                )
                requests.append(request)
            
            # 提交请求并测量性能
            start_time = time.time()
            
            # 提交所有请求
            for request in requests:
                await self.batch_scheduler.submit_request(request)
            
            # 处理批次
            batch_result = await self.batch_scheduler.process_batch(requests)
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = batch_result.throughput
            
            throughput_results.append({
                "batch_size": batch_size,
                "duration": duration,
                "throughput": throughput
            })
            
            print(f"    吞吐量: {throughput:.1f} req/s")
        
        # 停止调度器
        await self.batch_scheduler.stop()
        
        # 计算最大吞吐量
        max_throughput = max(result["throughput"] for result in throughput_results)
        
        return PerformanceResult(
            test_name="batch_processing_throughput",
            duration=time.time(),
            success=max_throughput >= 20,  # 目标20 req/s
            metrics={
                "max_throughput": max_throughput,
                "batch_results": throughput_results,
                "target_throughput": 20
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_system_stability(self) -> PerformanceResult:
        """测试系统稳定性"""
        print("\n🔄 测试系统稳定性...")
        
        # 简化稳定性测试
        test_requests = 100
        start_time = time.time()
        request_count = 0
        error_count = 0
        latencies = []
        
        print(f"  运行{test_requests}个请求的稳定性测试...")
        
        for i in range(test_requests):
            try:
                # 模拟语音处理请求
                audio_data = self._generate_test_audio(1)
                request_start = time.time()
                
                # 语音优化
                voice_result = await self.voice_optimizer.optimize_voice_pipeline(audio_data)
                
                # 情感识别
                emotion_result = await self.emotion_recognizer.analyze_text_only(
                    text=voice_result.text_response,
                    user_id="stability_test"
                )
                
                request_latency = (time.time() - request_start) * 1000
                latencies.append(request_latency)
                request_count += 1
                
                # 每20个请求输出一次状态
                if request_count % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"    已处理 {request_count} 个请求，耗时 {elapsed:.1f}s")
                
            except Exception as e:
                error_count += 1
                print(f"    错误: {e}")
        
        total_duration = time.time() - start_time
        success_rate = ((request_count - error_count) / request_count) * 100 if request_count > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        
        return PerformanceResult(
            test_name="system_stability",
            duration=total_duration,
            success=success_rate >= 99 and avg_latency <= 50,
            metrics={
                "total_requests": request_count,
                "error_count": error_count,
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "p95_latency": p95_latency,
                "test_duration": total_duration
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_test_audio(self, duration_seconds: int) -> bytes:
        """生成测试音频数据"""
        # 模拟音频数据（实际应该是真实的音频字节）
        return b"fake_audio_data_" + str(duration_seconds).encode() * 100
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有性能测试"""
        print("🚀 开始VoiceHelper v1.20.0性能测试")
        print("=" * 50)
        
        start_time = time.time()
        
        # 运行各项测试
        tests = [
            self.test_voice_latency_optimization(),
            self.test_emotion_recognition_accuracy(),
            self.test_batch_processing_throughput(),
            self.test_system_stability()
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
        print(f"🎯 测试完成！")
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
    tester = V120PerformanceTest()
    report = await tester.run_all_tests()
    
    # 保存测试报告
    report_file = f"v1_20_0_performance_results_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 测试报告已保存: {report_file}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())
