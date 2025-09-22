#!/usr/bin/env python3
"""
VoiceHelper v1.20.1 性能测试
测试情感识别准确率提升、缓存监控和性能优化
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

# 导入v1.20.1核心模块
try:
    from algo.core.enhanced_voice_optimizer import EnhancedVoiceOptimizer, VoiceResponse
    from algo.core.advanced_emotion_recognition import AdvancedEmotionRecognition, EmotionAnalysisResult
    from algo.core.simple_batch_scheduler import SimpleBatchScheduler, ProcessRequest, RequestType, RequestPriority
    from algo.core.production_emotion_model import predict_emotion_production, production_emotion_model
    from algo.core.cache_monitoring_system import get_cache_metrics, cache_monitor
    print("✅ 成功导入v1.20.1核心模块")
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

class V121PerformanceTest:
    """v1.20.1性能测试套件"""
    
    def __init__(self):
        self.voice_optimizer = EnhancedVoiceOptimizer()
        self.emotion_recognizer = AdvancedEmotionRecognition()
        self.batch_scheduler = SimpleBatchScheduler()
        self.results = []
        
    async def test_improved_emotion_recognition(self) -> PerformanceResult:
        """测试改进的情感识别准确率"""
        print("\n🧠 测试改进的情感识别准确率...")
        
        # 测试样本（与v1.20.0相同的样本）
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
                
                # 使用生产级情感识别模型
                result = await predict_emotion_production(
                    text=sample["text"],
                    context={"user_id": "test_user"}
                )
                
                processing_time = (time.time() - start_time) * 1000
                total_processing_time += processing_time
                
                # 检查预测结果
                predicted_emotion = result.emotion
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
            test_name="improved_emotion_recognition",
            duration=time.time(),
            success=accuracy >= 80,  # v1.20.1目标80%
            metrics={
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_samples": len(test_samples),
                "avg_processing_time": avg_processing_time,
                "avg_confidence": avg_confidence,
                "confidence_scores": confidence_scores,
                "model_version": production_emotion_model.model_version
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_cache_monitoring_system(self) -> PerformanceResult:
        """测试缓存监控系统"""
        print("\n📊 测试缓存监控系统...")
        
        # 模拟缓存操作
        cache_operations = [
            {"key": "user1:text:123", "type": "hit", "response_time": 0.01, "user": "user1", "request_type": "text_generation"},
            {"key": "user1:text:456", "type": "miss", "response_time": 0.05, "user": "user1", "request_type": "text_generation", "reason": "not_found"},
            {"key": "user2:voice:789", "type": "hit", "response_time": 0.02, "user": "user2", "request_type": "voice_synthesis"},
            {"key": "user2:voice:101", "type": "miss", "response_time": 0.08, "user": "user2", "request_type": "voice_synthesis", "reason": "expired"},
            {"key": "user3:emotion:202", "type": "hit", "response_time": 0.015, "user": "user3", "request_type": "emotion_analysis"},
        ]
        
        # 记录缓存操作
        for op in cache_operations:
            if op["type"] == "hit":
                cache_monitor.record_hit(
                    cache_key=op["key"],
                    response_time=op["response_time"],
                    cache_size=100,
                    user_id=op["user"],
                    request_type=op["request_type"]
                )
            else:
                cache_monitor.record_miss(
                    cache_key=op["key"],
                    response_time=op["response_time"],
                    cache_size=100,
                    user_id=op["user"],
                    request_type=op["request_type"],
                    reason=op["reason"]
                )
        
        # 获取缓存指标
        metrics = await get_cache_metrics()
        
        # 验证指标
        hit_rate = metrics["overall_metrics"]["hit_rate"]
        total_requests = metrics["overall_metrics"]["total_requests"]
        
        print(f"  总请求数: {total_requests}")
        print(f"  缓存命中率: {hit_rate:.2%}")
        print(f"  平均命中响应时间: {metrics['overall_metrics']['avg_hit_response_time']:.3f}s")
        print(f"  平均未命中响应时间: {metrics['overall_metrics']['avg_miss_response_time']:.3f}s")
        
        return PerformanceResult(
            test_name="cache_monitoring_system",
            duration=time.time(),
            success=hit_rate >= 0.4,  # 至少40%命中率
            metrics={
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "avg_hit_response_time": metrics["overall_metrics"]["avg_hit_response_time"],
                "avg_miss_response_time": metrics["overall_metrics"]["avg_miss_response_time"],
                "top_users": metrics["top_users"],
                "type_breakdown": metrics["type_breakdown"]
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_voice_latency_optimization(self) -> PerformanceResult:
        """测试语音延迟优化（与v1.20.0对比）"""
        print("\n🎤 测试语音延迟优化...")
        
        test_cases = [
            {"audio_length": 1, "expected_latency": 100},
            {"audio_length": 3, "expected_latency": 150},
            {"audio_length": 5, "expected_latency": 200},
            {"audio_length": 10, "expected_latency": 300},
        ]
        
        total_latency = 0
        passed_tests = 0
        latency_results = []
        
        for case in test_cases:
            audio_data = self._generate_test_audio(case["audio_length"])
            
            start_time = time.time()
            try:
                result = await self.voice_optimizer.optimize_voice_pipeline(audio_data)
                latency = (time.time() - start_time) * 1000
                
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
            success=success_rate >= 75,
            metrics={
                "average_latency": avg_latency,
                "success_rate": success_rate,
                "passed_tests": passed_tests,
                "total_tests": len(test_cases),
                "latency_results": latency_results
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def test_system_stability_improved(self) -> PerformanceResult:
        """测试改进的系统稳定性"""
        print("\n🔄 测试改进的系统稳定性...")
        
        test_requests = 50  # 减少测试请求数量以加快测试
        start_time = time.time()
        request_count = 0
        error_count = 0
        latencies = []
        
        print(f"  运行{test_requests}个请求的稳定性测试...")
        
        for i in range(test_requests):
            try:
                audio_data = self._generate_test_audio(1)
                request_start = time.time()
                
                # 语音优化
                voice_result = await self.voice_optimizer.optimize_voice_pipeline(audio_data)
                
                # 使用生产级情感识别
                emotion_result = await predict_emotion_production(
                    text=voice_result.text_response,
                    context={"user_id": "stability_test"}
                )
                
                request_latency = (time.time() - request_start) * 1000
                latencies.append(request_latency)
                request_count += 1
                
                if request_count % 10 == 0:
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
            test_name="system_stability_improved",
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
        return b"fake_audio_data_" + str(duration_seconds).encode() * 100
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有性能测试"""
        print("🚀 开始VoiceHelper v1.20.1性能测试")
        print("=" * 50)
        
        start_time = time.time()
        
        # 运行各项测试
        tests = [
            self.test_improved_emotion_recognition(),
            self.test_cache_monitoring_system(),
            self.test_voice_latency_optimization(),
            self.test_system_stability_improved()
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
            "version": "v1.20.1",
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
        print(f"🎯 v1.20.1测试完成！")
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
    tester = V121PerformanceTest()
    report = await tester.run_all_tests()
    
    # 保存测试报告
    report_file = f"v1_20_1_performance_results_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 测试报告已保存: {report_file}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())
