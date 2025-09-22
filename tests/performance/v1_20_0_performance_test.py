"""
VoiceHelper v1.20.0 性能测试套件
测试语音优化、情感识别和批处理性能
"""

import asyncio
import time
import random
import math
from typing import List, Dict, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from algo.core.enhanced_voice_optimizer import (
    EnhancedVoiceOptimizer, 
    optimize_voice_request,
    VoiceResponse
)
from algo.core.advanced_emotion_recognition import (
    AdvancedEmotionRecognition,
    analyze_emotion,
    EmotionAnalysisResult
)
from algo.core.adaptive_batch_scheduler import (
    AdaptiveBatchScheduler,
    submit_batch_request,
    RequestType,
    RequestPriority
)

class V120PerformanceTest:
    """v1.20.0 性能测试套件"""
    
    def __init__(self):
        self.voice_optimizer = EnhancedVoiceOptimizer()
        self.emotion_recognizer = AdvancedEmotionRecognition()
        self.batch_scheduler = AdaptiveBatchScheduler()
        
        # 性能基线
        self.baseline_voice_latency = 300  # ms
        self.baseline_emotion_accuracy = 0.85
        self.baseline_batch_throughput = 10  # requests/second
        
        # 目标性能
        self.target_voice_latency = 150  # ms
        self.target_emotion_accuracy = 0.95
        self.target_batch_improvement = 2.0  # 200% improvement
    
    def generate_test_audio(self, duration_seconds: float) -> bytes:
        """生成测试音频数据"""
        # 模拟音频数据：44.1kHz, 16-bit, mono
        sample_rate = 44100
        samples = int(duration_seconds * sample_rate)
        
        # 生成简单的正弦波
        frequency = 440  # A4 note
        audio_data = []
        
        for i in range(samples):
            t = i / sample_rate
            sample = math.sin(2 * math.pi * frequency * t)
            # 转换为16-bit整数
            audio_int16 = int(sample * 32767)
            audio_data.extend([audio_int16 & 0xFF, (audio_int16 >> 8) & 0xFF])
        
        return bytes(audio_data)
    
    def create_test_request(self, request_type: RequestType = RequestType.TEXT_GENERATION) -> Dict[str, Any]:
        """创建测试请求"""
        return {
            "type": request_type,
            "data": f"test_data_{int(time.time()*1000)}",
            "user_id": f"test_user_{random.randint(1, 100)}",
            "priority": random.choice(list(RequestPriority))
        }
    
    async def test_voice_latency_optimization(self) -> Dict[str, Any]:
        """测试语音延迟优化"""
        print("\n=== 语音延迟优化测试 ===")
        
        test_cases = [
            {"audio_length": 1, "expected_latency": 100},   # 1秒音频，期望100ms延迟
            {"audio_length": 3, "expected_latency": 150},   # 3秒音频，期望150ms延迟
            {"audio_length": 5, "expected_latency": 200},   # 5秒音频，期望200ms延迟
            {"audio_length": 10, "expected_latency": 300},  # 10秒音频，期望300ms延迟
        ]
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n测试用例 {i+1}: {case['audio_length']}秒音频")
            
            # 生成测试音频
            test_audio = self.generate_test_audio(case["audio_length"])
            
            # 多次测试取平均值
            latencies = []
            for _ in range(5):
                start_time = time.time()
                result = await optimize_voice_request(test_audio, f"test_user_{i}")
                latency = (time.time() - start_time) * 1000  # 转换为毫秒
                latencies.append(latency)
            
            avg_latency = sum(latencies) / len(latencies)
            sorted_latencies = sorted(latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_index]
            
            # 验证性能
            passed = avg_latency <= case["expected_latency"]
            improvement = (self.baseline_voice_latency - avg_latency) / self.baseline_voice_latency * 100
            
            result_data = {
                "audio_length": case["audio_length"],
                "expected_latency": case["expected_latency"],
                "average_latency": avg_latency,
                "p95_latency": p95_latency,
                "improvement_percent": improvement,
                "passed": passed
            }
            
            results.append(result_data)
            
            print(f"  平均延迟: {avg_latency:.2f}ms")
            print(f"  P95延迟: {p95_latency:.2f}ms")
            print(f"  性能提升: {improvement:.1f}%")
            print(f"  测试结果: {'✅ 通过' if passed else '❌ 失败'}")
        
        # 计算总体性能
        latency_values = [r["average_latency"] for r in results]
        overall_avg_latency = sum(latency_values) / len(latency_values)
        overall_improvement = (self.baseline_voice_latency - overall_avg_latency) / self.baseline_voice_latency * 100
        overall_passed = overall_avg_latency <= self.target_voice_latency
        
        summary = {
            "test_name": "语音延迟优化测试",
            "results": results,
            "overall_average_latency": overall_avg_latency,
            "overall_improvement": overall_improvement,
            "target_achieved": overall_passed,
            "baseline_latency": self.baseline_voice_latency,
            "target_latency": self.target_voice_latency
        }
        
        print(f"\n总体结果:")
        print(f"  平均延迟: {overall_avg_latency:.2f}ms")
        print(f"  性能提升: {overall_improvement:.1f}%")
        print(f"  目标达成: {'✅ 是' if overall_passed else '❌ 否'}")
        
        return summary
    
    async def test_emotion_recognition_accuracy(self) -> Dict[str, Any]:
        """测试情感识别准确率"""
        print("\n=== 情感识别准确率测试 ===")
        
        # 测试数据集
        test_dataset = [
            {"text": "我今天非常开心，工作进展很顺利！", "expected": "happy"},
            {"text": "我很担心这个项目能不能按时完成", "expected": "sad"},
            {"text": "这个结果让我很失望", "expected": "sad"},
            {"text": "太棒了！这正是我想要的", "expected": "happy"},
            {"text": "我对此感到很愤怒", "expected": "angry"},
            {"text": "好的，我知道了", "expected": "neutral"},
            {"text": "我对这个新功能感到很兴奋", "expected": "excited"},
            {"text": "让我冷静地思考一下", "expected": "calm"},
            {"text": "这让我感到很沮丧", "expected": "sad"},
            {"text": "我很满意这个结果", "expected": "happy"},
        ]
        
        correct_predictions = 0
        total_predictions = len(test_dataset)
        processing_times = []
        confidence_scores = []
        
        print(f"测试数据集大小: {total_predictions}")
        
        for i, sample in enumerate(test_dataset):
            # 生成测试音频
            test_audio = self.generate_test_audio(2.0)  # 2秒音频
            
            start_time = time.time()
            result = await analyze_emotion(
                audio=test_audio,
                text=sample["text"],
                user_id=f"test_user_{i}"
            )
            processing_time = (time.time() - start_time) * 1000
            
            processing_times.append(processing_time)
            confidence_scores.append(result.confidence)
            
            # 检查预测准确性
            predicted_emotion = result.primary_emotion
            expected_emotion = sample["expected"]
            
            is_correct = predicted_emotion == expected_emotion
            if is_correct:
                correct_predictions += 1
            
            print(f"  样本 {i+1}: '{sample['text'][:30]}...'")
            print(f"    预期: {expected_emotion}, 预测: {predicted_emotion}")
            print(f"    置信度: {result.confidence:.2f}, 处理时间: {processing_time:.2f}ms")
            print(f"    结果: {'✅' if is_correct else '❌'}")
        
        # 计算指标
        accuracy = correct_predictions / total_predictions
        avg_processing_time = sum(processing_times) / len(processing_times)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # 性能评估
        accuracy_improvement = (accuracy - self.baseline_emotion_accuracy) / self.baseline_emotion_accuracy * 100
        target_achieved = accuracy >= self.target_emotion_accuracy
        
        summary = {
            "test_name": "情感识别准确率测试",
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "accuracy_improvement": accuracy_improvement,
            "average_processing_time": avg_processing_time,
            "average_confidence": avg_confidence,
            "target_achieved": target_achieved,
            "baseline_accuracy": self.baseline_emotion_accuracy,
            "target_accuracy": self.target_emotion_accuracy
        }
        
        print(f"\n总体结果:")
        print(f"  准确率: {accuracy:.2%}")
        print(f"  性能提升: {accuracy_improvement:.1f}%")
        print(f"  平均处理时间: {avg_processing_time:.2f}ms")
        print(f"  平均置信度: {avg_confidence:.2f}")
        print(f"  目标达成: {'✅ 是' if target_achieved else '❌ 否'}")
        
        return summary
    
    async def test_batch_processing_throughput(self) -> Dict[str, Any]:
        """测试批处理吞吐量"""
        print("\n=== 批处理吞吐量测试 ===")
        
        # 启动批处理调度器
        await self.batch_scheduler.start()
        
        try:
            # 测试不同批大小的性能
            batch_sizes = [10, 25, 50, 100, 200]
            results = []
            
            for batch_size in batch_sizes:
                print(f"\n测试批大小: {batch_size}")
                
                # 生成测试请求
                requests = []
                for i in range(batch_size):
                    request_data = self.create_test_request()
                    requests.append(request_data)
                
                # 提交请求并测量时间
                start_time = time.time()
                
                tasks = []
                for req_data in requests:
                    task = submit_batch_request(
                        request_type=req_data["type"],
                        data=req_data["data"],
                        user_id=req_data["user_id"],
                        priority=req_data["priority"]
                    )
                    tasks.append(task)
                
                # 等待所有请求提交
                request_ids = await asyncio.gather(*tasks)
                
                # 等待处理完成
                await asyncio.sleep(2.0)  # 给足够时间处理
                
                end_time = time.time()
                total_time = end_time - start_time
                throughput = batch_size / total_time
                
                # 获取调度器统计
                stats = self.batch_scheduler.get_statistics()
                
                result_data = {
                    "batch_size": batch_size,
                    "total_time": total_time,
                    "throughput": throughput,
                    "average_batch_size": stats.get("average_batch_size", 0),
                    "average_wait_time": stats.get("average_wait_time_ms", 0),
                    "queue_size": stats.get("queue_size", 0)
                }
                
                results.append(result_data)
                
                print(f"  处理时间: {total_time:.2f}s")
                print(f"  吞吐量: {throughput:.2f} req/s")
                print(f"  平均批大小: {result_data['average_batch_size']:.1f}")
                print(f"  平均等待时间: {result_data['average_wait_time']:.2f}ms")
            
            # 计算性能提升
            max_throughput = max(r["throughput"] for r in results)
            throughput_improvement = (max_throughput - self.baseline_batch_throughput) / self.baseline_batch_throughput
            target_achieved = throughput_improvement >= (self.target_batch_improvement - 1)
            
            summary = {
                "test_name": "批处理吞吐量测试",
                "results": results,
                "max_throughput": max_throughput,
                "throughput_improvement": throughput_improvement * 100,
                "target_achieved": target_achieved,
                "baseline_throughput": self.baseline_batch_throughput,
                "target_improvement": self.target_batch_improvement * 100
            }
            
            print(f"\n总体结果:")
            print(f"  最大吞吐量: {max_throughput:.2f} req/s")
            print(f"  性能提升: {throughput_improvement*100:.1f}%")
            print(f"  目标达成: {'✅ 是' if target_achieved else '❌ 否'}")
            
            return summary
            
        finally:
            # 停止调度器
            await self.batch_scheduler.stop()
    
    async def test_system_stability(self) -> Dict[str, Any]:
        """测试系统稳定性"""
        print("\n=== 系统稳定性测试 ===")
        
        # 启动批处理调度器
        await self.batch_scheduler.start()
        
        try:
            # 长时间运行测试
            test_duration = 30  # 30秒
            request_interval = 0.1  # 100ms间隔
            
            start_time = time.time()
            request_count = 0
            error_count = 0
            latencies = []
            
            print(f"运行稳定性测试 {test_duration} 秒...")
            
            while time.time() - start_time < test_duration:
                try:
                    # 随机选择测试类型
                    test_type = random.choice(["voice", "emotion", "batch"])
                    
                    if test_type == "voice":
                        # 语音处理测试
                        test_audio = self.generate_test_audio(1.0)
                        req_start = time.time()
                        await optimize_voice_request(test_audio, f"stability_user_{request_count}")
                        latency = (time.time() - req_start) * 1000
                        latencies.append(latency)
                        
                    elif test_type == "emotion":
                        # 情感识别测试
                        test_audio = self.generate_test_audio(1.0)
                        req_start = time.time()
                        await analyze_emotion(
                            audio=test_audio,
                            text="稳定性测试文本",
                            user_id=f"stability_user_{request_count}"
                        )
                        latency = (time.time() - req_start) * 1000
                        latencies.append(latency)
                        
                    else:
                        # 批处理测试
                        req_start = time.time()
                        await submit_batch_request(
                            request_type=RequestType.TEXT_GENERATION,
                            data=f"stability_test_{request_count}",
                            user_id=f"stability_user_{request_count}",
                            priority=RequestPriority.NORMAL
                        )
                        latency = (time.time() - req_start) * 1000
                        latencies.append(latency)
                    
                    request_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"  错误 {error_count}: {str(e)[:50]}...")
                
                # 控制请求频率
                await asyncio.sleep(request_interval)
            
            # 计算稳定性指标
            total_time = time.time() - start_time
            success_rate = (request_count - error_count) / request_count if request_count > 0 else 0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            if latencies:
                sorted_latencies = sorted(latencies)
                p95_index = int(len(sorted_latencies) * 0.95)
                p95_latency = sorted_latencies[p95_index]
            else:
                p95_latency = 0
            throughput = request_count / total_time
            
            # 获取最终统计
            final_stats = self.batch_scheduler.get_statistics()
            
            summary = {
                "test_name": "系统稳定性测试",
                "test_duration": total_time,
                "total_requests": request_count,
                "error_count": error_count,
                "success_rate": success_rate,
                "average_latency": avg_latency,
                "p95_latency": p95_latency,
                "throughput": throughput,
                "scheduler_stats": final_stats
            }
            
            print(f"\n稳定性测试结果:")
            print(f"  测试时长: {total_time:.1f}s")
            print(f"  总请求数: {request_count}")
            print(f"  错误数量: {error_count}")
            print(f"  成功率: {success_rate:.2%}")
            print(f"  平均延迟: {avg_latency:.2f}ms")
            print(f"  P95延迟: {p95_latency:.2f}ms")
            print(f"  吞吐量: {throughput:.2f} req/s")
            
            return summary
            
        finally:
            await self.batch_scheduler.stop()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有性能测试"""
        print("🚀 开始 VoiceHelper v1.20.0 性能测试")
        print("=" * 60)
        
        all_results = {}
        
        try:
            # 1. 语音延迟优化测试
            voice_results = await self.test_voice_latency_optimization()
            all_results["voice_optimization"] = voice_results
            
            # 2. 情感识别准确率测试
            emotion_results = await self.test_emotion_recognition_accuracy()
            all_results["emotion_recognition"] = emotion_results
            
            # 3. 批处理吞吐量测试
            batch_results = await self.test_batch_processing_throughput()
            all_results["batch_processing"] = batch_results
            
            # 4. 系统稳定性测试
            stability_results = await self.test_system_stability()
            all_results["system_stability"] = stability_results
            
            # 生成总体评估
            overall_assessment = self._generate_overall_assessment(all_results)
            all_results["overall_assessment"] = overall_assessment
            
            print("\n" + "=" * 60)
            print("📊 v1.20.0 性能测试总结")
            print("=" * 60)
            
            for test_name, result in overall_assessment.items():
                if test_name != "overall_score":
                    status = "✅ 通过" if result["passed"] else "❌ 失败"
                    print(f"{result['name']}: {status}")
                    print(f"  目标: {result['target']}")
                    print(f"  实际: {result['actual']}")
                    print(f"  提升: {result['improvement']}")
            
            overall_score = overall_assessment["overall_score"]
            print(f"\n🎯 总体评分: {overall_score:.1f}/100")
            
            if overall_score >= 80:
                print("🎉 v1.20.0 性能测试全面通过！")
            elif overall_score >= 60:
                print("⚠️  v1.20.0 性能基本达标，部分指标需要优化")
            else:
                print("❌ v1.20.0 性能测试未达标，需要重大优化")
            
            return all_results
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            return {"error": str(e)}
    
    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成总体评估"""
        assessment = {}
        
        # 语音优化评估
        voice_result = results.get("voice_optimization", {})
        voice_passed = voice_result.get("target_achieved", False)
        voice_improvement = voice_result.get("overall_improvement", 0)
        
        assessment["voice_optimization"] = {
            "name": "语音延迟优化",
            "passed": voice_passed,
            "target": f"< {self.target_voice_latency}ms",
            "actual": f"{voice_result.get('overall_average_latency', 0):.1f}ms",
            "improvement": f"{voice_improvement:.1f}%",
            "score": 25 if voice_passed else 10
        }
        
        # 情感识别评估
        emotion_result = results.get("emotion_recognition", {})
        emotion_passed = emotion_result.get("target_achieved", False)
        emotion_accuracy = emotion_result.get("accuracy", 0)
        
        assessment["emotion_recognition"] = {
            "name": "情感识别准确率",
            "passed": emotion_passed,
            "target": f"> {self.target_emotion_accuracy:.0%}",
            "actual": f"{emotion_accuracy:.1%}",
            "improvement": f"{emotion_result.get('accuracy_improvement', 0):.1f}%",
            "score": 25 if emotion_passed else 10
        }
        
        # 批处理评估
        batch_result = results.get("batch_processing", {})
        batch_passed = batch_result.get("target_achieved", False)
        batch_improvement = batch_result.get("throughput_improvement", 0)
        
        assessment["batch_processing"] = {
            "name": "批处理吞吐量",
            "passed": batch_passed,
            "target": f"> {self.target_batch_improvement*100:.0f}% 提升",
            "actual": f"{batch_improvement:.1f}% 提升",
            "improvement": f"{batch_improvement:.1f}%",
            "score": 25 if batch_passed else 10
        }
        
        # 稳定性评估
        stability_result = results.get("system_stability", {})
        stability_success_rate = stability_result.get("success_rate", 0)
        stability_passed = stability_success_rate >= 0.99  # 99%成功率
        
        assessment["system_stability"] = {
            "name": "系统稳定性",
            "passed": stability_passed,
            "target": "> 99% 成功率",
            "actual": f"{stability_success_rate:.1%}",
            "improvement": f"{(stability_success_rate - 0.95) * 100:.1f}%",
            "score": 25 if stability_passed else 10
        }
        
        # 计算总分
        total_score = sum(item["score"] for item in assessment.values() if isinstance(item, dict) and "score" in item)
        assessment["overall_score"] = total_score
        
        return assessment

# 测试运行器
async def main():
    """主测试函数"""
    test_suite = V120PerformanceTest()
    results = await test_suite.run_all_tests()
    
    # 保存测试结果
    import json
    with open("v1_20_0_performance_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 测试结果已保存到: v1_20_0_performance_results.json")
    
    return results

if __name__ == "__main__":
    # 运行性能测试
    asyncio.run(main())
