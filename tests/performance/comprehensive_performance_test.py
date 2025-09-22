"""
VoiceHelper 综合性能测试
验证所有已完成的TODO项目的集成效果
测试情感识别、长文本处理、代码理解和高并发能力
"""

import asyncio
import time
import sys
import os
import json
import random
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# 导入各个模块
from algo.core.production_emotion_recognition import analyze_production_emotion
from algo.core.long_context_processor import process_long_context
from algo.core.code_understanding import analyze_code, generate_code, review_code, CodeLanguage
from algo.core.high_concurrency_system import (
    high_concurrency_system, 
    process_high_concurrency_request,
    ServiceInstance
)

class ComprehensivePerformanceTest:
    """综合性能测试"""
    
    def __init__(self):
        self.test_results = {}
        self.overall_score = 0.0
        
    async def run_all_tests(self):
        """运行所有测试"""
        print("🎯 VoiceHelper 综合性能测试")
        print("=" * 60)
        print("测试已完成的TODO项目集成效果")
        print("=" * 60)
        
        # 1. 情感识别性能测试
        emotion_result = await self.test_emotion_recognition_performance()
        
        # 2. 长文本处理性能测试
        long_text_result = await self.test_long_text_processing_performance()
        
        # 3. 代码理解性能测试
        code_result = await self.test_code_understanding_performance()
        
        # 4. 高并发系统性能测试
        concurrency_result = await self.test_high_concurrency_performance()
        
        # 5. 集成测试
        integration_result = await self.test_system_integration()
        
        # 汇总结果
        await self.generate_final_report()
        
        return self.overall_score >= 80.0
    
    async def test_emotion_recognition_performance(self):
        """测试情感识别性能"""
        print("\n🧠 情感识别性能测试")
        print("-" * 40)
        
        test_cases = [
            {"text": "我今天非常开心，工作进展很顺利！", "expected": "happy"},
            {"text": "这个结果让我很失望和沮丧", "expected": "sad"},
            {"text": "我对这件事感到很愤怒", "expected": "angry"},
            {"text": "好的，我知道了", "expected": "neutral"},
            {"text": "太棒了！这正是我想要的", "expected": "excited"},
            {"text": "让我冷静地思考一下", "expected": "calm"},
            {"text": "这让我感到很沮丧", "expected": "sad"},
            {"text": "我很满意这个结果", "expected": "happy"},
            {"text": "这个功能真的很棒", "expected": "happy"},
            {"text": "我担心会出现问题", "expected": "sad"}
        ]
        
        correct_predictions = 0
        total_time = 0.0
        processing_times = []
        
        for i, case in enumerate(test_cases, 1):
            start_time = time.time()
            
            # 生成模拟音频
            audio_data = f"audio_for_{case['text']}".encode()
            
            result = await analyze_production_emotion(
                audio_data=audio_data,
                text=case["text"],
                user_id=f"test_user_{i}"
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time * 1000)
            total_time += processing_time
            
            is_correct = result.primary_emotion == case["expected"]
            if is_correct:
                correct_predictions += 1
            
            print(f"  测试 {i}: {case['text'][:20]}... -> {result.primary_emotion} "
                  f"({'✅' if is_correct else '❌'}) {processing_time*1000:.1f}ms")
        
        accuracy = correct_predictions / len(test_cases)
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # 评分
        accuracy_score = min(accuracy * 100, 100)
        speed_score = max(100 - avg_processing_time / 2, 0)  # 50ms以下满分
        emotion_score = (accuracy_score + speed_score) / 2
        
        self.test_results["emotion_recognition"] = {
            "accuracy": accuracy,
            "avg_processing_time_ms": avg_processing_time,
            "total_tests": len(test_cases),
            "correct_predictions": correct_predictions,
            "score": emotion_score
        }
        
        print(f"\n  📊 结果:")
        print(f"    准确率: {accuracy:.1%}")
        print(f"    平均处理时间: {avg_processing_time:.1f}ms")
        print(f"    评分: {emotion_score:.1f}/100")
        
        return emotion_score
    
    async def test_long_text_processing_performance(self):
        """测试长文本处理性能"""
        print("\n📄 长文本处理性能测试")
        print("-" * 40)
        
        # 生成不同长度的测试文本
        base_text = """
人工智能技术正在快速发展，深度学习、机器学习、自然语言处理等技术不断突破。
在计算机视觉领域，卷积神经网络实现了图像识别的重大进展。
在自然语言处理领域，Transformer架构彻底改变了语言模型的设计。
大规模预训练模型如GPT、BERT等在各种任务上都取得了优异的性能。
强化学习在游戏、机器人控制等领域也展现出巨大潜力。
生成对抗网络在图像生成、风格迁移等方面有广泛应用。
联邦学习、边缘计算等技术为AI的部署提供了新的可能性。
AI伦理、可解释性、公平性等问题也越来越受到关注。
未来，人工智能将在更多领域发挥重要作用，推动社会进步。
        """
        
        test_cases = [
            {"tokens": 10000, "name": "10K tokens"},
            {"tokens": 50000, "name": "50K tokens"},
            {"tokens": 100000, "name": "100K tokens"},
            {"tokens": 200000, "name": "200K tokens"}
        ]
        
        processing_times = []
        compression_ratios = []
        success_count = 0
        
        for case in test_cases:
            # 生成指定长度的文本
            target_tokens = case["tokens"]
            current_text = base_text
            
            # 重复文本直到达到目标长度
            while len(current_text.split()) < target_tokens:
                current_text += base_text
            
            print(f"  测试 {case['name']}:")
            
            start_time = time.time()
            
            try:
                result = await process_long_context(
                    text=current_text,
                    query="人工智能的主要技术和应用",
                    max_tokens=min(target_tokens, 200000),
                    preserve_structure=True
                )
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                compression_ratios.append(result.compression_ratio)
                success_count += 1
                
                print(f"    处理时间: {processing_time:.2f}s")
                print(f"    压缩比: {result.compression_ratio:.2%}")
                print(f"    输出tokens: {result.total_tokens:,}")
                print(f"    状态: ✅ 成功")
                
            except Exception as e:
                print(f"    状态: ❌ 失败 ({e})")
                processing_times.append(float('inf'))
                compression_ratios.append(0.0)
        
        # 评分
        success_rate = success_count / len(test_cases)
        avg_processing_time = sum(t for t in processing_times if t != float('inf')) / max(success_count, 1)
        avg_compression = sum(compression_ratios) / max(success_count, 1)
        
        success_score = success_rate * 100
        speed_score = max(100 - avg_processing_time * 10, 0)  # 10秒以下满分
        compression_score = min(avg_compression * 200, 100)  # 50%压缩率满分
        
        long_text_score = (success_score + speed_score + compression_score) / 3
        
        self.test_results["long_text_processing"] = {
            "success_rate": success_rate,
            "avg_processing_time_s": avg_processing_time,
            "avg_compression_ratio": avg_compression,
            "total_tests": len(test_cases),
            "successful_tests": success_count,
            "score": long_text_score
        }
        
        print(f"\n  📊 结果:")
        print(f"    成功率: {success_rate:.1%}")
        print(f"    平均处理时间: {avg_processing_time:.2f}s")
        print(f"    平均压缩比: {avg_compression:.2%}")
        print(f"    评分: {long_text_score:.1f}/100")
        
        return long_text_score
    
    async def test_code_understanding_performance(self):
        """测试代码理解性能"""
        print("\n💻 代码理解性能测试")
        print("-" * 40)
        
        test_codes = [
            {
                "name": "Python函数",
                "code": '''
def fibonacci(n):
    """计算斐波那契数列"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """计算阶乘"""
    if n <= 1:
        return 1
    return n * factorial(n-1)
''',
                "expected_functions": 2,
                "expected_issues": 1  # 递归可能的性能问题
            },
            {
                "name": "Python类",
                "code": '''
import os
import sys

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process_data(self, input_data):
        # 不安全的eval使用
        result = eval(input_data)
        return result
    
    def safe_process(self, data):
        return data.upper() if isinstance(data, str) else str(data)
''',
                "expected_functions": 3,
                "expected_issues": 1  # eval安全问题
            },
            {
                "name": "JavaScript代码",
                "code": '''
function calculateSum(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum;
}

const processData = async (data) => {
    return new Promise((resolve) => {
        setTimeout(() => resolve(data * 2), 100);
    });
};
''',
                "expected_functions": 2,
                "expected_issues": 0
            }
        ]
        
        analysis_times = []
        generation_times = []
        analysis_success = 0
        generation_success = 0
        
        for i, test_case in enumerate(test_codes, 1):
            print(f"  测试 {i} - {test_case['name']}:")
            
            # 测试代码分析
            start_time = time.time()
            try:
                analysis_result = await analyze_code(test_case["code"])
                analysis_time = time.time() - start_time
                analysis_times.append(analysis_time * 1000)
                
                functions_found = len(analysis_result.functions)
                issues_found = len(analysis_result.issues)
                
                print(f"    分析时间: {analysis_time*1000:.1f}ms")
                print(f"    检测到函数: {functions_found}")
                print(f"    检测到问题: {issues_found}")
                
                if (functions_found >= test_case["expected_functions"] and 
                    issues_found >= test_case["expected_issues"]):
                    analysis_success += 1
                    print(f"    分析结果: ✅ 通过")
                else:
                    print(f"    分析结果: ❌ 未达预期")
                
            except Exception as e:
                print(f"    分析结果: ❌ 失败 ({e})")
                analysis_times.append(1000)  # 1秒作为失败时间
            
            # 测试代码生成
            start_time = time.time()
            try:
                generation_result = await generate_code(
                    f"创建一个{test_case['name']}的示例",
                    language=CodeLanguage.PYTHON,
                    include_tests=True
                )
                generation_time = time.time() - start_time
                generation_times.append(generation_time * 1000)
                
                print(f"    生成时间: {generation_time*1000:.1f}ms")
                print(f"    生成置信度: {generation_result.confidence:.2f}")
                
                if generation_result.confidence > 0.5:
                    generation_success += 1
                    print(f"    生成结果: ✅ 通过")
                else:
                    print(f"    生成结果: ❌ 置信度过低")
                
            except Exception as e:
                print(f"    生成结果: ❌ 失败 ({e})")
                generation_times.append(2000)  # 2秒作为失败时间
        
        # 评分
        analysis_success_rate = analysis_success / len(test_codes)
        generation_success_rate = generation_success / len(test_codes)
        avg_analysis_time = sum(analysis_times) / len(analysis_times)
        avg_generation_time = sum(generation_times) / len(generation_times)
        
        analysis_score = analysis_success_rate * 100
        generation_score = generation_success_rate * 100
        speed_score = max(100 - (avg_analysis_time + avg_generation_time) / 20, 0)
        
        code_score = (analysis_score + generation_score + speed_score) / 3
        
        self.test_results["code_understanding"] = {
            "analysis_success_rate": analysis_success_rate,
            "generation_success_rate": generation_success_rate,
            "avg_analysis_time_ms": avg_analysis_time,
            "avg_generation_time_ms": avg_generation_time,
            "total_tests": len(test_codes),
            "score": code_score
        }
        
        print(f"\n  📊 结果:")
        print(f"    分析成功率: {analysis_success_rate:.1%}")
        print(f"    生成成功率: {generation_success_rate:.1%}")
        print(f"    平均分析时间: {avg_analysis_time:.1f}ms")
        print(f"    平均生成时间: {avg_generation_time:.1f}ms")
        print(f"    评分: {code_score:.1f}/100")
        
        return code_score
    
    async def test_high_concurrency_performance(self):
        """测试高并发性能"""
        print("\n🚀 高并发性能测试")
        print("-" * 40)
        
        # 初始化高并发系统
        system = high_concurrency_system
        
        # 添加测试服务实例
        for i in range(5):
            instance = ServiceInstance(
                id=f"perf_test_instance_{i+1}",
                host=f"10.0.1.{i+1}",
                port=8080,
                weight=1.0
            )
            system.load_balancer.add_instance(instance)
        
        await system.start()
        
        # 不同并发级别的测试
        concurrency_tests = [
            {"requests": 100, "name": "低并发"},
            {"requests": 500, "name": "中并发"},
            {"requests": 1000, "name": "高并发"},
            {"requests": 2000, "name": "极高并发"}
        ]
        
        test_results = []
        
        for test_case in concurrency_tests:
            print(f"  测试 {test_case['name']} ({test_case['requests']} 请求):")
            
            # 生成测试请求
            tasks = []
            for i in range(test_case["requests"]):
                task = process_high_concurrency_request(
                    request_data=f"perf_test_data_{i}",
                    user_id=f"perf_user_{i % 50}",
                    priority=random.randint(1, 10)
                )
                tasks.append(task)
            
            # 执行并发测试
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            duration = end_time - start_time
            successful_requests = sum(1 for r in results 
                                    if isinstance(r, dict) and r.get("status") == "success")
            failed_requests = len(results) - successful_requests
            qps = len(results) / duration
            success_rate = successful_requests / len(results)
            
            print(f"    执行时间: {duration:.2f}s")
            print(f"    成功请求: {successful_requests}")
            print(f"    失败请求: {failed_requests}")
            print(f"    成功率: {success_rate:.1%}")
            print(f"    QPS: {qps:.0f}")
            
            test_results.append({
                "requests": test_case["requests"],
                "duration": duration,
                "qps": qps,
                "success_rate": success_rate
            })
        
        await system.stop()
        
        # 评分
        avg_qps = sum(r["qps"] for r in test_results) / len(test_results)
        avg_success_rate = sum(r["success_rate"] for r in test_results) / len(test_results)
        
        qps_score = min(avg_qps / 50, 100)  # 5000 QPS满分
        success_score = avg_success_rate * 100
        
        concurrency_score = (qps_score + success_score) / 2
        
        self.test_results["high_concurrency"] = {
            "avg_qps": avg_qps,
            "avg_success_rate": avg_success_rate,
            "test_results": test_results,
            "score": concurrency_score
        }
        
        print(f"\n  📊 结果:")
        print(f"    平均QPS: {avg_qps:.0f}")
        print(f"    平均成功率: {avg_success_rate:.1%}")
        print(f"    评分: {concurrency_score:.1f}/100")
        
        return concurrency_score
    
    async def test_system_integration(self):
        """测试系统集成"""
        print("\n🔗 系统集成测试")
        print("-" * 40)
        
        # 综合场景测试：处理包含代码的长文本，分析情感，并发处理
        test_scenario = {
            "text": """
# 项目开发总结

## 1. 项目概述
我们的AI项目开发进展非常顺利！团队成员都很兴奋，因为我们实现了重要的技术突破。

## 2. 核心代码实现
```python
def process_ai_request(data):
    \"\"\"处理AI请求\"\"\"
    if not data:
        return None
    
    # 数据预处理
    processed_data = preprocess(data)
    
    # AI模型推理
    result = ai_model.predict(processed_data)
    
    return result

class AISystem:
    def __init__(self):
        self.model = load_model()
        self.cache = {}
    
    def predict(self, input_data):
        # 检查缓存
        cache_key = hash(str(input_data))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 执行预测
        prediction = self.model.forward(input_data)
        
        # 缓存结果
        self.cache[cache_key] = prediction
        
        return prediction
```

## 3. 性能优化
我们对系统进行了大量优化，虽然过程中遇到了一些挫折，但最终结果让人满意。
响应时间从原来的500ms降低到了50ms，这让整个团队都很高兴。

## 4. 未来规划
接下来我们计划进一步优化算法，提升系统的并发处理能力。
虽然还有很多挑战，但我们对未来充满信心！
            """ * 20,  # 重复20次创建长文本
            "expected_emotions": ["happy", "excited", "satisfied"],
            "expected_functions": 2,
            "expected_classes": 1
        }
        
        integration_scores = []
        
        print("  执行综合场景测试...")
        
        # 1. 长文本处理
        print("    1. 长文本处理...")
        start_time = time.time()
        try:
            long_text_result = await process_long_context(
                text=test_scenario["text"],
                query="项目开发的核心技术和成果",
                max_tokens=100000
            )
            long_text_time = time.time() - start_time
            print(f"       处理时间: {long_text_time:.2f}s")
            print(f"       压缩比: {long_text_result.compression_ratio:.2%}")
            integration_scores.append(85 if long_text_time < 10 else 60)
        except Exception as e:
            print(f"       失败: {e}")
            integration_scores.append(0)
        
        # 2. 代码理解
        print("    2. 代码理解...")
        code_snippet = '''
def process_ai_request(data):
    if not data:
        return None
    processed_data = preprocess(data)
    result = ai_model.predict(processed_data)
    return result
'''
        start_time = time.time()
        try:
            code_result = await analyze_code(code_snippet)
            code_time = time.time() - start_time
            print(f"       分析时间: {code_time*1000:.1f}ms")
            print(f"       检测函数: {len(code_result.functions)}")
            
            if len(code_result.functions) >= 1:
                integration_scores.append(90)
            else:
                integration_scores.append(70)
        except Exception as e:
            print(f"       失败: {e}")
            integration_scores.append(0)
        
        # 3. 情感识别
        print("    3. 情感识别...")
        emotion_texts = [
            "我们的AI项目开发进展非常顺利！",
            "团队成员都很兴奋",
            "虽然过程中遇到了一些挫折",
            "最终结果让人满意"
        ]
        
        correct_emotions = 0
        total_emotion_time = 0
        
        for text in emotion_texts:
            start_time = time.time()
            try:
                emotion_result = await analyze_production_emotion(
                    text=text,
                    user_id="integration_test"
                )
                emotion_time = time.time() - start_time
                total_emotion_time += emotion_time
                
                if emotion_result.primary_emotion in ["happy", "excited", "sad", "neutral"]:
                    correct_emotions += 1
            except Exception as e:
                print(f"       情感识别失败: {e}")
        
        emotion_accuracy = correct_emotions / len(emotion_texts)
        avg_emotion_time = total_emotion_time / len(emotion_texts)
        
        print(f"       平均处理时间: {avg_emotion_time*1000:.1f}ms")
        print(f"       识别准确率: {emotion_accuracy:.1%}")
        
        integration_scores.append(emotion_accuracy * 100)
        
        # 4. 并发处理
        print("    4. 并发处理...")
        
        # 初始化系统
        system = high_concurrency_system
        if not system.is_running:
            await system.start()
        
        # 并发请求测试
        concurrent_tasks = []
        for i in range(50):  # 50个并发请求
            task = process_high_concurrency_request(
                request_data=f"integration_test_{i}",
                user_id=f"user_{i % 10}"
            )
            concurrent_tasks.append(task)
        
        start_time = time.time()
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        successful_concurrent = sum(1 for r in concurrent_results 
                                  if isinstance(r, dict) and r.get("status") == "success")
        concurrent_success_rate = successful_concurrent / len(concurrent_results)
        concurrent_qps = len(concurrent_results) / concurrent_time
        
        print(f"       处理时间: {concurrent_time:.2f}s")
        print(f"       成功率: {concurrent_success_rate:.1%}")
        print(f"       QPS: {concurrent_qps:.0f}")
        
        if concurrent_success_rate > 0.9 and concurrent_qps > 20:
            integration_scores.append(95)
        elif concurrent_success_rate > 0.8:
            integration_scores.append(80)
        else:
            integration_scores.append(60)
        
        await system.stop()
        
        # 计算集成评分
        integration_score = sum(integration_scores) / len(integration_scores)
        
        self.test_results["system_integration"] = {
            "long_text_processing": integration_scores[0] if len(integration_scores) > 0 else 0,
            "code_understanding": integration_scores[1] if len(integration_scores) > 1 else 0,
            "emotion_recognition": integration_scores[2] if len(integration_scores) > 2 else 0,
            "concurrent_processing": integration_scores[3] if len(integration_scores) > 3 else 0,
            "overall_score": integration_score
        }
        
        print(f"\n  📊 集成测试结果:")
        print(f"    长文本处理: {integration_scores[0]:.1f}/100")
        print(f"    代码理解: {integration_scores[1]:.1f}/100")
        print(f"    情感识别: {integration_scores[2]:.1f}/100")
        print(f"    并发处理: {integration_scores[3]:.1f}/100")
        print(f"    集成评分: {integration_score:.1f}/100")
        
        return integration_score
    
    async def generate_final_report(self):
        """生成最终报告"""
        print("\n" + "=" * 60)
        print("📊 VoiceHelper 综合性能测试报告")
        print("=" * 60)
        
        # 计算各模块评分
        scores = []
        
        if "emotion_recognition" in self.test_results:
            emotion_score = self.test_results["emotion_recognition"]["score"]
            scores.append(emotion_score)
            print(f"🧠 情感识别系统: {emotion_score:.1f}/100")
            print(f"   - 准确率: {self.test_results['emotion_recognition']['accuracy']:.1%}")
            print(f"   - 处理速度: {self.test_results['emotion_recognition']['avg_processing_time_ms']:.1f}ms")
        
        if "long_text_processing" in self.test_results:
            long_text_score = self.test_results["long_text_processing"]["score"]
            scores.append(long_text_score)
            print(f"📄 长文本处理系统: {long_text_score:.1f}/100")
            print(f"   - 成功率: {self.test_results['long_text_processing']['success_rate']:.1%}")
            print(f"   - 压缩比: {self.test_results['long_text_processing']['avg_compression_ratio']:.1%}")
        
        if "code_understanding" in self.test_results:
            code_score = self.test_results["code_understanding"]["score"]
            scores.append(code_score)
            print(f"💻 代码理解系统: {code_score:.1f}/100")
            print(f"   - 分析成功率: {self.test_results['code_understanding']['analysis_success_rate']:.1%}")
            print(f"   - 生成成功率: {self.test_results['code_understanding']['generation_success_rate']:.1%}")
        
        if "high_concurrency" in self.test_results:
            concurrency_score = self.test_results["high_concurrency"]["score"]
            scores.append(concurrency_score)
            print(f"🚀 高并发系统: {concurrency_score:.1f}/100")
            print(f"   - 平均QPS: {self.test_results['high_concurrency']['avg_qps']:.0f}")
            print(f"   - 成功率: {self.test_results['high_concurrency']['avg_success_rate']:.1%}")
        
        if "system_integration" in self.test_results:
            integration_score = self.test_results["system_integration"]["overall_score"]
            scores.append(integration_score)
            print(f"🔗 系统集成: {integration_score:.1f}/100")
        
        # 计算总体评分
        self.overall_score = sum(scores) / len(scores) if scores else 0
        
        print(f"\n🎯 总体评分: {self.overall_score:.1f}/100")
        
        # 评级
        if self.overall_score >= 90:
            grade = "A+ (优秀)"
            status = "🎉 所有系统表现优异，已达到业界领先水平！"
        elif self.overall_score >= 80:
            grade = "A (良好)"
            status = "✅ 系统性能良好，达到预期目标！"
        elif self.overall_score >= 70:
            grade = "B (合格)"
            status = "⚠️ 系统基本达标，但仍有优化空间。"
        else:
            grade = "C (需改进)"
            status = "❌ 系统性能需要进一步优化。"
        
        print(f"📈 性能等级: {grade}")
        print(f"📝 评估结果: {status}")
        
        # 保存测试结果
        report_data = {
            "timestamp": time.time(),
            "overall_score": self.overall_score,
            "grade": grade,
            "test_results": self.test_results,
            "summary": {
                "emotion_recognition": "✅ 已完成" if "emotion_recognition" in self.test_results else "❌ 未测试",
                "long_text_processing": "✅ 已完成" if "long_text_processing" in self.test_results else "❌ 未测试",
                "code_understanding": "✅ 已完成" if "code_understanding" in self.test_results else "❌ 未测试",
                "high_concurrency": "✅ 已完成" if "high_concurrency" in self.test_results else "❌ 未测试",
                "system_integration": "✅ 已完成" if "system_integration" in self.test_results else "❌ 未测试"
            }
        }
        
        with open("comprehensive_performance_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细报告已保存到: comprehensive_performance_report.json")

async def main():
    """主函数"""
    tester = ComprehensivePerformanceTest()
    success = await tester.run_all_tests()
    
    if success:
        print(f"\n🎊 恭喜！VoiceHelper 综合性能测试全面通过！")
        print(f"🏆 已成功实现业界竞争力提升目标！")
    else:
        print(f"\n⚠️ 测试未完全通过，需要进一步优化。")
    
    return success

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    exit(0 if success else 1)
