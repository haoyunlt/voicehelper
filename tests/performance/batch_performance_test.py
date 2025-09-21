"""
批量化系统性能测试

测试批量化系统的吞吐量提升效果
"""

import asyncio
import time
import statistics
import json
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algo.services.batch_service import LLMBatchService, BatchingConfig
from algo.core.integrated_batch_system import IntegratedBatchSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """测试配置"""
    # 基础测试参数
    total_requests: int = 100
    concurrent_users: int = 10
    request_interval: float = 0.01  # 请求间隔
    
    # 批处理配置
    batch_size: int = 4
    max_wait_time: float = 0.1
    enable_batching: bool = True
    enable_merging: bool = True
    
    # 测试场景
    scenario: str = "mixed"  # mixed, translation, qa, similar
    
    # 输出配置
    output_file: Optional[str] = None
    verbose: bool = False


@dataclass
class TestResult:
    """测试结果"""
    scenario: str
    config: TestConfig
    
    # 性能指标
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float  # requests/second
    
    # 批处理统计
    batch_stats: Dict[str, Any]
    
    # 资源使用
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results: List[TestResult] = []
    
    async def run_test(self, test_name: str) -> TestResult:
        """运行单个测试"""
        logger.info(f"Starting test: {test_name}")
        
        # 创建批处理服务
        batch_config = BatchingConfig(
            initial_batch_size=self.config.batch_size,
            max_batch_size=self.config.batch_size * 2,
            max_wait_time=self.config.max_wait_time,
            enable_request_merging=self.config.enable_merging,
            enable_dynamic_adjustment=False  # 测试时禁用动态调整
        )
        
        service = LLMBatchService(config=batch_config) if self.config.enable_batching else None
        
        try:
            if service:
                await service.start()
            
            # 生成测试请求
            test_requests = self._generate_test_requests()
            
            # 执行测试
            start_time = time.time()
            response_times = []
            
            if self.config.enable_batching and service:
                # 使用批处理服务
                results = await self._run_batch_test(service, test_requests)
            else:
                # 直接调用 (无批处理)
                results = await self._run_direct_test(test_requests)
            
            total_time = time.time() - start_time
            
            # 计算响应时间
            for result in results:
                if hasattr(result, 'processing_time'):
                    response_times.append(result.processing_time)
                elif isinstance(result, dict) and 'processing_time' in result:
                    response_times.append(result['processing_time'])
            
            # 统计结果
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            # 计算性能指标
            avg_response_time = statistics.mean(response_times) if response_times else 0
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else avg_response_time
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else avg_response_time
            throughput = successful / total_time if total_time > 0 else 0
            
            # 获取批处理统计
            batch_stats = {}
            if service:
                batch_stats = await service.get_service_stats()
            
            result = TestResult(
                scenario=test_name,
                config=self.config,
                total_requests=len(test_requests),
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                avg_response_time=avg_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                throughput=throughput,
                batch_stats=batch_stats
            )
            
            self.results.append(result)
            logger.info(f"Test {test_name} completed: {successful}/{len(test_requests)} successful, {throughput:.2f} req/s")
            
            return result
            
        finally:
            if service:
                await service.stop()
    
    def _generate_test_requests(self) -> List[Dict[str, Any]]:
        """生成测试请求"""
        requests = []
        
        if self.config.scenario == "translation":
            # 翻译场景 - 高相似度
            base_texts = [
                "Hello world",
                "Good morning",
                "How are you",
                "Thank you",
                "Goodbye"
            ]
            
            for i in range(self.config.total_requests):
                text = base_texts[i % len(base_texts)]
                request = {
                    'messages': [
                        {"role": "user", "content": f"Translate '{text} {i}' to Chinese"}
                    ],
                    'model': 'gpt-3.5-turbo',
                    'max_tokens': 50
                }
                requests.append(request)
        
        elif self.config.scenario == "qa":
            # 问答场景 - 中等相似度
            questions = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What is the difference between AI and ML?",
                "Explain deep learning",
                "What are neural networks?"
            ]
            
            for i in range(self.config.total_requests):
                question = questions[i % len(questions)]
                request = {
                    'messages': [
                        {"role": "user", "content": f"{question} (question {i})"}
                    ],
                    'model': 'gpt-3.5-turbo',
                    'max_tokens': 200
                }
                requests.append(request)
        
        elif self.config.scenario == "similar":
            # 高相似度场景
            for i in range(self.config.total_requests):
                request = {
                    'messages': [
                        {"role": "user", "content": f"Hello, how are you today? (request {i})"}
                    ],
                    'model': 'gpt-3.5-turbo',
                    'max_tokens': 100
                }
                requests.append(request)
        
        else:  # mixed
            # 混合场景 - 低相似度
            templates = [
                "Translate '{}' to Spanish",
                "Summarize the following text: {}",
                "What is the meaning of '{}'?",
                "Write a poem about {}",
                "Explain the concept of {}",
                "Generate code for {}",
                "Analyze the sentiment of '{}'",
                "Create a story about {}"
            ]
            
            topics = [
                "artificial intelligence", "machine learning", "data science",
                "cloud computing", "blockchain", "quantum computing",
                "cybersecurity", "internet of things", "virtual reality",
                "augmented reality", "robotics", "automation"
            ]
            
            for i in range(self.config.total_requests):
                template = templates[i % len(templates)]
                topic = topics[i % len(topics)]
                content = template.format(topic)
                
                request = {
                    'messages': [
                        {"role": "user", "content": content}
                    ],
                    'model': 'gpt-3.5-turbo',
                    'max_tokens': 150
                }
                requests.append(request)
        
        return requests
    
    async def _run_batch_test(self, service: LLMBatchService, requests: List[Dict[str, Any]]) -> List[Any]:
        """运行批处理测试"""
        # 创建并发任务
        async def send_request(req: Dict[str, Any], delay: float):
            await asyncio.sleep(delay)
            return await service.chat_completion(**req)
        
        tasks = []
        for i, req in enumerate(requests):
            delay = (i // self.config.concurrent_users) * self.config.request_interval
            task = asyncio.create_task(send_request(req, delay))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _run_direct_test(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """运行直接测试 (无批处理)"""
        from algo.services.batch_service import MockLLMClient, BatchRequest
        
        client = MockLLMClient()
        
        async def send_request(req: Dict[str, Any], delay: float):
            await asyncio.sleep(delay)
            
            # 转换为BatchRequest格式
            batch_req = BatchRequest(
                messages=req['messages'],
                model=req.get('model', 'gpt-3.5-turbo'),
                max_tokens=req.get('max_tokens', 1000)
            )
            
            # 单个请求处理
            responses = await client.batch_chat_completion([batch_req])
            return responses[0] if responses else None
        
        tasks = []
        for i, req in enumerate(requests):
            delay = (i // self.config.concurrent_users) * self.config.request_interval
            task = asyncio.create_task(send_request(req, delay))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def run_comparison_test(self) -> Dict[str, TestResult]:
        """运行对比测试"""
        logger.info("Running comparison test: batch vs direct")
        
        results = {}
        
        # 测试批处理
        self.config.enable_batching = True
        batch_result = await self.run_test("batch_processing")
        results["batch"] = batch_result
        
        # 测试直接处理
        self.config.enable_batching = False
        direct_result = await self.run_test("direct_processing")
        results["direct"] = direct_result
        
        return results
    
    def print_results(self, results: Dict[str, TestResult]):
        """打印测试结果"""
        print("\n" + "="*80)
        print("PERFORMANCE TEST RESULTS")
        print("="*80)
        
        for name, result in results.items():
            print(f"\n{name.upper()} PROCESSING:")
            print(f"  Total Requests: {result.total_requests}")
            print(f"  Successful: {result.successful_requests}")
            print(f"  Failed: {result.failed_requests}")
            print(f"  Total Time: {result.total_time:.3f}s")
            print(f"  Throughput: {result.throughput:.2f} req/s")
            print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
            print(f"  P95 Response Time: {result.p95_response_time:.3f}s")
            print(f"  P99 Response Time: {result.p99_response_time:.3f}s")
            
            if result.batch_stats:
                print(f"  Batch Stats:")
                for key, value in result.batch_stats.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if 'rate' in key or 'efficiency' in key:
                            print(f"    {key}: {value:.2%}")
                        else:
                            print(f"    {key}: {value}")
        
        # 计算改进
        if "batch" in results and "direct" in results:
            batch_result = results["batch"]
            direct_result = results["direct"]
            
            throughput_improvement = (batch_result.throughput - direct_result.throughput) / direct_result.throughput
            latency_improvement = (direct_result.avg_response_time - batch_result.avg_response_time) / direct_result.avg_response_time
            
            print(f"\nIMPROVEMENT ANALYSIS:")
            print(f"  Throughput Improvement: {throughput_improvement:.1%}")
            print(f"  Latency Improvement: {latency_improvement:.1%}")
            
            if throughput_improvement > 0.3:
                print(f"  ✅ Throughput improvement target (30%+) achieved!")
            else:
                print(f"  ❌ Throughput improvement target (30%+) not achieved")
    
    def save_results(self, results: Dict[str, TestResult], filename: str):
        """保存测试结果"""
        output_data = {}
        
        for name, result in results.items():
            output_data[name] = {
                'scenario': result.scenario,
                'total_requests': result.total_requests,
                'successful_requests': result.successful_requests,
                'failed_requests': result.failed_requests,
                'total_time': result.total_time,
                'avg_response_time': result.avg_response_time,
                'p95_response_time': result.p95_response_time,
                'p99_response_time': result.p99_response_time,
                'throughput': result.throughput,
                'batch_stats': result.batch_stats,
                'config': {
                    'total_requests': result.config.total_requests,
                    'concurrent_users': result.config.concurrent_users,
                    'batch_size': result.config.batch_size,
                    'max_wait_time': result.config.max_wait_time,
                    'enable_batching': result.config.enable_batching,
                    'enable_merging': result.config.enable_merging,
                    'scenario': result.config.scenario
                }
            }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Batch processing performance test")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--scenario", choices=["mixed", "translation", "qa", "similar"], 
                       default="mixed", help="Test scenario")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # 创建测试配置
    config = TestConfig(
        total_requests=args.requests,
        concurrent_users=args.users,
        batch_size=args.batch_size,
        scenario=args.scenario,
        output_file=args.output,
        verbose=args.verbose
    )
    
    # 创建测试器
    tester = PerformanceTester(config)
    
    # 运行对比测试
    results = await tester.run_comparison_test()
    
    # 打印结果
    tester.print_results(results)
    
    # 保存结果
    if args.output:
        tester.save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
