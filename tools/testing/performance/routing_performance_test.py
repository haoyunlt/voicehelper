"""
动态模型路由性能测试

测试智能路由系统的成本降低效果
"""

import asyncio
import time
import statistics
import sys
import os
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algo.services.routing_service import IntelligentRoutingService, RoutingConfig


class RoutingPerformanceTester:
    """路由性能测试器"""
    
    def __init__(self):
        self.test_results = {}
    
    async def create_mock_processors(self) -> dict:
        """创建模拟处理器"""
        
        # 模拟不同模型的成本和性能特征
        model_configs = {
            'gpt-4': {
                'cost_per_1k_tokens': 0.03,
                'base_latency': 2.0,
                'quality_score': 0.95
            },
            'gpt-3.5-turbo': {
                'cost_per_1k_tokens': 0.002,
                'base_latency': 1.0,
                'quality_score': 0.85
            },
            'claude-3-haiku': {
                'cost_per_1k_tokens': 0.00025,
                'base_latency': 0.8,
                'quality_score': 0.75
            },
            'gemini-pro': {
                'cost_per_1k_tokens': 0.001,
                'base_latency': 1.2,
                'quality_score': 0.80
            }
        }
        
        async def create_processor(model_name: str, config: dict):
            async def processor(content: str, params: dict) -> str:
                # 模拟处理延迟
                latency = config['base_latency'] * random.uniform(0.8, 1.2)
                await asyncio.sleep(latency)
                
                # 模拟响应质量
                quality_indicator = "★" * int(config['quality_score'] * 5)
                
                return f"[{model_name}] {quality_indicator} Response to: {content[:50]}..."
            
            return processor
        
        processors = {}
        for model_name, config in model_configs.items():
            processors[model_name] = await create_processor(model_name, config)
        
        return processors
    
    def generate_test_scenarios(self) -> list:
        """生成测试场景"""
        scenarios = []
        
        # 场景1: 混合复杂度任务 (真实场景)
        mixed_tasks = [
            # 简单任务 (40%)
            "Hello", "Thanks", "Yes", "What is AI?", "Define ML",
            "Hi there", "Good morning", "How are you?", "What's new?", "Tell me a joke",
            
            # 中等任务 (40%)
            "How does machine learning work?", "Explain neural networks",
            "Compare Python and Java", "What are the benefits of cloud computing?",
            "How to optimize database performance?", "Explain REST API design",
            "What is microservices architecture?", "How does blockchain work?",
            
            # 复杂任务 (20%)
            "Design a scalable distributed system for real-time analytics",
            "Create a comprehensive AI strategy for enterprise transformation",
            "Analyze the economic impact of automation on employment",
            "Develop an algorithm for optimizing supply chain logistics",
            "Write a detailed technical specification for a recommendation engine"
        ]
        
        scenarios.append({
            'name': 'mixed_complexity',
            'description': '混合复杂度场景 (模拟真实使用)',
            'tasks': mixed_tasks,
            'expected_cost_reduction': 0.35  # 期望35%成本降低
        })
        
        # 场景2: 高比例简单任务 (客服场景)
        simple_heavy_tasks = [
            "Hello", "Hi", "Thanks", "Bye", "Yes", "No", "OK", "Help",
            "What is your name?", "How can I contact support?", "What are your hours?",
            "Where is my order?", "How to reset password?", "What is the price?",
            "Is this available?", "How to cancel?", "What is the refund policy?",
            "How to login?", "Forgot password", "Account locked", "Payment failed"
        ] * 3  # 重复以增加简单任务比例
        
        scenarios.append({
            'name': 'simple_heavy',
            'description': '简单任务为主场景 (客服/FAQ)',
            'tasks': simple_heavy_tasks,
            'expected_cost_reduction': 0.50  # 期望50%成本降低
        })
        
        # 场景3: 高比例复杂任务 (研发场景)
        complex_heavy_tasks = [
            "Design a distributed microservices architecture",
            "Implement a machine learning pipeline for real-time predictions",
            "Create a comprehensive security framework for cloud applications",
            "Develop an optimization algorithm for resource allocation",
            "Build a data processing system for petabyte-scale analytics",
            "Design a fault-tolerant messaging system",
            "Implement a recommendation engine with collaborative filtering",
            "Create a real-time fraud detection system",
            "Design a scalable search engine architecture",
            "Develop a distributed caching strategy"
        ] * 2
        
        scenarios.append({
            'name': 'complex_heavy',
            'description': '复杂任务为主场景 (研发/咨询)',
            'tasks': complex_heavy_tasks,
            'expected_cost_reduction': 0.20  # 期望20%成本降低
        })
        
        return scenarios
    
    async def run_baseline_test(self, tasks: list, processors: dict) -> dict:
        """运行基线测试 (总是使用最贵的模型)"""
        print("🔄 Running baseline test (always GPT-4)...")
        
        baseline_model = 'gpt-4'
        processor = processors[baseline_model]
        
        total_cost = 0.0
        response_times = []
        start_time = time.time()
        
        for task in tasks:
            task_start = time.time()
            
            # 模拟成本计算
            estimated_tokens = len(task) * 2.5  # 简化估算
            task_cost = (estimated_tokens / 1000) * 0.03  # GPT-4价格
            total_cost += task_cost
            
            # 执行处理
            await processor(task, {'model': baseline_model})
            
            response_time = time.time() - task_start
            response_times.append(response_time)
        
        total_time = time.time() - start_time
        
        return {
            'total_cost': total_cost,
            'total_time': total_time,
            'avg_response_time': statistics.mean(response_times),
            'throughput': len(tasks) / total_time,
            'model_usage': {baseline_model: len(tasks)}
        }
    
    async def run_routing_test(self, tasks: list, processors: dict, config: RoutingConfig) -> dict:
        """运行智能路由测试"""
        print("🚀 Running intelligent routing test...")
        
        # 创建路由服务
        routing_service = IntelligentRoutingService(
            model_processors=processors,
            config=config
        )
        
        await routing_service.start()
        
        try:
            total_cost = 0.0
            response_times = []
            model_usage = {}
            successful_requests = 0
            start_time = time.time()
            
            for i, task in enumerate(tasks):
                task_start = time.time()
                
                # 模拟不同用户
                user_id = f"user_{i % 10}"
                
                # 执行路由处理
                result = await routing_service.route_and_process(
                    content=task,
                    user_id=user_id
                )
                
                response_time = time.time() - task_start
                response_times.append(response_time)
                
                if result['success']:
                    successful_requests += 1
                    routing_info = result['routing_info']
                    
                    # 累计成本
                    total_cost += routing_info['estimated_cost']
                    
                    # 统计模型使用
                    selected_model = routing_info['selected_model']
                    model_usage[selected_model] = model_usage.get(selected_model, 0) + 1
                
                # 每20个任务输出进度
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(tasks)}")
            
            total_time = time.time() - start_time
            
            # 获取详细统计
            service_stats = routing_service.get_comprehensive_stats()
            
            return {
                'total_cost': total_cost,
                'total_time': total_time,
                'avg_response_time': statistics.mean(response_times),
                'throughput': len(tasks) / total_time,
                'model_usage': model_usage,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / len(tasks),
                'service_stats': service_stats
            }
            
        finally:
            await routing_service.stop()
    
    def analyze_cost_reduction(self, baseline: dict, routing: dict) -> dict:
        """分析成本降低效果"""
        cost_reduction = (baseline['total_cost'] - routing['total_cost']) / baseline['total_cost']
        
        # 性能对比
        performance_impact = (routing['avg_response_time'] - baseline['avg_response_time']) / baseline['avg_response_time']
        
        # 吞吐量对比
        throughput_change = (routing['throughput'] - baseline['throughput']) / baseline['throughput']
        
        return {
            'cost_reduction': cost_reduction,
            'performance_impact': performance_impact,
            'throughput_change': throughput_change,
            'cost_savings': baseline['total_cost'] - routing['total_cost'],
            'baseline_cost': baseline['total_cost'],
            'routing_cost': routing['total_cost']
        }
    
    async def run_comprehensive_test(self):
        """运行综合性能测试"""
        print("🚀 Dynamic Model Routing Performance Test")
        print("=" * 70)
        
        # 创建模拟处理器
        processors = await self.create_mock_processors()
        
        # 生成测试场景
        scenarios = self.generate_test_scenarios()
        
        # 路由配置
        config = RoutingConfig(
            cost_priority=0.7,  # 70%优先考虑成本
            daily_budget=100.0,
            monthly_budget=2000.0,
            per_user_budget=10.0,
            enable_auto_optimization=True
        )
        
        for scenario in scenarios:
            print(f"\n📊 {scenario['description']}")
            print("-" * 50)
            
            tasks = scenario['tasks']
            print(f"Task count: {len(tasks)}")
            
            # 运行基线测试
            baseline_results = await self.run_baseline_test(tasks, processors)
            
            # 运行路由测试
            routing_results = await self.run_routing_test(tasks, processors, config)
            
            # 分析结果
            analysis = self.analyze_cost_reduction(baseline_results, routing_results)
            
            # 输出结果
            self._print_scenario_results(
                scenario['name'],
                scenario['description'],
                baseline_results,
                routing_results,
                analysis,
                scenario['expected_cost_reduction']
            )
            
            # 保存结果
            self.test_results[scenario['name']] = {
                'baseline': baseline_results,
                'routing': routing_results,
                'analysis': analysis,
                'expected_reduction': scenario['expected_cost_reduction']
            }
        
        # 输出总结
        self._print_summary()
    
    def _print_scenario_results(
        self,
        scenario_name: str,
        description: str,
        baseline: dict,
        routing: dict,
        analysis: dict,
        expected_reduction: float
    ):
        """输出场景测试结果"""
        print(f"\n📈 {description} 结果:")
        
        print(f"\n  基线测试 (总是使用GPT-4):")
        print(f"    总成本: ${baseline['total_cost']:.4f}")
        print(f"    平均响应时间: {baseline['avg_response_time']:.3f}s")
        print(f"    吞吐量: {baseline['throughput']:.2f} req/s")
        print(f"    总耗时: {baseline['total_time']:.3f}s")
        
        print(f"\n  智能路由测试:")
        print(f"    总成本: ${routing['total_cost']:.4f}")
        print(f"    平均响应时间: {routing['avg_response_time']:.3f}s")
        print(f"    吞吐量: {routing['throughput']:.2f} req/s")
        print(f"    总耗时: {routing['total_time']:.3f}s")
        print(f"    成功率: {routing['success_rate']:.2%}")
        
        print(f"\n  模型使用分布:")
        for model, count in routing['model_usage'].items():
            percentage = count / sum(routing['model_usage'].values()) * 100
            print(f"    {model}: {count} ({percentage:.1f}%)")
        
        print(f"\n  🎯 成本优化效果:")
        print(f"    成本降低: {analysis['cost_reduction']:.1%}")
        print(f"    成本节省: ${analysis['cost_savings']:.4f}")
        print(f"    性能影响: {analysis['performance_impact']:.1%}")
        print(f"    吞吐量变化: {analysis['throughput_change']:.1%}")
        
        # 检查是否达到目标
        if analysis['cost_reduction'] >= expected_reduction:
            print(f"    ✅ 达到预期{expected_reduction:.0%}成本降低目标!")
        elif analysis['cost_reduction'] >= expected_reduction * 0.8:
            print(f"    🟡 接近目标，达到{analysis['cost_reduction']:.1%}成本降低")
        else:
            print(f"    ❌ 未达到{expected_reduction:.0%}成本降低目标")
    
    def _print_summary(self):
        """输出测试总结"""
        print(f"\n🎯 测试总结")
        print("=" * 70)
        
        total_cost_reductions = []
        scenarios_meeting_target = 0
        
        for scenario_name, results in self.test_results.items():
            cost_reduction = results['analysis']['cost_reduction']
            expected_reduction = results['expected_reduction']
            
            total_cost_reductions.append(cost_reduction)
            
            if cost_reduction >= expected_reduction:
                scenarios_meeting_target += 1
        
        avg_cost_reduction = statistics.mean(total_cost_reductions)
        
        print(f"平均成本降低: {avg_cost_reduction:.1%}")
        print(f"达到预期目标的场景: {scenarios_meeting_target}/{len(self.test_results)}")
        
        if avg_cost_reduction >= 0.30:
            print(f"\n🏆 动态模型路由系统成功实现30-50%成本降低目标!")
        elif avg_cost_reduction >= 0.20:
            print(f"\n🟡 接近目标，平均降低{avg_cost_reduction:.1%}成本")
        else:
            print(f"\n❌ 未达到30%成本降低目标")
        
        # 最佳场景分析
        best_scenario = max(
            self.test_results.items(),
            key=lambda x: x[1]['analysis']['cost_reduction']
        )
        
        print(f"\n📊 最佳场景: {best_scenario[0]}")
        print(f"   成本降低: {best_scenario[1]['analysis']['cost_reduction']:.1%}")
        print(f"   模型分布: {best_scenario[1]['routing']['model_usage']}")
        
        # 路由策略效果分析
        print(f"\n💡 路由策略效果分析:")
        print(f"   简单任务场景: 成本降低最显著，适合客服/FAQ应用")
        print(f"   混合任务场景: 平衡成本和性能，适合通用聊天应用")
        print(f"   复杂任务场景: 保证质量前提下适度降本，适合专业咨询")
        
        # 成本优化建议
        print(f"\n🔧 成本优化建议:")
        print(f"   1. 针对简单查询优先使用轻量模型")
        print(f"   2. 复杂任务保持高质量模型以确保效果")
        print(f"   3. 结合缓存系统进一步降低重复查询成本")
        print(f"   4. 根据用户等级动态调整模型选择策略")


async def main():
    """主函数"""
    tester = RoutingPerformanceTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())
