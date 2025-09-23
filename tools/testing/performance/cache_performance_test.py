"""
缓存系统性能测试

测试智能缓存系统的响应速度提升效果
"""

import asyncio
import time
import statistics
import sys
import os
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algo.services.cache_service import IntegratedCacheService, CacheConfig


class CachePerformanceTester:
    """缓存性能测试器"""
    
    def __init__(self):
        self.test_results = {}
    
    async def mock_llm_processing(self, content: str, model: str, parameters: dict) -> str:
        """模拟LLM处理"""
        # 模拟不同类型查询的处理时间
        base_time = 0.8  # 800ms基础处理时间
        
        # 根据内容长度调整处理时间
        length_factor = len(content) / 100.0
        processing_time = base_time * (0.8 + 0.4 * length_factor)
        
        # 添加随机变化
        processing_time *= random.uniform(0.9, 1.1)
        
        await asyncio.sleep(processing_time)
        
        return f"AI response to: {content}"
    
    def generate_test_queries(self, scenario: str, count: int) -> list:
        """生成测试查询"""
        if scenario == "high_similarity":
            # 高相似度场景
            base_queries = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Explain deep learning concepts",
                "What are neural networks?",
                "Tell me about natural language processing"
            ]
            
            queries = []
            for i in range(count):
                base_query = base_queries[i % len(base_queries)]
                # 创建相似变体
                variations = [
                    base_query,
                    base_query.replace("?", " in detail?"),
                    f"Can you {base_query.lower()}",
                    f"Please {base_query.lower()}",
                    base_query.replace("What", "What exactly")
                ]
                queries.append(variations[i % len(variations)])
            
            return queries
        
        elif scenario == "mixed_similarity":
            # 混合相似度场景
            queries = [
                "What is AI?", "How does ML work?", "Explain DL",
                "What is artificial intelligence?", "How does machine learning work?",
                "Tell me about Python programming", "What is Docker?",
                "Explain REST APIs", "How to use Git?", "What is microservices?",
                "What's AI?", "How ML works?", "Deep learning explanation",
                "AI definition", "Machine learning basics", "Neural network intro"
            ]
            return (queries * (count // len(queries) + 1))[:count]
        
        else:  # low_similarity
            # 低相似度场景
            topics = [
                "artificial intelligence", "machine learning", "data science",
                "cloud computing", "blockchain technology", "quantum computing",
                "cybersecurity", "internet of things", "virtual reality",
                "augmented reality", "robotics", "automation", "big data",
                "edge computing", "5G networks", "digital transformation"
            ]
            
            templates = [
                "What is {}?", "How does {} work?", "Explain {} in detail",
                "Tell me about {}", "What are the benefits of {}?",
                "How to implement {}?", "What are {} applications?",
                "Compare {} with alternatives", "{} best practices",
                "Future of {}", "{} challenges and solutions"
            ]
            
            queries = []
            for i in range(count):
                topic = topics[i % len(topics)]
                template = templates[i % len(templates)]
                queries.append(template.format(topic))
            
            return queries
    
    async def run_baseline_test(self, queries: list) -> dict:
        """运行基线测试 (无缓存)"""
        print("🔄 Running baseline test (no cache)...")
        
        response_times = []
        start_time = time.time()
        
        for query in queries:
            query_start = time.time()
            await self.mock_llm_processing(query, "gpt-3.5-turbo", {"temperature": 0.7})
            response_time = time.time() - query_start
            response_times.append(response_time)
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'p95_response_time': self._calculate_percentile(response_times, 0.95),
            'throughput': len(queries) / total_time
        }
    
    async def run_cache_test(self, queries: list, config: CacheConfig) -> dict:
        """运行缓存测试"""
        print("🚀 Running cache test...")
        
        # 创建缓存服务
        cache_service = IntegratedCacheService(
            processing_func=self.mock_llm_processing,
            config=config
        )
        
        await cache_service.start()
        
        try:
            response_times = []
            cache_hits = 0
            start_time = time.time()
            
            for i, query in enumerate(queries):
                query_start = time.time()
                
                # 记录缓存命中前的统计
                stats_before = cache_service.get_comprehensive_stats()
                hits_before = stats_before['service_stats']['cache_hits']
                
                # 执行查询
                await cache_service.get_response(
                    content=query,
                    model="gpt-3.5-turbo",
                    parameters={"temperature": 0.7}
                )
                
                response_time = time.time() - query_start
                response_times.append(response_time)
                
                # 检查是否命中缓存
                stats_after = cache_service.get_comprehensive_stats()
                hits_after = stats_after['service_stats']['cache_hits']
                
                if hits_after > hits_before:
                    cache_hits += 1
                
                # 每10个查询输出进度
                if (i + 1) % 10 == 0:
                    current_hit_rate = cache_hits / (i + 1)
                    print(f"  Progress: {i+1}/{len(queries)}, Hit rate: {current_hit_rate:.2%}")
            
            total_time = time.time() - start_time
            
            # 获取最终统计
            final_stats = cache_service.get_comprehensive_stats()
            
            return {
                'total_time': total_time,
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p95_response_time': self._calculate_percentile(response_times, 0.95),
                'throughput': len(queries) / total_time,
                'cache_hit_rate': final_stats['service_stats']['hit_rate'],
                'cache_stats': final_stats
            }
            
        finally:
            await cache_service.stop()
    
    def _calculate_percentile(self, data: list, percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def analyze_performance_improvement(self, baseline: dict, cached: dict) -> dict:
        """分析性能提升"""
        improvements = {}
        
        # 响应时间改善
        improvements['avg_response_time_improvement'] = (
            (baseline['avg_response_time'] - cached['avg_response_time']) / 
            baseline['avg_response_time']
        )
        
        improvements['p95_response_time_improvement'] = (
            (baseline['p95_response_time'] - cached['p95_response_time']) / 
            baseline['p95_response_time']
        )
        
        # 吞吐量提升
        improvements['throughput_improvement'] = (
            (cached['throughput'] - baseline['throughput']) / 
            baseline['throughput']
        )
        
        # 总体性能提升
        improvements['overall_performance_improvement'] = (
            (baseline['total_time'] - cached['total_time']) / 
            baseline['total_time']
        )
        
        return improvements
    
    async def run_comprehensive_test(self):
        """运行综合性能测试"""
        print("🚀 Comprehensive Cache Performance Test")
        print("=" * 60)
        
        # 测试场景
        scenarios = [
            ("high_similarity", "高相似度场景 (80%+相似查询)"),
            ("mixed_similarity", "混合相似度场景 (50%相似查询)"),
            ("low_similarity", "低相似度场景 (20%相似查询)")
        ]
        
        test_query_count = 60
        
        for scenario_id, scenario_name in scenarios:
            print(f"\n📊 {scenario_name}")
            print("-" * 40)
            
            # 生成测试查询
            queries = self.generate_test_queries(scenario_id, test_query_count)
            
            # 缓存配置
            config = CacheConfig(
                semantic_cache_size=200,
                semantic_similarity_threshold=0.85,
                hotspot_initial_size=20,
                hotspot_max_size=50,
                enable_prewarming=True,
                prewarm_workers=2
            )
            
            # 运行基线测试
            baseline_results = await self.run_baseline_test(queries)
            
            # 运行缓存测试
            cache_results = await self.run_cache_test(queries, config)
            
            # 分析性能提升
            improvements = self.analyze_performance_improvement(baseline_results, cache_results)
            
            # 输出结果
            self._print_scenario_results(
                scenario_name, 
                baseline_results, 
                cache_results, 
                improvements
            )
            
            # 保存结果
            self.test_results[scenario_id] = {
                'baseline': baseline_results,
                'cached': cache_results,
                'improvements': improvements
            }
        
        # 输出总结
        self._print_summary()
    
    def _print_scenario_results(self, scenario_name: str, baseline: dict, cached: dict, improvements: dict):
        """输出场景测试结果"""
        print(f"\n📈 {scenario_name} 结果:")
        
        print(f"\n  基线测试 (无缓存):")
        print(f"    平均响应时间: {baseline['avg_response_time']:.3f}s")
        print(f"    P95响应时间: {baseline['p95_response_time']:.3f}s")
        print(f"    吞吐量: {baseline['throughput']:.2f} req/s")
        print(f"    总耗时: {baseline['total_time']:.3f}s")
        
        print(f"\n  缓存测试:")
        print(f"    平均响应时间: {cached['avg_response_time']:.3f}s")
        print(f"    P95响应时间: {cached['p95_response_time']:.3f}s")
        print(f"    吞吐量: {cached['throughput']:.2f} req/s")
        print(f"    总耗时: {cached['total_time']:.3f}s")
        print(f"    缓存命中率: {cached['cache_hit_rate']:.2%}")
        
        print(f"\n  🎯 性能提升:")
        print(f"    平均响应时间改善: {improvements['avg_response_time_improvement']:.1%}")
        print(f"    P95响应时间改善: {improvements['p95_response_time_improvement']:.1%}")
        print(f"    吞吐量提升: {improvements['throughput_improvement']:.1%}")
        print(f"    总体性能提升: {improvements['overall_performance_improvement']:.1%}")
        
        # 检查是否达到目标
        if improvements['avg_response_time_improvement'] >= 0.40:
            print(f"    ✅ 达到40%+响应速度提升目标!")
        elif improvements['avg_response_time_improvement'] >= 0.30:
            print(f"    🟡 接近目标，达到30%+响应速度提升")
        else:
            print(f"    ❌ 未达到40%响应速度提升目标")
    
    def _print_summary(self):
        """输出测试总结"""
        print(f"\n🎯 测试总结")
        print("=" * 60)
        
        total_improvements = []
        scenarios_meeting_target = 0
        
        for scenario_id, results in self.test_results.items():
            improvement = results['improvements']['avg_response_time_improvement']
            total_improvements.append(improvement)
            
            if improvement >= 0.40:
                scenarios_meeting_target += 1
        
        avg_improvement = statistics.mean(total_improvements)
        
        print(f"平均响应速度提升: {avg_improvement:.1%}")
        print(f"达到40%+目标的场景: {scenarios_meeting_target}/{len(self.test_results)}")
        
        if avg_improvement >= 0.40:
            print(f"\n🏆 智能缓存系统成功实现40-60%响应速度提升目标!")
        elif avg_improvement >= 0.30:
            print(f"\n🟡 接近目标，平均提升30%+响应速度")
        else:
            print(f"\n❌ 未达到40%响应速度提升目标")
        
        # 最佳场景分析
        best_scenario = max(
            self.test_results.items(), 
            key=lambda x: x[1]['improvements']['avg_response_time_improvement']
        )
        
        print(f"\n📊 最佳场景: {best_scenario[0]}")
        print(f"   响应速度提升: {best_scenario[1]['improvements']['avg_response_time_improvement']:.1%}")
        print(f"   缓存命中率: {best_scenario[1]['cached']['cache_hit_rate']:.2%}")
        
        # 缓存效果分析
        print(f"\n💡 缓存效果分析:")
        print(f"   高相似度场景效果最佳，适合FAQ和常见问题")
        print(f"   混合场景平衡性能和覆盖面")
        print(f"   低相似度场景主要依靠热点检测和预热")


async def main():
    """主函数"""
    tester = CachePerformanceTester()
    await tester.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())
