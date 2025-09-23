#!/usr/bin/env python3
"""
性能优化验证器
验证优化措施的效果
"""

import time
import json
import requests
import psutil
import os
import sys
from typing import Dict, Any, List

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from algo.core.memory_optimizer import memory_optimizer, get_memory_report
except ImportError:
    print("警告: 无法导入内存优化器，跳过内存优化测试")
    memory_optimizer = None

class OptimizationValidator:
    """性能优化验证器"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8080"
        self.algo_url = "http://localhost:8000"
        self.results = {}
        
    def run_validation(self):
        """运行优化验证"""
        print("🔍 VoiceHelper 性能优化验证")
        print("=" * 50)
        
        # 1. 基准测试 (优化前)
        print("\n📊 执行基准测试...")
        baseline = self.run_performance_test("baseline")
        
        # 2. 应用内存优化
        if memory_optimizer:
            print("\n🧠 应用内存优化...")
            self.apply_memory_optimizations()
        
        # 3. 优化后测试
        print("\n📈 执行优化后测试...")
        optimized = self.run_performance_test("optimized")
        
        # 4. 对比分析
        print("\n📋 生成对比分析...")
        comparison = self.compare_results(baseline, optimized)
        
        # 5. 生成报告
        self.generate_validation_report(baseline, optimized, comparison)
        
        print("\n✅ 性能优化验证完成！")
        
    def run_performance_test(self, test_name: str) -> Dict[str, Any]:
        """运行性能测试"""
        print(f"  执行 {test_name} 测试...")
        
        results = {
            'test_name': test_name,
            'timestamp': time.time(),
            'system_metrics': self.get_system_metrics(),
            'api_performance': self.test_api_performance(),
            'memory_usage': self.test_memory_usage(),
            'concurrent_performance': self.test_concurrent_performance()
        }
        
        return results
    
    def get_system_metrics(self) -> Dict[str, float]:
        """获取系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用
        memory = psutil.virtual_memory()
        
        # 进程内存使用
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'process_memory_mb': process_memory.rss / (1024**2),
            'process_memory_percent': process.memory_percent()
        }
    
    def test_api_performance(self) -> Dict[str, Any]:
        """测试API性能"""
        endpoints = [
            (f"{self.backend_url}/health", "backend_health"),
            (f"{self.algo_url}/health", "algo_health")
        ]
        
        results = {}
        
        for url, name in endpoints:
            try:
                # 预热
                requests.get(url, timeout=5)
                
                # 测试多次取平均值
                response_times = []
                for _ in range(10):
                    start_time = time.time()
                    response = requests.get(url, timeout=5)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_times.append((end_time - start_time) * 1000)
                
                if response_times:
                    results[name] = {
                        'avg_response_time_ms': sum(response_times) / len(response_times),
                        'min_response_time_ms': min(response_times),
                        'max_response_time_ms': max(response_times),
                        'success_rate': len(response_times) / 10
                    }
                else:
                    results[name] = {'error': 'All requests failed'}
                    
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用"""
        # 获取初始内存
        initial_memory = psutil.Process().memory_info().rss
        
        # 模拟内存密集操作
        test_data = []
        for i in range(5000):
            test_data.append({
                'id': i,
                'content': f'Test content {i}' * 10,
                'metadata': {'timestamp': time.time(), 'index': i}
            })
        
        # 获取峰值内存
        peak_memory = psutil.Process().memory_info().rss
        
        # 清理数据
        del test_data
        
        # 获取清理后内存
        final_memory = psutil.Process().memory_info().rss
        
        return {
            'initial_memory_mb': initial_memory / (1024**2),
            'peak_memory_mb': peak_memory / (1024**2),
            'final_memory_mb': final_memory / (1024**2),
            'memory_increase_mb': (peak_memory - initial_memory) / (1024**2),
            'memory_retained_mb': (final_memory - initial_memory) / (1024**2)
        }
    
    def test_concurrent_performance(self) -> Dict[str, Any]:
        """测试并发性能"""
        import concurrent.futures
        
        def make_request():
            try:
                start_time = time.time()
                response = requests.get(f"{self.backend_url}/health", timeout=5)
                end_time = time.time()
                return {
                    'success': response.status_code == 200,
                    'response_time_ms': (end_time - start_time) * 1000
                }
            except:
                return {'success': False, 'response_time_ms': None}
        
        # 测试不同并发级别
        concurrent_levels = [5, 10, 20]
        results = {}
        
        for level in concurrent_levels:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(make_request) for _ in range(level)]
                responses = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
            
            successful = [r for r in responses if r['success']]
            response_times = [r['response_time_ms'] for r in successful if r['response_time_ms']]
            
            results[f'concurrent_{level}'] = {
                'total_time_ms': (end_time - start_time) * 1000,
                'success_count': len(successful),
                'success_rate': len(successful) / level,
                'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
                'throughput_rps': level / ((end_time - start_time) if (end_time - start_time) > 0 else 1)
            }
        
        return results
    
    def apply_memory_optimizations(self):
        """应用内存优化"""
        if not memory_optimizer:
            print("  内存优化器不可用，跳过优化")
            return
        
        try:
            # 执行内存优化
            optimization_result = memory_optimizer.optimize_memory()
            
            print("  内存优化结果:")
            print(f"    清理死亡引用: {optimization_result.get('dead_refs_cleaned', 0)}")
            print(f"    垃圾回收释放对象: {optimization_result.get('gc_stats', {}).get('objects_freed', 0)}")
            
            # 创建优化缓存
            cache = memory_optimizer.create_cache('api_cache', 500)
            print(f"    创建API缓存: 最大{500}项")
            
            # 创建对象池
            def create_request_obj():
                return {'headers': {}, 'data': None, 'response': None}
            
            def reset_request_obj(obj):
                obj['headers'].clear()
                obj['data'] = None
                obj['response'] = None
            
            pool = memory_optimizer.create_object_pool('request_pool', create_request_obj, reset_request_obj, 50)
            print(f"    创建请求对象池: 最大{50}个对象")
            
        except Exception as e:
            print(f"  内存优化失败: {e}")
    
    def compare_results(self, baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
        """对比测试结果"""
        comparison = {
            'timestamp': time.time(),
            'improvements': {},
            'regressions': {},
            'summary': {}
        }
        
        # 对比系统指标
        baseline_sys = baseline['system_metrics']
        optimized_sys = optimized['system_metrics']
        
        sys_comparison = {}
        for metric in ['cpu_percent', 'memory_percent', 'process_memory_mb']:
            baseline_val = baseline_sys.get(metric, 0)
            optimized_val = optimized_sys.get(metric, 0)
            
            if baseline_val > 0:
                improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                sys_comparison[metric] = {
                    'baseline': baseline_val,
                    'optimized': optimized_val,
                    'improvement_percent': improvement
                }
        
        comparison['system_metrics'] = sys_comparison
        
        # 对比API性能
        api_comparison = {}
        baseline_api = baseline['api_performance']
        optimized_api = optimized['api_performance']
        
        for endpoint in baseline_api:
            if endpoint in optimized_api and 'avg_response_time_ms' in baseline_api[endpoint] and 'avg_response_time_ms' in optimized_api[endpoint]:
                baseline_time = baseline_api[endpoint]['avg_response_time_ms']
                optimized_time = optimized_api[endpoint]['avg_response_time_ms']
                
                improvement = ((baseline_time - optimized_time) / baseline_time) * 100
                api_comparison[endpoint] = {
                    'baseline_ms': baseline_time,
                    'optimized_ms': optimized_time,
                    'improvement_percent': improvement
                }
        
        comparison['api_performance'] = api_comparison
        
        # 对比内存使用
        baseline_mem = baseline['memory_usage']
        optimized_mem = optimized['memory_usage']
        
        mem_comparison = {}
        for metric in ['memory_increase_mb', 'memory_retained_mb']:
            baseline_val = baseline_mem.get(metric, 0)
            optimized_val = optimized_mem.get(metric, 0)
            
            if baseline_val > 0:
                improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                mem_comparison[metric] = {
                    'baseline': baseline_val,
                    'optimized': optimized_val,
                    'improvement_percent': improvement
                }
        
        comparison['memory_usage'] = mem_comparison
        
        # 生成总结
        improvements = []
        regressions = []
        
        # 检查各项指标的改善情况
        for category, metrics in [
            ('system_metrics', sys_comparison),
            ('api_performance', api_comparison),
            ('memory_usage', mem_comparison)
        ]:
            for metric, data in metrics.items():
                improvement = data.get('improvement_percent', 0)
                if improvement > 5:  # 改善超过5%
                    improvements.append(f"{category}.{metric}: {improvement:.1f}%")
                elif improvement < -5:  # 恶化超过5%
                    regressions.append(f"{category}.{metric}: {improvement:.1f}%")
        
        comparison['improvements'] = improvements
        comparison['regressions'] = regressions
        
        # 计算总体改善评分
        all_improvements = []
        for category in [sys_comparison, api_comparison, mem_comparison]:
            for data in category.values():
                all_improvements.append(data.get('improvement_percent', 0))
        
        if all_improvements:
            avg_improvement = sum(all_improvements) / len(all_improvements)
            comparison['summary'] = {
                'overall_improvement_percent': avg_improvement,
                'total_metrics_improved': len(improvements),
                'total_metrics_regressed': len(regressions),
                'optimization_success': avg_improvement > 0 and len(improvements) > len(regressions)
            }
        
        return comparison
    
    def generate_validation_report(self, baseline: Dict[str, Any], optimized: Dict[str, Any], comparison: Dict[str, Any]):
        """生成验证报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'baseline_results': baseline,
            'optimized_results': optimized,
            'comparison': comparison,
            'recommendations': self.get_optimization_recommendations(comparison)
        }
        
        # 保存详细报告
        report_file = 'tests/performance/optimization_validation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成简化报告
        self.print_summary_report(comparison)
        
        print(f"\n📄 详细报告已保存: {report_file}")
    
    def print_summary_report(self, comparison: Dict[str, Any]):
        """打印摘要报告"""
        print("\n" + "="*60)
        print("🎯 性能优化验证摘要报告")
        print("="*60)
        
        summary = comparison.get('summary', {})
        
        # 总体结果
        overall_improvement = summary.get('overall_improvement_percent', 0)
        success = summary.get('optimization_success', False)
        
        status_icon = "✅" if success else "⚠️"
        print(f"\n{status_icon} 总体优化效果: {overall_improvement:+.1f}%")
        
        # 改善项目
        improvements = comparison.get('improvements', [])
        if improvements:
            print(f"\n📈 性能改善项目 ({len(improvements)}项):")
            for improvement in improvements[:5]:  # 显示前5项
                print(f"  • {improvement}")
        
        # 回归项目
        regressions = comparison.get('regressions', [])
        if regressions:
            print(f"\n📉 性能回归项目 ({len(regressions)}项):")
            for regression in regressions[:3]:  # 显示前3项
                print(f"  • {regression}")
        
        # 关键指标对比
        print(f"\n📊 关键指标对比:")
        
        # 内存使用对比
        mem_metrics = comparison.get('memory_usage', {})
        for metric, data in mem_metrics.items():
            baseline = data['baseline']
            optimized = data['optimized']
            improvement = data['improvement_percent']
            print(f"  内存{metric}: {baseline:.2f} → {optimized:.2f} ({improvement:+.1f}%)")
        
        # API性能对比
        api_metrics = comparison.get('api_performance', {})
        for endpoint, data in api_metrics.items():
            baseline = data['baseline_ms']
            optimized = data['optimized_ms']
            improvement = data['improvement_percent']
            print(f"  {endpoint}: {baseline:.2f}ms → {optimized:.2f}ms ({improvement:+.1f}%)")
    
    def get_optimization_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        summary = comparison.get('summary', {})
        overall_improvement = summary.get('overall_improvement_percent', 0)
        
        if overall_improvement > 10:
            recommendations.append("优化效果显著，建议将当前优化措施应用到生产环境")
        elif overall_improvement > 5:
            recommendations.append("优化效果良好，建议进一步测试后应用到生产环境")
        elif overall_improvement > 0:
            recommendations.append("优化效果轻微，建议继续寻找其他优化点")
        else:
            recommendations.append("优化效果不明显，建议重新评估优化策略")
        
        # 基于具体指标给出建议
        regressions = comparison.get('regressions', [])
        if regressions:
            recommendations.append("注意部分指标出现回归，需要进一步调优")
        
        improvements = comparison.get('improvements', [])
        if len(improvements) > 3:
            recommendations.append("多项指标得到改善，优化方向正确")
        
        return recommendations

if __name__ == "__main__":
    validator = OptimizationValidator()
    validator.run_validation()
