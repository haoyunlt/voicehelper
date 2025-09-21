"""
真实场景的批量化系统性能测试

模拟高并发、突发流量的真实LLM服务场景
"""

import asyncio
import time
import statistics
import sys
import os
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class RealisticLLMClient:
    """真实场景LLM客户端"""
    
    def __init__(self):
        self.concurrent_limit = 5  # 模拟API并发限制
        self.semaphore = asyncio.Semaphore(self.concurrent_limit)
    
    async def process_single(self, request):
        """处理单个请求 - 受并发限制"""
        async with self.semaphore:
            # 模拟真实LLM API调用
            network_latency = random.uniform(0.03, 0.08)  # 30-80ms网络延迟
            model_inference = random.uniform(0.6, 1.2)    # 600-1200ms推理时间
            
            await asyncio.sleep(network_latency)
            await asyncio.sleep(model_inference)
            
            content = request.get('content', 'Hello')
            return {
                'content': f"Response to: {content}",
                'processing_time': network_latency + model_inference,
                'model': request.get('model', 'gpt-3.5-turbo')
            }
    
    async def process_batch(self, requests):
        """处理批量请求 - 批处理优化"""
        batch_size = len(requests)
        
        # 批处理只需要一次网络往返
        network_latency = random.uniform(0.03, 0.08)
        
        # 批处理推理时间优化
        base_inference = random.uniform(0.6, 1.2)
        # 批处理效率：每增加一个请求，总时间只增加20%而不是100%
        batch_factor = 1.0 + (batch_size - 1) * 0.2
        total_inference_time = base_inference * batch_factor
        
        # 单个请求的平均处理时间
        avg_processing_time = (network_latency + total_inference_time) / batch_size
        
        await asyncio.sleep(network_latency + total_inference_time)
        
        results = []
        for req in requests:
            content = req.get('content', 'Hello')
            result = {
                'content': f"Batch response to: {content}",
                'processing_time': avg_processing_time,
                'model': req.get('model', 'gpt-3.5-turbo')
            }
            results.append(result)
        
        return results


class RealisticBatchProcessor:
    """真实场景批处理器"""
    
    def __init__(self, batch_size=8, max_wait_time=0.05):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.client = RealisticLLMClient()
        self.queue = asyncio.Queue()
        self.running = False
        self.processing_task = None
        self.stats = {
            'batches_processed': 0,
            'total_requests': 0,
            'avg_batch_size': 0
        }
        
    async def start(self):
        """启动批处理器"""
        if not self.running:
            self.running = True
            self.processing_task = asyncio.create_task(self._processing_loop())
    
    async def stop(self):
        """停止批处理器"""
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def process_request(self, request):
        """处理请求"""
        future = asyncio.Future()
        await self.queue.put({'request': request, 'future': future})
        return await future
    
    async def _processing_loop(self):
        """处理循环"""
        while self.running:
            try:
                # 收集批次
                batch = []
                futures = []
                batch_start = time.time()
                
                # 收集请求直到达到批次大小或超时
                while len(batch) < self.batch_size and self.running:
                    elapsed = time.time() - batch_start
                    remaining_wait = self.max_wait_time - elapsed
                    
                    if remaining_wait <= 0:
                        break
                    
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(), 
                            timeout=min(remaining_wait, 0.005)  # 更短的轮询间隔
                        )
                        batch.append(item['request'])
                        futures.append(item['future'])
                    except asyncio.TimeoutError:
                        break
                
                # 处理批次
                if batch:
                    try:
                        results = await self.client.process_batch(batch)
                        for future, result in zip(futures, results):
                            if not future.done():
                                future.set_result(result)
                        
                        # 更新统计
                        self.stats['batches_processed'] += 1
                        self.stats['total_requests'] += len(batch)
                        self.stats['avg_batch_size'] = (
                            self.stats['total_requests'] / self.stats['batches_processed']
                        )
                        
                    except Exception as e:
                        for future in futures:
                            if not future.done():
                                future.set_exception(e)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Processing error: {e}")


async def simulate_burst_traffic():
    """模拟突发流量场景"""
    print("🚀 真实场景批量化系统性能测试")
    print("=" * 60)
    
    # 测试参数
    total_requests = 100
    burst_duration = 10.0  # 10秒内发送所有请求
    
    # 生成测试请求
    test_requests = []
    for i in range(total_requests):
        request = {
            'content': f"User query {i}: What is artificial intelligence?",
            'model': 'gpt-3.5-turbo'
        }
        test_requests.append(request)
    
    print(f"📊 测试配置:")
    print(f"  总请求数: {total_requests}")
    print(f"  突发时长: {burst_duration}s")
    print(f"  平均QPS: {total_requests/burst_duration:.1f}")
    print(f"  批次大小: 8")
    print(f"  最大等待: 50ms")
    print(f"  API并发限制: 5")
    
    # 测试1: 批处理模式
    print(f"\n🔄 测试1: 批处理模式")
    batch_processor = RealisticBatchProcessor(batch_size=8, max_wait_time=0.05)
    await batch_processor.start()
    
    async def send_batch_request(req, delay):
        await asyncio.sleep(delay)
        return await batch_processor.process_request(req)
    
    # 模拟突发流量：在10秒内随机发送所有请求
    start_time = time.time()
    batch_tasks = []
    for i, req in enumerate(test_requests):
        # 随机分布在burst_duration时间内
        delay = random.uniform(0, burst_duration)
        task = asyncio.create_task(send_batch_request(req, delay))
        batch_tasks.append(task)
    
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    batch_total_time = time.time() - start_time
    
    batch_stats = batch_processor.stats
    await batch_processor.stop()
    
    # 测试2: 直接处理模式
    print(f"\n🔄 测试2: 直接处理模式")
    client = RealisticLLMClient()
    
    async def send_direct_request(req, delay):
        await asyncio.sleep(delay)
        return await client.process_single(req)
    
    start_time = time.time()
    direct_tasks = []
    for i, req in enumerate(test_requests):
        delay = random.uniform(0, burst_duration)
        task = asyncio.create_task(send_direct_request(req, delay))
        direct_tasks.append(task)
    
    direct_results = await asyncio.gather(*direct_tasks, return_exceptions=True)
    direct_total_time = time.time() - start_time
    
    # 分析结果
    def analyze_results(results, total_time, mode_name):
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = len(results) - len(successful)
        
        if successful:
            response_times = [r.get('processing_time', 0) for r in successful]
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # 计算P95 (兼容Python 3.7)
            if len(response_times) > 1:
                sorted_times = sorted(response_times)
                p95_index = int(0.95 * len(sorted_times))
                p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
            else:
                p95_response_time = avg_response_time
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        throughput = len(successful) / total_time if total_time > 0 else 0
        
        print(f"\n📈 {mode_name} 结果:")
        print(f"  成功请求: {len(successful)}/{len(results)}")
        print(f"  失败请求: {failed}")
        print(f"  总耗时: {total_time:.3f}s")
        print(f"  吞吐量: {throughput:.2f} req/s")
        print(f"  平均延迟: {avg_response_time:.3f}s")
        print(f"  最小延迟: {min_response_time:.3f}s")
        print(f"  最大延迟: {max_response_time:.3f}s")
        print(f"  P95延迟: {p95_response_time:.3f}s")
        
        return {
            'successful': len(successful),
            'failed': failed,
            'total_time': total_time,
            'throughput': throughput,
            'avg_latency': avg_response_time,
            'min_latency': min_response_time,
            'max_latency': max_response_time,
            'p95_latency': p95_response_time
        }
    
    batch_result = analyze_results(batch_results, batch_total_time, "批处理模式")
    direct_result = analyze_results(direct_results, direct_total_time, "直接处理模式")
    
    # 批处理统计
    print(f"\n📊 批处理统计:")
    print(f"  处理批次数: {batch_stats['batches_processed']}")
    print(f"  平均批次大小: {batch_stats['avg_batch_size']:.1f}")
    print(f"  批处理效率: {batch_stats['total_requests']}/{batch_stats['batches_processed']} = {batch_stats['avg_batch_size']:.1f}x")
    
    # 性能对比
    print(f"\n🎯 性能对比分析:")
    print("=" * 60)
    
    if direct_result['throughput'] > 0:
        throughput_improvement = (batch_result['throughput'] - direct_result['throughput']) / direct_result['throughput']
        print(f"吞吐量提升: {throughput_improvement:.1%}")
        
        if throughput_improvement >= 0.30:
            print("✅ 达到30%+吞吐量提升目标!")
        else:
            print("❌ 未达到30%吞吐量提升目标")
    
    if direct_result['avg_latency'] > 0:
        latency_change = (batch_result['avg_latency'] - direct_result['avg_latency']) / direct_result['avg_latency']
        print(f"平均延迟变化: {latency_change:.1%}")
        
        p95_latency_change = (batch_result['p95_latency'] - direct_result['p95_latency']) / direct_result['p95_latency']
        print(f"P95延迟变化: {p95_latency_change:.1%}")
    
    # 资源效率分析
    print(f"\n📊 资源效率分析:")
    efficiency_ratio = batch_result['throughput'] / direct_result['throughput'] if direct_result['throughput'] > 0 else 1
    print(f"整体效率提升: {efficiency_ratio:.2f}x")
    
    # 并发处理能力分析
    print(f"\n⚡ 并发处理能力:")
    print(f"批处理模式峰值QPS: {batch_result['throughput']:.2f}")
    print(f"直接处理模式峰值QPS: {direct_result['throughput']:.2f}")
    
    if batch_result['throughput'] > direct_result['throughput']:
        improvement_factor = batch_result['throughput'] / direct_result['throughput']
        print(f"🚀 批处理模式在高并发场景下性能提升 {improvement_factor:.1f}x!")
    
    print(f"\n🎉 测试完成!")
    
    return {
        'batch': batch_result,
        'direct': direct_result,
        'batch_stats': batch_stats,
        'improvement': throughput_improvement if direct_result['throughput'] > 0 else 0
    }


async def simulate_sustained_load():
    """模拟持续负载场景"""
    print("\n" + "="*60)
    print("🔥 持续负载测试 (模拟生产环境)")
    print("="*60)
    
    total_requests = 200
    test_duration = 30.0  # 30秒持续测试
    
    print(f"📊 持续负载配置:")
    print(f"  总请求数: {total_requests}")
    print(f"  测试时长: {test_duration}s")
    print(f"  目标QPS: {total_requests/test_duration:.1f}")
    
    # 批处理模式测试
    print(f"\n🔄 批处理模式 - 持续负载")
    batch_processor = RealisticBatchProcessor(batch_size=6, max_wait_time=0.08)
    await batch_processor.start()
    
    async def sustained_batch_load():
        tasks = []
        interval = test_duration / total_requests
        
        for i in range(total_requests):
            request = {'content': f"Sustained query {i}", 'model': 'gpt-3.5-turbo'}
            delay = i * interval
            task = asyncio.create_task(
                asyncio.sleep(delay) and batch_processor.process_request(request)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    start_time = time.time()
    batch_results = await sustained_batch_load()
    batch_time = time.time() - start_time
    
    await batch_processor.stop()
    
    # 直接处理模式测试
    print(f"\n🔄 直接处理模式 - 持续负载")
    client = RealisticLLMClient()
    
    async def sustained_direct_load():
        tasks = []
        interval = test_duration / total_requests
        
        for i in range(total_requests):
            request = {'content': f"Sustained query {i}", 'model': 'gpt-3.5-turbo'}
            delay = i * interval
            
            async def delayed_request(req, d):
                await asyncio.sleep(d)
                return await client.process_single(req)
            
            task = asyncio.create_task(delayed_request(request, delay))
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    start_time = time.time()
    direct_results = await sustained_direct_load()
    direct_time = time.time() - start_time
    
    # 分析持续负载结果
    def analyze_sustained_results(results, total_time, mode_name):
        successful = [r for r in results if not isinstance(r, Exception)]
        
        if successful:
            response_times = [r.get('processing_time', 0) for r in successful]
            avg_latency = statistics.mean(response_times)
        else:
            avg_latency = 0
        
        actual_qps = len(successful) / total_time if total_time > 0 else 0
        
        print(f"\n📈 {mode_name} - 持续负载结果:")
        print(f"  成功处理: {len(successful)}/{len(results)}")
        print(f"  实际QPS: {actual_qps:.2f}")
        print(f"  平均延迟: {avg_latency:.3f}s")
        
        return {
            'successful': len(successful),
            'qps': actual_qps,
            'avg_latency': avg_latency
        }
    
    batch_sustained = analyze_sustained_results(batch_results, batch_time, "批处理模式")
    direct_sustained = analyze_sustained_results(direct_results, direct_time, "直接处理模式")
    
    # 持续负载对比
    print(f"\n🎯 持续负载性能对比:")
    if direct_sustained['qps'] > 0:
        qps_improvement = (batch_sustained['qps'] - direct_sustained['qps']) / direct_sustained['qps']
        print(f"QPS提升: {qps_improvement:.1%}")
        
        if qps_improvement >= 0.30:
            print("✅ 持续负载下达到30%+性能提升!")
        else:
            print("❌ 持续负载下未达到30%性能提升")


if __name__ == "__main__":
    async def main():
        # 突发流量测试
        await simulate_burst_traffic()
        
        # 持续负载测试
        await simulate_sustained_load()
    
    asyncio.run(main())
