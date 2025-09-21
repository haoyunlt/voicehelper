"""
简化的批量化系统性能测试

不依赖外部库的基础性能测试
"""

import asyncio
import time
import statistics
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class MockLLMClient:
    """模拟LLM客户端"""
    
    async def process_single(self, request):
        """处理单个请求"""
        # 模拟真实LLM调用：包含网络延迟和模型推理时间
        network_latency = 0.05  # 50ms网络延迟
        model_inference = 0.8   # 800ms模型推理时间
        
        await asyncio.sleep(network_latency)  # 网络延迟
        await asyncio.sleep(model_inference)  # 模型推理
        
        content = request.get('content', 'Hello')
        return {
            'content': f"Response to: {content}",
            'processing_time': network_latency + model_inference,
            'model': request.get('model', 'gpt-3.5-turbo')
        }
    
    async def process_batch(self, requests):
        """处理批量请求"""
        batch_size = len(requests)
        
        # 模拟真实批处理：网络延迟只有一次，模型推理有批处理效率
        network_latency = 0.05  # 50ms网络延迟（只有一次）
        base_inference_time = 0.8  # 800ms基础推理时间
        
        # 批处理效率：批次越大，单个请求的平均推理时间越短
        batch_efficiency = min(2.0, 1.0 + (batch_size - 1) * 0.15)  # 每增加一个请求，效率提升15%
        actual_inference_time = base_inference_time / batch_efficiency
        
        total_processing_time = network_latency + actual_inference_time
        
        await asyncio.sleep(total_processing_time)
        
        results = []
        for req in requests:
            content = req.get('content', 'Hello')
            result = {
                'content': f"Batch response to: {content}",
                'processing_time': total_processing_time,
                'model': req.get('model', 'gpt-3.5-turbo')
            }
            results.append(result)
        
        return results


class SimpleBatchProcessor:
    """简化的批处理器"""
    
    def __init__(self, batch_size=4, max_wait_time=0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.client = MockLLMClient()
        self.queue = asyncio.Queue()
        self.running = False
        self.processing_task = None
        
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
                            timeout=min(remaining_wait, 0.01)
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
                    except Exception as e:
                        for future in futures:
                            if not future.done():
                                future.set_exception(e)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Processing error: {e}")


async def run_performance_test():
    """运行性能测试"""
    print("🚀 批量化系统性能测试")
    print("=" * 50)
    
    # 测试参数
    total_requests = 40  # 减少请求数以便观察批处理效果
    concurrent_users = 8  # 增加并发用户
    
    # 生成测试请求
    test_requests = []
    for i in range(total_requests):
        request = {
            'content': f"Test request {i}",
            'model': 'gpt-3.5-turbo'
        }
        test_requests.append(request)
    
    print(f"📊 测试配置:")
    print(f"  总请求数: {total_requests}")
    print(f"  并发用户: {concurrent_users}")
    print(f"  批次大小: 4")
    print(f"  最大等待: 0.1s")
    print(f"  模拟场景: 真实LLM调用 (网络50ms + 推理800ms)")
    
    # 测试1: 批处理模式
    print(f"\n🔄 测试1: 批处理模式")
    batch_processor = SimpleBatchProcessor(batch_size=4, max_wait_time=0.1)
    await batch_processor.start()
    
    async def send_batch_request(req, delay):
        await asyncio.sleep(delay)
        return await batch_processor.process_request(req)
    
    start_time = time.time()
    batch_tasks = []
    for i, req in enumerate(test_requests):
        delay = (i // concurrent_users) * 0.01  # 模拟用户请求间隔
        task = asyncio.create_task(send_batch_request(req, delay))
        batch_tasks.append(task)
    
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    batch_total_time = time.time() - start_time
    
    await batch_processor.stop()
    
    # 测试2: 直接处理模式
    print(f"\n🔄 测试2: 直接处理模式")
    client = MockLLMClient()
    
    async def send_direct_request(req, delay):
        await asyncio.sleep(delay)
        return await client.process_single(req)
    
    start_time = time.time()
    direct_tasks = []
    for i, req in enumerate(test_requests):
        delay = (i // concurrent_users) * 0.01
        task = asyncio.create_task(send_direct_request(req, delay))
        direct_tasks.append(task)
    
    direct_results = await asyncio.gather(*direct_tasks, return_exceptions=True)
    direct_total_time = time.time() - start_time
    
    # 统计结果
    def analyze_results(results, total_time, mode_name):
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = len(results) - len(successful)
        
        if successful:
            response_times = [r.get('processing_time', 0) for r in successful]
            avg_response_time = statistics.mean(response_times)
            # 计算P95 (兼容Python 3.7)
            if len(response_times) > 1:
                sorted_times = sorted(response_times)
                p95_index = int(0.95 * len(sorted_times))
                p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
            else:
                p95_response_time = avg_response_time
        else:
            avg_response_time = 0
            p95_response_time = 0
        
        throughput = len(successful) / total_time if total_time > 0 else 0
        
        print(f"\n📈 {mode_name} 结果:")
        print(f"  成功请求: {len(successful)}/{len(results)}")
        print(f"  失败请求: {failed}")
        print(f"  总耗时: {total_time:.3f}s")
        print(f"  吞吐量: {throughput:.2f} req/s")
        print(f"  平均延迟: {avg_response_time:.3f}s")
        print(f"  P95延迟: {p95_response_time:.3f}s")
        
        return {
            'successful': len(successful),
            'failed': failed,
            'total_time': total_time,
            'throughput': throughput,
            'avg_latency': avg_response_time,
            'p95_latency': p95_response_time
        }
    
    batch_stats = analyze_results(batch_results, batch_total_time, "批处理模式")
    direct_stats = analyze_results(direct_results, direct_total_time, "直接处理模式")
    
    # 性能对比
    print(f"\n🎯 性能对比分析:")
    print("=" * 50)
    
    if direct_stats['throughput'] > 0:
        throughput_improvement = (batch_stats['throughput'] - direct_stats['throughput']) / direct_stats['throughput']
        print(f"吞吐量提升: {throughput_improvement:.1%}")
        
        if throughput_improvement >= 0.30:
            print("✅ 达到30%+吞吐量提升目标!")
        else:
            print("❌ 未达到30%吞吐量提升目标")
    
    if direct_stats['avg_latency'] > 0:
        latency_change = (batch_stats['avg_latency'] - direct_stats['avg_latency']) / direct_stats['avg_latency']
        print(f"延迟变化: {latency_change:.1%}")
    
    efficiency_gain = batch_stats['throughput'] / direct_stats['throughput'] if direct_stats['throughput'] > 0 else 1
    print(f"整体效率提升: {efficiency_gain:.2f}x")
    
    # 资源效率分析
    print(f"\n📊 资源效率分析:")
    batch_req_per_sec = batch_stats['successful'] / batch_stats['total_time']
    direct_req_per_sec = direct_stats['successful'] / direct_stats['total_time']
    
    print(f"批处理模式: {batch_req_per_sec:.2f} req/s")
    print(f"直接处理模式: {direct_req_per_sec:.2f} req/s")
    print(f"效率比值: {batch_req_per_sec / direct_req_per_sec:.2f}")
    
    print(f"\n🎉 测试完成!")
    return {
        'batch': batch_stats,
        'direct': direct_stats,
        'improvement': throughput_improvement if direct_stats['throughput'] > 0 else 0
    }


if __name__ == "__main__":
    asyncio.run(run_performance_test())
