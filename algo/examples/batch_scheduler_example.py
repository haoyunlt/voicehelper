#!/usr/bin/env python3
"""
批处理调度器使用示例
展示企业级批处理调度器的各种功能
"""

import asyncio
import time
import uuid
from algo.core.simple_batch_scheduler import (
    BatchScheduler, SimpleBatchScheduler, SchedulerConfig,
    ProcessRequest, RequestType, RequestPriority
)

async def basic_example():
    """基础使用示例"""
    print("=== 基础批处理调度器示例 ===")
    
    # 使用简化版调度器
    scheduler = SimpleBatchScheduler()
    await scheduler.start()
    
    try:
        # 创建测试请求
        requests = []
        for i in range(5):
            request = ProcessRequest(
                id=str(uuid.uuid4()),
                type=RequestType.TEXT_GENERATION,
                priority=RequestPriority.NORMAL,
                data=f"测试请求 {i}",
                user_id=f"user_{i}"
            )
            requests.append(request)
            await scheduler.submit_request(request)
        
        # 等待处理完成
        await asyncio.sleep(1)
        
        # 显示统计信息
        stats = scheduler.get_statistics()
        print(f"处理请求数: {stats['total_requests']}")
        print(f"批次数: {stats['total_batches']}")
        print(f"平均批次大小: {stats['average_batch_size']:.2f}")
        
    finally:
        await scheduler.stop()

async def advanced_example():
    """高级功能示例"""
    print("\n=== 企业级批处理调度器示例 ===")
    
    # 配置企业级调度器
    config = SchedulerConfig(
        max_batch_size=16,
        max_wait_time=0.2,
        max_concurrent_batches=3,
        worker_pool_size=6,
        enable_metrics=True,
        enable_circuit_breaker=True,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=10.0
    )
    
    scheduler = BatchScheduler(config)
    await scheduler.start()
    
    try:
        # 创建不同优先级和类型的请求
        futures = []
        
        # 高优先级请求
        for i in range(3):
            request = ProcessRequest(
                id=f"urgent_{i}",
                type=RequestType.TEXT_GENERATION,
                priority=RequestPriority.URGENT,
                data=f"紧急请求 {i}",
                user_id=f"vip_user_{i}",
                timeout=5.0
            )
            future = await scheduler.submit_request(request)
            futures.append(future)
        
        # 普通优先级请求
        for i in range(10):
            request = ProcessRequest(
                id=f"normal_{i}",
                type=RequestType.EMBEDDING,
                priority=RequestPriority.NORMAL,
                data=f"普通请求 {i}",
                user_id=f"user_{i}"
            )
            future = await scheduler.submit_request(request)
            futures.append(future)
        
        # 低优先级请求
        for i in range(5):
            request = ProcessRequest(
                id=f"low_{i}",
                type=RequestType.VOICE_SYNTHESIS,
                priority=RequestPriority.LOW,
                data=f"低优先级请求 {i}",
                user_id=f"user_{i}"
            )
            future = await scheduler.submit_request(request)
            futures.append(future)
        
        print(f"提交了 {len(futures)} 个请求")
        
        # 等待部分结果
        completed_futures = []
        for future in futures[:5]:  # 只等待前5个
            try:
                result = await asyncio.wait_for(future, timeout=3.0)
                completed_futures.append(result)
                print(f"完成请求: {result}")
            except asyncio.TimeoutError:
                print("请求超时")
            except Exception as e:
                print(f"请求失败: {e}")
        
        # 等待所有处理完成
        await asyncio.sleep(2)
        
        # 显示详细统计
        stats = scheduler.get_statistics()
        print(f"\n=== 统计信息 ===")
        print(f"总请求数: {stats['total_requests']}")
        print(f"总批次数: {stats['total_batches']}")
        print(f"成功批次: {stats['successful_batches']}")
        print(f"失败批次: {stats['failed_batches']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"平均批次大小: {stats['average_batch_size']:.2f}")
        print(f"平均处理时间: {stats['average_processing_time']:.3f}s")
        print(f"队列大小: {stats['queue_size']}")
        print(f"待处理请求: {stats['pending_requests']}")
        
        if 'circuit_breaker_state' in stats:
            print(f"熔断器状态: {stats['circuit_breaker_state']}")
            print(f"失败次数: {stats['circuit_breaker_failures']}")
        
        # 显示性能指标
        metrics = scheduler.get_metrics()
        if metrics:
            print(f"\n=== 性能指标 ===")
            if 'batch_sizes' in metrics:
                avg_batch_size = sum(metrics['batch_sizes']) / len(metrics['batch_sizes'])
                print(f"平均批次大小: {avg_batch_size:.2f}")
            
            if 'processing_times' in metrics:
                avg_time = sum(metrics['processing_times']) / len(metrics['processing_times'])
                print(f"平均处理时间: {avg_time:.3f}s")
            
            if 'throughputs' in metrics:
                avg_throughput = sum(metrics['throughputs']) / len(metrics['throughputs'])
                print(f"平均吞吐量: {avg_throughput:.2f} req/s")
        
    finally:
        await scheduler.stop()

async def stress_test():
    """压力测试示例"""
    print("\n=== 压力测试 ===")
    
    config = SchedulerConfig(
        max_batch_size=32,
        max_wait_time=0.05,
        max_concurrent_batches=8,
        worker_pool_size=12,
        enable_metrics=True,
        enable_circuit_breaker=True
    )
    
    scheduler = BatchScheduler(config)
    await scheduler.start()
    
    try:
        start_time = time.time()
        
        # 快速提交大量请求
        futures = []
        for i in range(1000):
            request = ProcessRequest(
                id=f"stress_{i}",
                type=RequestType.EMBEDDING,
                priority=RequestPriority.NORMAL,
                data=f"压力测试请求 {i}",
                user_id=f"user_{i % 100}"  # 100个用户
            )
            future = await scheduler.submit_request(request)
            futures.append(future)
        
        submit_time = time.time() - start_time
        print(f"提交1000个请求耗时: {submit_time:.3f}s")
        
        # 等待处理完成
        await asyncio.sleep(5)
        
        # 统计结果
        stats = scheduler.get_statistics()
        total_time = time.time() - start_time
        
        print(f"总处理时间: {total_time:.3f}s")
        print(f"平均QPS: {stats['total_requests'] / total_time:.2f}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"平均批次大小: {stats['average_batch_size']:.2f}")
        
    finally:
        await scheduler.stop()

async def main():
    """主函数"""
    await basic_example()
    await advanced_example()
    await stress_test()

if __name__ == "__main__":
    asyncio.run(main())
