"""
集成路由服务

整合模型路由和成本优化功能
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
import json

from ..core.model_router import ModelRouter, RoutingDecision, TaskComplexity
from ..core.cost_optimizer import CostOptimizer, CostBudget

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """路由配置"""
    # 成本优先级 (0-1, 1为完全优先成本)
    cost_priority: float = 0.7
    
    # 预算配置
    daily_budget: float = 100.0
    monthly_budget: float = 2000.0
    per_user_budget: float = 10.0
    
    # 性能要求
    min_performance_score: float = 0.7
    max_latency_ms: float = 5000.0
    
    # 优化策略
    enable_auto_optimization: bool = True
    enable_fallback_routing: bool = True
    
    # 监控配置
    enable_cost_tracking: bool = True
    enable_performance_monitoring: bool = True


class IntelligentRoutingService:
    """智能路由服务"""
    
    def __init__(
        self,
        model_processors: Dict[str, Callable[[str, Dict[str, Any]], Awaitable[Any]]],
        config: Optional[RoutingConfig] = None
    ):
        self.model_processors = model_processors
        self.config = config or RoutingConfig()
        
        # 核心组件
        self.model_router = ModelRouter()
        
        # 成本优化器
        budget = CostBudget(
            daily_limit=self.config.daily_budget,
            monthly_limit=self.config.monthly_budget,
            per_user_limit=self.config.per_user_budget
        )
        self.cost_optimizer = CostOptimizer(budget)
        
        # 性能监控
        self.performance_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_cost': 0.0,
            'avg_response_time': 0.0,
            'fallback_used': 0,
            'optimization_triggered': 0
        }
        
        # 运行状态
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动路由服务"""
        if not self._running:
            self._running = True
            
            if self.config.enable_performance_monitoring:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Intelligent routing service started")
    
    async def stop(self):
        """停止路由服务"""
        if self._running:
            self._running = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Intelligent routing service stopped")
    
    async def route_and_process(
        self,
        content: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        override_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """路由并处理请求"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 检查预算限制
            if self.config.enable_cost_tracking and user_id:
                if await self._check_budget_limits(user_id):
                    return self._create_error_response(
                        "Budget limit exceeded",
                        "BUDGET_EXCEEDED"
                    )
            
            # 获取路由决策
            if override_model:
                # 使用指定模型
                routing_decision = self._create_override_decision(override_model, content)
            else:
                # 智能路由
                routing_decision = await self._get_routing_decision(content, context, user_id)
            
            # 检查模型可用性
            if routing_decision.selected_model not in self.model_processors:
                if self.config.enable_fallback_routing:
                    routing_decision = await self._get_fallback_decision(content, context)
                    self.stats['fallback_used'] += 1
                else:
                    return self._create_error_response(
                        f"Model {routing_decision.selected_model} not available",
                        "MODEL_UNAVAILABLE"
                    )
            
            # 执行处理
            response = await self._execute_processing(
                content, routing_decision, context or {}
            )
            
            # 记录成本和性能
            processing_time = time.time() - start_time
            await self._record_request_metrics(
                user_id, routing_decision, processing_time, True
            )
            
            self.stats['successful_requests'] += 1
            
            return {
                'success': True,
                'response': response,
                'routing_info': {
                    'selected_model': routing_decision.selected_model,
                    'confidence': routing_decision.confidence,
                    'estimated_cost': routing_decision.estimated_cost,
                    'reasoning': routing_decision.reasoning
                },
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['failed_requests'] += 1
            
            logger.error(f"Request processing failed: {e}")
            
            return self._create_error_response(
                str(e),
                "PROCESSING_ERROR",
                processing_time
            )
    
    async def _check_budget_limits(self, user_id: str) -> bool:
        """检查预算限制"""
        user_cost = self.cost_optimizer.cost_tracker.get_current_cost("per_user", user_id)
        daily_cost = self.cost_optimizer.cost_tracker.get_current_cost("daily")
        monthly_cost = self.cost_optimizer.cost_tracker.get_current_cost("monthly")
        
        budget = self.cost_optimizer.budget
        
        return (
            budget.is_exceeded(user_cost, "per_user") or
            budget.is_exceeded(daily_cost, "daily") or
            budget.is_exceeded(monthly_cost, "monthly")
        )
    
    async def _get_routing_decision(
        self,
        content: str,
        context: Optional[Dict[str, Any]],
        user_id: Optional[str]
    ) -> RoutingDecision:
        """获取路由决策"""
        
        # 检查是否需要成本优化
        cost_priority = self.config.cost_priority
        
        if self.config.enable_auto_optimization:
            # 根据当前预算使用情况调整成本优先级
            daily_cost = self.cost_optimizer.cost_tracker.get_current_cost("daily")
            daily_usage = daily_cost / self.cost_optimizer.budget.daily_limit
            
            if daily_usage > 0.8:
                cost_priority = min(1.0, cost_priority + 0.2)  # 提高成本优先级
                self.stats['optimization_triggered'] += 1
        
        # 构建路由上下文
        routing_context = context.copy() if context else {}
        if user_id:
            routing_context['user_id'] = user_id
        
        # 执行路由决策
        decision = self.model_router.route_request(
            content=content,
            context=routing_context,
            cost_priority=cost_priority
        )
        
        return decision
    
    def _create_override_decision(self, model: str, content: str) -> RoutingDecision:
        """创建覆盖决策"""
        model_config = self.model_router.models.get(model)
        
        if model_config:
            estimated_tokens = len(content) * 2  # 简单估算
            estimated_cost = model_config.calculate_cost(estimated_tokens)
        else:
            estimated_cost = 0.0
        
        return RoutingDecision(
            selected_model=model,
            confidence=1.0,
            reasoning=f"Manual override to {model}",
            estimated_cost=estimated_cost,
            estimated_performance=model_config.performance_score if model_config else 0.8,
            alternatives=[]
        )
    
    async def _get_fallback_decision(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> RoutingDecision:
        """获取回退决策"""
        # 寻找可用的模型
        available_models = [
            model for model in self.model_processors.keys()
            if model in self.model_router.models
        ]
        
        if not available_models:
            raise RuntimeError("No available models for fallback")
        
        # 选择成本最低的可用模型
        cheapest_model = min(
            available_models,
            key=lambda m: self.model_router.models[m].cost_per_1k_tokens
        )
        
        return self._create_override_decision(cheapest_model, content)
    
    async def _execute_processing(
        self,
        content: str,
        routing_decision: RoutingDecision,
        context: Dict[str, Any]
    ) -> Any:
        """执行处理"""
        model = routing_decision.selected_model
        processor = self.model_processors[model]
        
        # 准备处理参数
        processing_params = context.copy()
        processing_params['model'] = model
        
        # 执行处理
        response = await processor(content, processing_params)
        
        return response
    
    async def _record_request_metrics(
        self,
        user_id: Optional[str],
        routing_decision: RoutingDecision,
        processing_time: float,
        success: bool
    ):
        """记录请求指标"""
        
        # 记录成本
        if self.config.enable_cost_tracking:
            self.cost_optimizer.record_request_cost(
                user_id=user_id,
                model=routing_decision.selected_model,
                tokens_used=int(routing_decision.estimated_cost * 1000 / 0.002),  # 简化估算
                cost=routing_decision.estimated_cost,
                success=success
            )
            
            self.stats['total_cost'] += routing_decision.estimated_cost
        
        # 记录性能
        if self.config.enable_performance_monitoring:
            performance_record = {
                'timestamp': time.time(),
                'model': routing_decision.selected_model,
                'processing_time': processing_time,
                'estimated_cost': routing_decision.estimated_cost,
                'confidence': routing_decision.confidence,
                'success': success,
                'user_id': user_id
            }
            
            self.performance_history.append(performance_record)
            
            # 保持历史记录大小
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # 更新平均响应时间
            self._update_avg_response_time(processing_time)
    
    def _update_avg_response_time(self, processing_time: float):
        """更新平均响应时间"""
        if self.stats['successful_requests'] == 1:
            self.stats['avg_response_time'] = processing_time
        else:
            # 指数移动平均
            alpha = 0.1
            self.stats['avg_response_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['avg_response_time']
            )
    
    def _create_error_response(
        self,
        message: str,
        error_code: str,
        processing_time: float = 0.0
    ) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            'success': False,
            'error': {
                'message': message,
                'code': error_code
            },
            'processing_time': processing_time
        }
    
    async def _monitoring_loop(self):
        """监控循环"""
        logger.info("Routing service monitoring started")
        
        while self._running:
            try:
                await asyncio.sleep(30.0)  # 30秒监控间隔
                
                # 分析性能趋势
                await self._analyze_performance_trends()
                
                # 检查成本优化机会
                await self._check_optimization_opportunities()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
        
        logger.info("Routing service monitoring stopped")
    
    async def _analyze_performance_trends(self):
        """分析性能趋势"""
        if len(self.performance_history) < 10:
            return
        
        recent_records = self.performance_history[-20:]
        
        # 分析响应时间趋势
        response_times = [r['processing_time'] for r in recent_records if r['success']]
        if len(response_times) >= 5:
            recent_avg = sum(response_times[-5:]) / 5
            older_avg = sum(response_times[-10:-5]) / 5 if len(response_times) >= 10 else recent_avg
            
            if recent_avg > older_avg * 1.2:  # 响应时间增加20%
                logger.warning(f"Performance degradation detected: {recent_avg:.3f}s vs {older_avg:.3f}s")
        
        # 分析成功率
        success_rate = sum(1 for r in recent_records if r['success']) / len(recent_records)
        if success_rate < 0.95:
            logger.warning(f"Low success rate detected: {success_rate:.2%}")
    
    async def _check_optimization_opportunities(self):
        """检查优化机会"""
        # 获取优化建议
        recommendations = self.cost_optimizer.get_optimization_recommendations()
        
        if recommendations:
            high_priority_recs = [r for r in recommendations if r['priority'] == 'high']
            if high_priority_recs:
                logger.info(f"Found {len(high_priority_recs)} high-priority optimization opportunities")
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """获取路由分析"""
        # 模型使用统计
        model_usage = {}
        model_costs = {}
        model_performance = {}
        
        for record in self.performance_history:
            model = record['model']
            
            if model not in model_usage:
                model_usage[model] = 0
                model_costs[model] = 0.0
                model_performance[model] = []
            
            model_usage[model] += 1
            model_costs[model] += record['estimated_cost']
            if record['success']:
                model_performance[model].append(record['processing_time'])
        
        # 计算平均性能
        for model in model_performance:
            if model_performance[model]:
                model_performance[model] = sum(model_performance[model]) / len(model_performance[model])
            else:
                model_performance[model] = 0.0
        
        # 成本效率分析
        cost_efficiency = {}
        for model in model_usage:
            if model_costs[model] > 0 and model_performance[model] > 0:
                # 效率 = 请求数 / (成本 * 响应时间)
                cost_efficiency[model] = model_usage[model] / (model_costs[model] * model_performance[model])
            else:
                cost_efficiency[model] = 0.0
        
        return {
            'model_usage': model_usage,
            'model_costs': model_costs,
            'model_performance': model_performance,
            'cost_efficiency': cost_efficiency,
            'total_requests': len(self.performance_history),
            'avg_cost_per_request': sum(model_costs.values()) / len(self.performance_history) if self.performance_history else 0.0
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        # 基础统计
        success_rate = (
            self.stats['successful_requests'] / self.stats['total_requests']
            if self.stats['total_requests'] > 0 else 0.0
        )
        
        # 路由统计
        router_stats = self.model_router.get_stats()
        
        # 成本统计
        cost_stats = self.cost_optimizer.get_comprehensive_stats()
        
        # 路由分析
        routing_analytics = self.get_routing_analytics()
        
        return {
            'service_stats': {
                **self.stats,
                'success_rate': success_rate
            },
            'router_stats': router_stats,
            'cost_stats': cost_stats,
            'routing_analytics': routing_analytics,
            'config': {
                'cost_priority': self.config.cost_priority,
                'daily_budget': self.config.daily_budget,
                'monthly_budget': self.config.monthly_budget,
                'per_user_budget': self.config.per_user_budget
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查模型可用性
            available_models = list(self.model_processors.keys())
            
            # 测试路由决策
            test_decision = self.model_router.route_request(
                "Test query for health check",
                context={'health_check': True}
            )
            
            # 检查预算状态
            daily_cost = self.cost_optimizer.cost_tracker.get_current_cost("daily")
            daily_usage = daily_cost / self.cost_optimizer.budget.daily_limit
            
            return {
                'status': 'healthy',
                'available_models': available_models,
                'routing_functional': test_decision is not None,
                'daily_budget_usage': daily_usage,
                'avg_response_time': self.stats['avg_response_time'],
                'success_rate': (
                    self.stats['successful_requests'] / self.stats['total_requests']
                    if self.stats['total_requests'] > 0 else 1.0
                )
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 模拟模型处理器
    async def mock_gpt4_processor(content: str, params: Dict[str, Any]) -> str:
        await asyncio.sleep(0.8)  # 模拟处理时间
        return f"GPT-4 response to: {content}"
    
    async def mock_gpt35_processor(content: str, params: Dict[str, Any]) -> str:
        await asyncio.sleep(0.5)
        return f"GPT-3.5 response to: {content}"
    
    async def mock_claude_processor(content: str, params: Dict[str, Any]) -> str:
        await asyncio.sleep(0.3)
        return f"Claude response to: {content}"
    
    # 模型处理器映射
    model_processors = {
        'gpt-4': mock_gpt4_processor,
        'gpt-3.5-turbo': mock_gpt35_processor,
        'claude-3-haiku': mock_claude_processor
    }
    
    # 创建路由配置
    config = RoutingConfig(
        cost_priority=0.7,
        daily_budget=50.0,
        monthly_budget=1000.0,
        per_user_budget=5.0,
        enable_auto_optimization=True
    )
    
    # 创建路由服务
    routing_service = IntelligentRoutingService(
        model_processors=model_processors,
        config=config
    )
    
    await routing_service.start()
    
    try:
        print("🚀 Intelligent Routing Service Testing")
        print("=" * 60)
        
        # 测试不同复杂度的请求
        test_requests = [
            {
                'content': "Hello, how are you?",
                'user_id': 'user1',
                'expected_model': 'claude-3-haiku'  # 简单任务，便宜模型
            },
            {
                'content': "Explain the theory of relativity in detail",
                'user_id': 'user2',
                'expected_model': 'gpt-3.5-turbo'  # 中等任务
            },
            {
                'content': "Design a distributed system architecture for handling 1M concurrent users with real-time data processing and machine learning inference",
                'user_id': 'user3',
                'expected_model': 'gpt-4'  # 复杂任务
            }
        ]
        
        total_start_time = time.time()
        
        for i, request in enumerate(test_requests):
            print(f"\n--- Request {i+1} ---")
            print(f"Content: {request['content'][:80]}...")
            print(f"User: {request['user_id']}")
            
            start_time = time.time()
            result = await routing_service.route_and_process(
                content=request['content'],
                user_id=request['user_id']
            )
            response_time = time.time() - start_time
            
            if result['success']:
                routing_info = result['routing_info']
                print(f"✅ Success")
                print(f"Selected Model: {routing_info['selected_model']}")
                print(f"Confidence: {routing_info['confidence']:.2f}")
                print(f"Estimated Cost: ${routing_info['estimated_cost']:.4f}")
                print(f"Response Time: {response_time:.3f}s")
                print(f"Response: {result['response'][:100]}...")
            else:
                print(f"❌ Failed: {result['error']['message']}")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal test time: {total_time:.3f}s")
        
        # 获取统计信息
        stats = routing_service.get_comprehensive_stats()
        
        print(f"\n📊 Service Statistics:")
        print(f"Total Requests: {stats['service_stats']['total_requests']}")
        print(f"Success Rate: {stats['service_stats']['success_rate']:.2%}")
        print(f"Total Cost: ${stats['service_stats']['total_cost']:.4f}")
        print(f"Average Response Time: {stats['service_stats']['avg_response_time']:.3f}s")
        
        print(f"\n💰 Cost Analysis:")
        cost_savings = stats['router_stats']
        print(f"Cost Savings Rate: {cost_savings.get('savings_rate', 0):.1%}")
        print(f"Total Savings: ${cost_savings.get('total_savings', 0):.4f}")
        
        print(f"\n🎯 Model Usage:")
        for model, count in stats['router_stats']['routing_decisions'].items():
            print(f"  {model}: {count} requests")
        
        # 健康检查
        health = await routing_service.health_check()
        print(f"\n🏥 Health Check: {health['status']}")
        print(f"Available Models: {len(health['available_models'])}")
        print(f"Daily Budget Usage: {health['daily_budget_usage']:.1%}")
        
        print(f"\n✅ Routing service test completed!")
        
    finally:
        await routing_service.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
