"""
成本优化器

智能成本控制和优化策略
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json

logger = logging.getLogger(__name__)


@dataclass
class CostBudget:
    """成本预算"""
    daily_limit: float
    monthly_limit: float
    per_user_limit: float
    alert_threshold: float = 0.8  # 80%时告警
    
    def is_exceeded(self, current_cost: float, period: str) -> bool:
        """检查是否超出预算"""
        if period == "daily":
            return current_cost > self.daily_limit
        elif period == "monthly":
            return current_cost > self.monthly_limit
        elif period == "per_user":
            return current_cost > self.per_user_limit
        return False
    
    def should_alert(self, current_cost: float, period: str) -> bool:
        """检查是否应该告警"""
        if period == "daily":
            return current_cost > self.daily_limit * self.alert_threshold
        elif period == "monthly":
            return current_cost > self.monthly_limit * self.alert_threshold
        elif period == "per_user":
            return current_cost > self.per_user_limit * self.alert_threshold
        return False


@dataclass
class CostRecord:
    """成本记录"""
    timestamp: float
    user_id: Optional[str]
    model: str
    tokens_used: int
    cost: float
    request_type: str
    success: bool = True


@dataclass
class OptimizationStrategy:
    """优化策略"""
    name: str
    description: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 策略效果统计
    applications: int = 0
    cost_saved: float = 0.0
    performance_impact: float = 0.0


class CostTracker:
    """成本跟踪器"""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.cost_records: deque = deque()
        self.daily_costs: Dict[str, float] = defaultdict(float)  # date -> cost
        self.monthly_costs: Dict[str, float] = defaultdict(float)  # month -> cost
        self.user_costs: Dict[str, float] = defaultdict(float)  # user_id -> cost
        self.model_costs: Dict[str, float] = defaultdict(float)  # model -> cost
    
    def record_cost(self, record: CostRecord):
        """记录成本"""
        self.cost_records.append(record)
        
        # 更新各维度统计
        date_str = time.strftime("%Y-%m-%d", time.localtime(record.timestamp))
        month_str = time.strftime("%Y-%m", time.localtime(record.timestamp))
        
        self.daily_costs[date_str] += record.cost
        self.monthly_costs[month_str] += record.cost
        
        if record.user_id:
            self.user_costs[record.user_id] += record.cost
        
        self.model_costs[record.model] += record.cost
        
        # 清理过期记录
        self._cleanup_old_records()
    
    def _cleanup_old_records(self):
        """清理过期记录"""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        
        while self.cost_records and self.cost_records[0].timestamp < cutoff_time:
            old_record = self.cost_records.popleft()
            
            # 从统计中移除
            date_str = time.strftime("%Y-%m-%d", time.localtime(old_record.timestamp))
            month_str = time.strftime("%Y-%m", time.localtime(old_record.timestamp))
            
            self.daily_costs[date_str] -= old_record.cost
            if self.daily_costs[date_str] <= 0:
                del self.daily_costs[date_str]
            
            self.monthly_costs[month_str] -= old_record.cost
            if self.monthly_costs[month_str] <= 0:
                del self.monthly_costs[month_str]
            
            if old_record.user_id:
                self.user_costs[old_record.user_id] -= old_record.cost
                if self.user_costs[old_record.user_id] <= 0:
                    del self.user_costs[old_record.user_id]
            
            self.model_costs[old_record.model] -= old_record.cost
            if self.model_costs[old_record.model] <= 0:
                del self.model_costs[old_record.model]
    
    def get_current_cost(self, period: str, user_id: Optional[str] = None) -> float:
        """获取当前成本"""
        if period == "daily":
            today = time.strftime("%Y-%m-%d")
            return self.daily_costs.get(today, 0.0)
        elif period == "monthly":
            this_month = time.strftime("%Y-%m")
            return self.monthly_costs.get(this_month, 0.0)
        elif period == "per_user" and user_id:
            return self.user_costs.get(user_id, 0.0)
        return 0.0
    
    def get_cost_trend(self, days: int = 7) -> List[Tuple[str, float]]:
        """获取成本趋势"""
        trend = []
        for i in range(days):
            date = time.time() - (i * 24 * 3600)
            date_str = time.strftime("%Y-%m-%d", time.localtime(date))
            cost = self.daily_costs.get(date_str, 0.0)
            trend.append((date_str, cost))
        
        return list(reversed(trend))
    
    def get_top_cost_users(self, limit: int = 10) -> List[Tuple[str, float]]:
        """获取成本最高的用户"""
        return sorted(self.user_costs.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_model_cost_distribution(self) -> Dict[str, float]:
        """获取模型成本分布"""
        total_cost = sum(self.model_costs.values())
        if total_cost == 0:
            return {}
        
        return {
            model: cost / total_cost 
            for model, cost in self.model_costs.items()
        }


class CostOptimizer:
    """成本优化器"""
    
    def __init__(self, budget: CostBudget):
        self.budget = budget
        self.cost_tracker = CostTracker()
        
        # 优化策略
        self.strategies: Dict[str, OptimizationStrategy] = {}
        self._init_default_strategies()
        
        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            'total_cost_saved': 0.0,
            'total_optimizations': 0,
            'budget_alerts': 0,
            'cost_overruns': 0
        }
    
    def _init_default_strategies(self):
        """初始化默认优化策略"""
        strategies = [
            OptimizationStrategy(
                name="model_downgrade",
                description="在预算紧张时降级到更便宜的模型",
                parameters={
                    'trigger_threshold': 0.8,  # 预算使用80%时触发
                    'downgrade_mapping': {
                        'gpt-4': 'gpt-3.5-turbo',
                        'gpt-3.5-turbo': 'claude-3-haiku'
                    }
                }
            ),
            OptimizationStrategy(
                name="token_limit",
                description="限制输出token数量",
                parameters={
                    'max_tokens_reduction': 0.5,  # 减少50%
                    'trigger_threshold': 0.9
                }
            ),
            OptimizationStrategy(
                name="request_throttling",
                description="限制高成本用户的请求频率",
                parameters={
                    'cost_threshold': 10.0,  # 日成本超过$10
                    'throttle_rate': 0.5  # 限制到50%
                }
            ),
            OptimizationStrategy(
                name="batch_processing",
                description="将多个请求合并处理",
                parameters={
                    'min_batch_size': 3,
                    'max_wait_time': 2.0
                }
            ),
            OptimizationStrategy(
                name="cache_aggressive",
                description="激进缓存策略",
                parameters={
                    'similarity_threshold': 0.7,  # 降低相似度阈值
                    'ttl_extension': 2.0  # 延长缓存时间
                }
            )
        ]
        
        for strategy in strategies:
            self.strategies[strategy.name] = strategy
    
    def record_request_cost(
        self,
        user_id: Optional[str],
        model: str,
        tokens_used: int,
        cost: float,
        request_type: str = "chat",
        success: bool = True
    ):
        """记录请求成本"""
        record = CostRecord(
            timestamp=time.time(),
            user_id=user_id,
            model=model,
            tokens_used=tokens_used,
            cost=cost,
            request_type=request_type,
            success=success
        )
        
        self.cost_tracker.record_cost(record)
        
        # 检查预算告警
        self._check_budget_alerts(user_id)
    
    def _check_budget_alerts(self, user_id: Optional[str]):
        """检查预算告警"""
        # 检查日预算
        daily_cost = self.cost_tracker.get_current_cost("daily")
        if self.budget.should_alert(daily_cost, "daily"):
            self._trigger_alert("daily", daily_cost)
        
        # 检查月预算
        monthly_cost = self.cost_tracker.get_current_cost("monthly")
        if self.budget.should_alert(monthly_cost, "monthly"):
            self._trigger_alert("monthly", monthly_cost)
        
        # 检查用户预算
        if user_id:
            user_cost = self.cost_tracker.get_current_cost("per_user", user_id)
            if self.budget.should_alert(user_cost, "per_user"):
                self._trigger_alert("per_user", user_cost, user_id)
    
    def _trigger_alert(self, period: str, current_cost: float, user_id: Optional[str] = None):
        """触发预算告警"""
        self.stats['budget_alerts'] += 1
        
        alert_info = {
            'timestamp': time.time(),
            'period': period,
            'current_cost': current_cost,
            'user_id': user_id
        }
        
        logger.warning(f"Budget alert: {period} cost ${current_cost:.4f}")
        
        # 触发自动优化
        self._auto_optimize(period, current_cost, user_id)
    
    def _auto_optimize(self, period: str, current_cost: float, user_id: Optional[str] = None):
        """自动优化"""
        optimization_actions = []
        
        # 计算预算使用率
        if period == "daily":
            usage_rate = current_cost / self.budget.daily_limit
        elif period == "monthly":
            usage_rate = current_cost / self.budget.monthly_limit
        elif period == "per_user":
            usage_rate = current_cost / self.budget.per_user_limit
        else:
            usage_rate = 0.0
        
        # 根据使用率选择优化策略
        if usage_rate > 0.95:
            # 严重超预算，激进优化
            optimization_actions.extend([
                "model_downgrade",
                "token_limit",
                "request_throttling",
                "cache_aggressive"
            ])
        elif usage_rate > 0.85:
            # 接近预算，中等优化
            optimization_actions.extend([
                "model_downgrade",
                "batch_processing",
                "cache_aggressive"
            ])
        elif usage_rate > 0.75:
            # 预警阶段，轻度优化
            optimization_actions.extend([
                "batch_processing",
                "cache_aggressive"
            ])
        
        # 执行优化策略
        for action in optimization_actions:
            if action in self.strategies and self.strategies[action].enabled:
                self._apply_optimization_strategy(action, usage_rate, user_id)
    
    def _apply_optimization_strategy(
        self,
        strategy_name: str,
        usage_rate: float,
        user_id: Optional[str] = None
    ):
        """应用优化策略"""
        strategy = self.strategies[strategy_name]
        
        optimization_record = {
            'timestamp': time.time(),
            'strategy': strategy_name,
            'usage_rate': usage_rate,
            'user_id': user_id,
            'parameters': strategy.parameters.copy()
        }
        
        # 更新策略统计
        strategy.applications += 1
        
        # 记录优化历史
        self.optimization_history.append(optimization_record)
        
        # 保持历史记录大小
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
        
        self.stats['total_optimizations'] += 1
        
        logger.info(f"Applied optimization strategy: {strategy_name} for usage rate {usage_rate:.2%}")
    
    def get_optimization_recommendations(
        self,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取优化建议"""
        recommendations = []
        
        # 分析成本趋势
        cost_trend = self.cost_tracker.get_cost_trend(7)
        if len(cost_trend) >= 2:
            recent_costs = [cost for _, cost in cost_trend[-3:]]
            if recent_costs and statistics.mean(recent_costs) > 0:
                trend_direction = "increasing" if recent_costs[-1] > recent_costs[0] else "stable"
                
                if trend_direction == "increasing":
                    recommendations.append({
                        'type': 'cost_trend',
                        'priority': 'high',
                        'message': '成本呈上升趋势，建议启用成本优化策略',
                        'suggested_actions': ['model_downgrade', 'batch_processing']
                    })
        
        # 分析模型使用分布
        model_distribution = self.cost_tracker.get_model_cost_distribution()
        expensive_models = ['gpt-4', 'claude-3-opus']
        
        for model in expensive_models:
            if model in model_distribution and model_distribution[model] > 0.5:
                recommendations.append({
                    'type': 'model_usage',
                    'priority': 'medium',
                    'message': f'{model}使用占比过高({model_distribution[model]:.1%})，考虑部分场景使用更便宜的模型',
                    'suggested_actions': ['model_downgrade']
                })
        
        # 分析高成本用户
        if not user_id:  # 管理员视角
            top_users = self.cost_tracker.get_top_cost_users(5)
            if top_users:
                high_cost_users = [user for user, cost in top_users if cost > self.budget.per_user_limit * 0.8]
                if high_cost_users:
                    recommendations.append({
                        'type': 'high_cost_users',
                        'priority': 'high',
                        'message': f'发现{len(high_cost_users)}个高成本用户，建议实施用户级别优化',
                        'suggested_actions': ['request_throttling', 'token_limit']
                    })
        
        return recommendations
    
    def simulate_cost_savings(self, strategy_name: str) -> Dict[str, Any]:
        """模拟成本节省效果"""
        if strategy_name not in self.strategies:
            return {'error': 'Unknown strategy'}
        
        strategy = self.strategies[strategy_name]
        
        # 基于历史数据模拟
        recent_records = list(self.cost_tracker.cost_records)[-100:]  # 最近100条记录
        
        if not recent_records:
            return {'estimated_savings': 0.0, 'confidence': 0.0}
        
        total_original_cost = sum(r.cost for r in recent_records)
        estimated_savings = 0.0
        
        if strategy_name == "model_downgrade":
            # 模拟模型降级节省
            downgrade_mapping = strategy.parameters.get('downgrade_mapping', {})
            for record in recent_records:
                if record.model in downgrade_mapping:
                    # 假设降级模型成本为原来的20%
                    estimated_savings += record.cost * 0.8
        
        elif strategy_name == "token_limit":
            # 模拟token限制节省
            reduction_rate = strategy.parameters.get('max_tokens_reduction', 0.5)
            estimated_savings = total_original_cost * reduction_rate
        
        elif strategy_name == "batch_processing":
            # 模拟批处理节省
            estimated_savings = total_original_cost * 0.3  # 假设节省30%
        
        elif strategy_name == "cache_aggressive":
            # 模拟激进缓存节省
            estimated_savings = total_original_cost * 0.4  # 假设节省40%
        
        savings_rate = estimated_savings / total_original_cost if total_original_cost > 0 else 0.0
        
        return {
            'estimated_savings': estimated_savings,
            'savings_rate': savings_rate,
            'confidence': 0.7,  # 模拟置信度
            'based_on_records': len(recent_records)
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        # 基础统计
        daily_cost = self.cost_tracker.get_current_cost("daily")
        monthly_cost = self.cost_tracker.get_current_cost("monthly")
        
        # 预算使用率
        daily_usage = daily_cost / self.budget.daily_limit if self.budget.daily_limit > 0 else 0
        monthly_usage = monthly_cost / self.budget.monthly_limit if self.budget.monthly_limit > 0 else 0
        
        # 成本趋势
        cost_trend = self.cost_tracker.get_cost_trend(7)
        
        # 策略效果
        strategy_stats = {}
        for name, strategy in self.strategies.items():
            strategy_stats[name] = {
                'applications': strategy.applications,
                'cost_saved': strategy.cost_saved,
                'enabled': strategy.enabled
            }
        
        return {
            'current_costs': {
                'daily': daily_cost,
                'monthly': monthly_cost,
                'daily_usage_rate': daily_usage,
                'monthly_usage_rate': monthly_usage
            },
            'budget': {
                'daily_limit': self.budget.daily_limit,
                'monthly_limit': self.budget.monthly_limit,
                'per_user_limit': self.budget.per_user_limit
            },
            'cost_trend': cost_trend,
            'model_distribution': self.cost_tracker.get_model_cost_distribution(),
            'top_users': self.cost_tracker.get_top_cost_users(10),
            'optimization_stats': {
                **self.stats,
                'strategies': strategy_stats
            },
            'recommendations': self.get_optimization_recommendations()
        }


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 创建预算配置
    budget = CostBudget(
        daily_limit=100.0,    # $100/天
        monthly_limit=2000.0, # $2000/月
        per_user_limit=10.0   # $10/用户
    )
    
    # 创建成本优化器
    optimizer = CostOptimizer(budget)
    
    print("🚀 Cost Optimizer Testing")
    print("=" * 50)
    
    # 模拟一些请求成本
    test_requests = [
        {'user_id': 'user1', 'model': 'gpt-4', 'tokens': 1000, 'cost': 0.03},
        {'user_id': 'user1', 'model': 'gpt-4', 'tokens': 1500, 'cost': 0.045},
        {'user_id': 'user2', 'model': 'gpt-3.5-turbo', 'tokens': 800, 'cost': 0.0016},
        {'user_id': 'user2', 'model': 'gpt-3.5-turbo', 'tokens': 1200, 'cost': 0.0024},
        {'user_id': 'user3', 'model': 'claude-3-haiku', 'tokens': 2000, 'cost': 0.0005},
    ]
    
    # 记录成本
    for req in test_requests:
        optimizer.record_request_cost(
            user_id=req['user_id'],
            model=req['model'],
            tokens_used=req['tokens'],
            cost=req['cost']
        )
    
    # 获取统计信息
    stats = optimizer.get_comprehensive_stats()
    
    print(f"\n📊 Cost Statistics:")
    print(f"Daily Cost: ${stats['current_costs']['daily']:.4f}")
    print(f"Daily Usage: {stats['current_costs']['daily_usage_rate']:.1%}")
    print(f"Monthly Cost: ${stats['current_costs']['monthly']:.4f}")
    print(f"Monthly Usage: {stats['current_costs']['monthly_usage_rate']:.1%}")
    
    print(f"\n🏷️ Model Distribution:")
    for model, percentage in stats['model_distribution'].items():
        print(f"  {model}: {percentage:.1%}")
    
    print(f"\n👥 Top Users:")
    for user_id, cost in stats['top_users'][:3]:
        print(f"  {user_id}: ${cost:.4f}")
    
    print(f"\n💡 Optimization Recommendations:")
    for rec in stats['recommendations']:
        print(f"  [{rec['priority'].upper()}] {rec['message']}")
    
    # 模拟成本节省
    print(f"\n💰 Cost Savings Simulation:")
    for strategy_name in ['model_downgrade', 'batch_processing', 'cache_aggressive']:
        simulation = optimizer.simulate_cost_savings(strategy_name)
        if 'error' not in simulation:
            print(f"  {strategy_name}: ${simulation['estimated_savings']:.4f} ({simulation['savings_rate']:.1%})")
    
    print(f"\n✅ Cost optimizer test completed!")


if __name__ == "__main__":
    asyncio.run(example_usage())
