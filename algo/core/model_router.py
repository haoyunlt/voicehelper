"""
动态模型路由系统

智能选择最优模型，降低30-50%成本
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import statistics

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"           # 简单任务
    MODERATE = "moderate"       # 中等任务
    COMPLEX = "complex"         # 复杂任务
    VERY_COMPLEX = "very_complex"  # 非常复杂任务


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    cost_per_1k_tokens: float  # 每1K token的成本
    max_tokens: int
    capabilities: List[str]     # 模型能力
    performance_score: float    # 性能评分 (0-1)
    latency_ms: float          # 平均延迟 (毫秒)
    suitable_complexity: List[TaskComplexity]  # 适合的任务复杂度
    
    def calculate_cost(self, token_count: int) -> float:
        """计算成本"""
        return (token_count / 1000.0) * self.cost_per_1k_tokens
    
    def get_cost_efficiency(self, performance_requirement: float) -> float:
        """获取成本效率 (性能/成本比)"""
        if self.cost_per_1k_tokens == 0:
            return float('inf')
        
        # 如果性能不满足要求，效率为0
        if self.performance_score < performance_requirement:
            return 0.0
        
        return self.performance_score / self.cost_per_1k_tokens


@dataclass
class RoutingDecision:
    """路由决策"""
    selected_model: str
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_performance: float
    alternatives: List[Tuple[str, float]]  # (model_name, score)


@dataclass
class TaskAnalysis:
    """任务分析结果"""
    complexity: TaskComplexity
    estimated_tokens: int
    required_capabilities: List[str]
    performance_requirement: float
    latency_requirement: float
    reasoning: str


class TaskAnalyzer:
    """任务分析器"""
    
    def __init__(self):
        # 复杂度检测规则
        self.complexity_patterns = {
            TaskComplexity.SIMPLE: [
                r'\b(hello|hi|thanks|bye)\b',
                r'\bwhat\s+is\s+\w+\?',
                r'\bdefine\s+\w+',
                r'\byes|no\b',
                r'^.{1,20}$'  # 很短的文本
            ],
            TaskComplexity.MODERATE: [
                r'\bhow\s+to\s+',
                r'\bexplain\s+',
                r'\bcompare\s+',
                r'\blist\s+',
                r'\bsummarize\s+',
                r'^.{21,100}$'  # 中等长度文本
            ],
            TaskComplexity.COMPLEX: [
                r'\banalyze\s+',
                r'\bevaluate\s+',
                r'\bwrite\s+.*\b(essay|article|report)\b',
                r'\bcreate\s+.*\b(plan|strategy)\b',
                r'\bsolve\s+.*\b(problem|equation)\b',
                r'^.{101,300}$'  # 较长文本
            ],
            TaskComplexity.VERY_COMPLEX: [
                r'\bdesign\s+.*\b(system|architecture)\b',
                r'\bdevelop\s+.*\b(algorithm|model)\b',
                r'\bresearch\s+',
                r'\boptimize\s+',
                r'^.{301,}$'  # 很长文本
            ]
        }
        
        # 能力需求检测
        self.capability_patterns = {
            'reasoning': [r'\banalyze\b', r'\bevaluate\b', r'\bcompare\b', r'\breason\b'],
            'coding': [r'\bcode\b', r'\bprogram\b', r'\bfunction\b', r'\balgorithm\b'],
            'math': [r'\bcalculate\b', r'\bsolve\b', r'\bequation\b', r'\bmathematics\b'],
            'creative': [r'\bwrite\b', r'\bcreate\b', r'\bgenerate\b', r'\bimagine\b'],
            'translation': [r'\btranslate\b', r'\btranslation\b'],
            'summarization': [r'\bsummarize\b', r'\bsummary\b', r'\babstract\b']
        }
    
    def analyze_task(self, content: str, context: Optional[Dict[str, Any]] = None) -> TaskAnalysis:
        """分析任务"""
        content_lower = content.lower()
        
        # 检测复杂度
        complexity = self._detect_complexity(content_lower)
        
        # 估算token数量
        estimated_tokens = self._estimate_tokens(content)
        
        # 检测所需能力
        required_capabilities = self._detect_capabilities(content_lower)
        
        # 确定性能要求
        performance_requirement = self._determine_performance_requirement(
            complexity, required_capabilities, context
        )
        
        # 确定延迟要求
        latency_requirement = self._determine_latency_requirement(context)
        
        # 生成推理说明
        reasoning = self._generate_reasoning(
            complexity, required_capabilities, estimated_tokens
        )
        
        return TaskAnalysis(
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            required_capabilities=required_capabilities,
            performance_requirement=performance_requirement,
            latency_requirement=latency_requirement,
            reasoning=reasoning
        )
    
    def _detect_complexity(self, content: str) -> TaskComplexity:
        """检测任务复杂度"""
        complexity_scores = {complexity: 0 for complexity in TaskComplexity}
        
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    complexity_scores[complexity] += 1
        
        # 返回得分最高的复杂度
        max_complexity = max(complexity_scores.items(), key=lambda x: x[1])
        
        # 如果没有匹配到任何模式，根据长度判断
        if max_complexity[1] == 0:
            if len(content) < 50:
                return TaskComplexity.SIMPLE
            elif len(content) < 200:
                return TaskComplexity.MODERATE
            else:
                return TaskComplexity.COMPLEX
        
        return max_complexity[0]
    
    def _estimate_tokens(self, content: str) -> int:
        """估算token数量"""
        # 简单估算：英文约4字符/token，中文约1.5字符/token
        char_count = len(content)
        
        # 检测中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        chinese_ratio = chinese_chars / char_count if char_count > 0 else 0
        
        # 混合估算
        if chinese_ratio > 0.5:
            estimated_tokens = int(char_count / 1.5)
        else:
            estimated_tokens = int(char_count / 4)
        
        # 考虑输出token (通常是输入的1-3倍)
        total_tokens = estimated_tokens * 2.5
        
        return int(total_tokens)
    
    def _detect_capabilities(self, content: str) -> List[str]:
        """检测所需能力"""
        required_capabilities = []
        
        for capability, patterns in self.capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    required_capabilities.append(capability)
                    break
        
        return required_capabilities
    
    def _determine_performance_requirement(
        self, 
        complexity: TaskComplexity, 
        capabilities: List[str], 
        context: Optional[Dict[str, Any]]
    ) -> float:
        """确定性能要求"""
        base_requirements = {
            TaskComplexity.SIMPLE: 0.6,
            TaskComplexity.MODERATE: 0.7,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.VERY_COMPLEX: 0.9
        }
        
        requirement = base_requirements[complexity]
        
        # 特殊能力需求调整
        if 'reasoning' in capabilities or 'math' in capabilities:
            requirement += 0.1
        
        if 'coding' in capabilities:
            requirement += 0.05
        
        # 上下文调整
        if context:
            if context.get('user_tier') == 'premium':
                requirement += 0.1
            elif context.get('user_tier') == 'free':
                requirement -= 0.1
        
        return min(1.0, max(0.5, requirement))
    
    def _determine_latency_requirement(self, context: Optional[Dict[str, Any]]) -> float:
        """确定延迟要求 (毫秒)"""
        if context:
            if context.get('real_time', False):
                return 500.0  # 实时场景要求500ms内
            elif context.get('interactive', True):
                return 2000.0  # 交互场景要求2s内
        
        return 5000.0  # 默认5s内
    
    def _generate_reasoning(
        self, 
        complexity: TaskComplexity, 
        capabilities: List[str], 
        tokens: int
    ) -> str:
        """生成推理说明"""
        reasoning_parts = [
            f"任务复杂度: {complexity.value}",
            f"预估token数: {tokens}",
        ]
        
        if capabilities:
            reasoning_parts.append(f"所需能力: {', '.join(capabilities)}")
        
        return "; ".join(reasoning_parts)


class ModelRouter:
    """模型路由器"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.task_analyzer = TaskAnalyzer()
        self.routing_history: List[Dict[str, Any]] = []
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'cost_savings': 0.0,
            'performance_maintained': 0,
            'routing_decisions': {}
        }
        
        # 初始化默认模型
        self._init_default_models()
    
    def _init_default_models(self):
        """初始化默认模型配置"""
        models = [
            ModelConfig(
                name="gpt-4",
                cost_per_1k_tokens=0.03,
                max_tokens=8192,
                capabilities=['reasoning', 'coding', 'math', 'creative', 'translation', 'summarization'],
                performance_score=0.95,
                latency_ms=2000,
                suitable_complexity=[TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]
            ),
            ModelConfig(
                name="gpt-3.5-turbo",
                cost_per_1k_tokens=0.002,
                max_tokens=4096,
                capabilities=['reasoning', 'coding', 'creative', 'translation', 'summarization'],
                performance_score=0.85,
                latency_ms=1000,
                suitable_complexity=[TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
            ),
            ModelConfig(
                name="claude-3-haiku",
                cost_per_1k_tokens=0.00025,
                max_tokens=200000,
                capabilities=['reasoning', 'creative', 'translation', 'summarization'],
                performance_score=0.75,
                latency_ms=800,
                suitable_complexity=[TaskComplexity.SIMPLE, TaskComplexity.MODERATE]
            ),
            ModelConfig(
                name="gemini-pro",
                cost_per_1k_tokens=0.001,
                max_tokens=32768,
                capabilities=['reasoning', 'coding', 'math', 'creative'],
                performance_score=0.80,
                latency_ms=1200,
                suitable_complexity=[TaskComplexity.MODERATE, TaskComplexity.COMPLEX]
            )
        ]
        
        for model in models:
            self.models[model.name] = model
    
    def add_model(self, model_config: ModelConfig):
        """添加模型配置"""
        self.models[model_config.name] = model_config
        logger.info(f"Added model: {model_config.name}")
    
    def route_request(
        self, 
        content: str, 
        context: Optional[Dict[str, Any]] = None,
        cost_priority: float = 0.7  # 成本优先级 (0-1, 1为完全优先成本)
    ) -> RoutingDecision:
        """路由请求到最优模型"""
        self.stats['total_requests'] += 1
        
        # 分析任务
        task_analysis = self.task_analyzer.analyze_task(content, context)
        
        # 筛选候选模型
        candidate_models = self._filter_candidate_models(task_analysis)
        
        if not candidate_models:
            # 如果没有候选模型，使用默认模型
            default_model = "gpt-3.5-turbo"
            return RoutingDecision(
                selected_model=default_model,
                confidence=0.5,
                reasoning="No suitable models found, using default",
                estimated_cost=self.models[default_model].calculate_cost(task_analysis.estimated_tokens),
                estimated_performance=self.models[default_model].performance_score,
                alternatives=[]
            )
        
        # 评分和选择
        model_scores = self._score_models(candidate_models, task_analysis, cost_priority)
        
        # 选择最佳模型
        best_model, best_score = max(model_scores.items(), key=lambda x: x[1])
        
        # 计算置信度
        confidence = self._calculate_confidence(model_scores, best_model)
        
        # 生成推理说明
        reasoning = self._generate_routing_reasoning(
            best_model, task_analysis, model_scores
        )
        
        # 计算预估成本和性能
        estimated_cost = self.models[best_model].calculate_cost(task_analysis.estimated_tokens)
        estimated_performance = self.models[best_model].performance_score
        
        # 生成备选方案
        alternatives = sorted(
            [(model, score) for model, score in model_scores.items() if model != best_model],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        decision = RoutingDecision(
            selected_model=best_model,
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            estimated_performance=estimated_performance,
            alternatives=alternatives
        )
        
        # 记录路由历史
        self._record_routing_decision(task_analysis, decision)
        
        return decision
    
    def _filter_candidate_models(self, task_analysis: TaskAnalysis) -> List[str]:
        """筛选候选模型"""
        candidates = []
        
        for model_name, model_config in self.models.items():
            # 检查复杂度匹配
            if task_analysis.complexity not in model_config.suitable_complexity:
                continue
            
            # 检查token限制
            if task_analysis.estimated_tokens > model_config.max_tokens:
                continue
            
            # 检查性能要求
            if model_config.performance_score < task_analysis.performance_requirement:
                continue
            
            # 检查延迟要求
            if model_config.latency_ms > task_analysis.latency_requirement:
                continue
            
            # 检查能力要求
            missing_capabilities = set(task_analysis.required_capabilities) - set(model_config.capabilities)
            if missing_capabilities:
                continue
            
            candidates.append(model_name)
        
        return candidates
    
    def _score_models(
        self, 
        candidates: List[str], 
        task_analysis: TaskAnalysis, 
        cost_priority: float
    ) -> Dict[str, float]:
        """为候选模型评分"""
        scores = {}
        
        for model_name in candidates:
            model_config = self.models[model_name]
            
            # 性能评分 (0-1)
            performance_score = model_config.performance_score
            
            # 成本效率评分 (0-1)
            cost_efficiency = model_config.get_cost_efficiency(task_analysis.performance_requirement)
            # 标准化成本效率评分
            max_efficiency = max(
                self.models[m].get_cost_efficiency(task_analysis.performance_requirement) 
                for m in candidates
            )
            if max_efficiency > 0:
                cost_score = cost_efficiency / max_efficiency
            else:
                cost_score = 0.0
            
            # 延迟评分 (0-1, 延迟越低评分越高)
            max_latency = max(self.models[m].latency_ms for m in candidates)
            latency_score = 1.0 - (model_config.latency_ms / max_latency) if max_latency > 0 else 1.0
            
            # 综合评分
            final_score = (
                cost_priority * cost_score +
                (1 - cost_priority) * 0.7 * performance_score +
                (1 - cost_priority) * 0.3 * latency_score
            )
            
            scores[model_name] = final_score
        
        return scores
    
    def _calculate_confidence(self, model_scores: Dict[str, float], best_model: str) -> float:
        """计算路由决策的置信度"""
        if len(model_scores) < 2:
            return 1.0
        
        scores = list(model_scores.values())
        best_score = model_scores[best_model]
        
        # 计算与第二名的差距
        scores.remove(best_score)
        second_best = max(scores)
        
        # 置信度基于分数差距
        confidence = min(1.0, (best_score - second_best) / best_score + 0.5)
        
        return confidence
    
    def _generate_routing_reasoning(
        self, 
        selected_model: str, 
        task_analysis: TaskAnalysis, 
        model_scores: Dict[str, float]
    ) -> str:
        """生成路由推理说明"""
        model_config = self.models[selected_model]
        
        reasoning_parts = [
            f"选择 {selected_model}",
            f"任务复杂度: {task_analysis.complexity.value}",
            f"预估成本: ${model_config.calculate_cost(task_analysis.estimated_tokens):.4f}",
            f"性能评分: {model_config.performance_score:.2f}"
        ]
        
        if len(model_scores) > 1:
            reasoning_parts.append(f"候选模型数: {len(model_scores)}")
        
        return "; ".join(reasoning_parts)
    
    def _record_routing_decision(self, task_analysis: TaskAnalysis, decision: RoutingDecision):
        """记录路由决策"""
        record = {
            'timestamp': time.time(),
            'task_complexity': task_analysis.complexity.value,
            'estimated_tokens': task_analysis.estimated_tokens,
            'selected_model': decision.selected_model,
            'estimated_cost': decision.estimated_cost,
            'confidence': decision.confidence
        }
        
        self.routing_history.append(record)
        
        # 保持历史记录大小
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        # 更新统计
        model_name = decision.selected_model
        if model_name not in self.stats['routing_decisions']:
            self.stats['routing_decisions'][model_name] = 0
        self.stats['routing_decisions'][model_name] += 1
    
    def calculate_cost_savings(self, baseline_model: str = "gpt-4") -> Dict[str, Any]:
        """计算成本节省"""
        if not self.routing_history:
            return {'total_savings': 0.0, 'savings_rate': 0.0}
        
        baseline_config = self.models.get(baseline_model)
        if not baseline_config:
            return {'total_savings': 0.0, 'savings_rate': 0.0}
        
        total_actual_cost = 0.0
        total_baseline_cost = 0.0
        
        for record in self.routing_history:
            actual_cost = record['estimated_cost']
            baseline_cost = baseline_config.calculate_cost(record['estimated_tokens'])
            
            total_actual_cost += actual_cost
            total_baseline_cost += baseline_cost
        
        total_savings = total_baseline_cost - total_actual_cost
        savings_rate = total_savings / total_baseline_cost if total_baseline_cost > 0 else 0.0
        
        return {
            'total_savings': total_savings,
            'savings_rate': savings_rate,
            'total_actual_cost': total_actual_cost,
            'total_baseline_cost': total_baseline_cost
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        cost_savings = self.calculate_cost_savings()
        
        # 计算平均置信度
        if self.routing_history:
            avg_confidence = statistics.mean(r['confidence'] for r in self.routing_history)
        else:
            avg_confidence = 0.0
        
        return {
            **self.stats,
            **cost_savings,
            'avg_confidence': avg_confidence,
            'total_models': len(self.models),
            'routing_history_size': len(self.routing_history)
        }


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 创建模型路由器
    router = ModelRouter()
    
    # 测试不同复杂度的任务
    test_cases = [
        {
            'content': "Hello, how are you?",
            'context': {'user_tier': 'free'},
            'expected_complexity': TaskComplexity.SIMPLE
        },
        {
            'content': "Can you explain how machine learning works?",
            'context': {'user_tier': 'standard'},
            'expected_complexity': TaskComplexity.MODERATE
        },
        {
            'content': "Write a comprehensive analysis of the economic impact of artificial intelligence on the job market, including statistical data and policy recommendations.",
            'context': {'user_tier': 'premium'},
            'expected_complexity': TaskComplexity.COMPLEX
        },
        {
            'content': "Design a distributed system architecture for a real-time recommendation engine that can handle 1 million concurrent users with sub-100ms latency.",
            'context': {'user_tier': 'enterprise'},
            'expected_complexity': TaskComplexity.VERY_COMPLEX
        }
    ]
    
    print("🚀 Model Router Testing")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Content: {test_case['content'][:100]}...")
        
        # 路由决策
        decision = router.route_request(
            content=test_case['content'],
            context=test_case['context'],
            cost_priority=0.7
        )
        
        print(f"Selected Model: {decision.selected_model}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Estimated Cost: ${decision.estimated_cost:.4f}")
        print(f"Reasoning: {decision.reasoning}")
        
        if decision.alternatives:
            print(f"Alternatives: {', '.join([f'{m}({s:.2f})' for m, s in decision.alternatives])}")
    
    # 获取统计信息
    stats = router.get_stats()
    print(f"\n📊 Router Statistics:")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Cost Savings Rate: {stats['savings_rate']:.1%}")
    print(f"Total Savings: ${stats['total_savings']:.4f}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Model Usage: {stats['routing_decisions']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
