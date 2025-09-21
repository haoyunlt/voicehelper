"""
因果推理模块 - 实现因果关系分析和推理能力
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """因果关系类型"""
    DIRECT_CAUSE = "direct_cause"          # 直接因果
    INDIRECT_CAUSE = "indirect_cause"      # 间接因果
    NECESSARY_CAUSE = "necessary_cause"    # 必要条件
    SUFFICIENT_CAUSE = "sufficient_cause"  # 充分条件
    CONTRIBUTORY_CAUSE = "contributory_cause"  # 促成因素
    PREVENTIVE_CAUSE = "preventive_cause"  # 阻止因素


@dataclass
class CausalNode:
    """因果图中的节点"""
    id: str
    name: str
    description: str
    node_type: str  # event, condition, action, outcome
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    temporal_info: Optional[Dict[str, Any]] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class CausalEdge:
    """因果图中的边"""
    source: str
    target: str
    relation_type: CausalRelationType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    conditions: List[str] = field(default_factory=list)
    temporal_delay: Optional[float] = None  # 时间延迟（秒）
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalQuery:
    """因果查询"""
    query_id: str
    query_type: str  # "what_if", "why", "how", "counterfactual"
    target_event: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CausalExplanation:
    """因果解释"""
    query_id: str
    explanation_type: str
    causal_chain: List[str]
    confidence: float
    reasoning_steps: List[str]
    evidence: List[str]
    alternative_explanations: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class CausalGraph:
    """因果图"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.temporal_constraints: List[Dict[str, Any]] = []
        
    def add_node(self, node: CausalNode):
        """添加节点"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.properties)
        
    def add_edge(self, edge: CausalEdge):
        """添加边"""
        key = (edge.source, edge.target)
        self.edges[key] = edge
        self.graph.add_edge(
            edge.source, 
            edge.target,
            relation_type=edge.relation_type,
            strength=edge.strength,
            confidence=edge.confidence
        )
        
    def get_causal_paths(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """获取因果路径"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths
        except nx.NetworkXNoPath:
            return []
            
    def get_direct_causes(self, node_id: str) -> List[str]:
        """获取直接原因"""
        return list(self.graph.predecessors(node_id))
        
    def get_direct_effects(self, node_id: str) -> List[str]:
        """获取直接结果"""
        return list(self.graph.successors(node_id))
        
    def get_root_causes(self, node_id: str) -> List[str]:
        """获取根本原因"""
        root_causes = []
        visited = set()
        
        def dfs(current):
            if current in visited:
                return
            visited.add(current)
            
            predecessors = list(self.graph.predecessors(current))
            if not predecessors:
                root_causes.append(current)
            else:
                for pred in predecessors:
                    dfs(pred)
                    
        dfs(node_id)
        return root_causes
        
    def calculate_causal_strength(self, source: str, target: str) -> float:
        """计算因果强度"""
        paths = self.get_causal_paths(source, target)
        if not paths:
            return 0.0
            
        max_strength = 0.0
        for path in paths:
            path_strength = 1.0
            for i in range(len(path) - 1):
                edge_key = (path[i], path[i + 1])
                if edge_key in self.edges:
                    edge_strength = self.edges[edge_key].strength
                    path_strength *= edge_strength
                else:
                    path_strength = 0.0
                    break
            max_strength = max(max_strength, path_strength)
            
        return max_strength


class CausalReasoningEngine:
    """因果推理引擎"""
    
    def __init__(self):
        self.causal_graph = CausalGraph()
        self.knowledge_base: Dict[str, Any] = {}
        self.reasoning_rules: List[Dict[str, Any]] = []
        self.temporal_model: Dict[str, Any] = {}
        
        # 初始化推理规则
        self._initialize_reasoning_rules()
        
    def _initialize_reasoning_rules(self):
        """初始化推理规则"""
        self.reasoning_rules = [
            {
                "name": "transitivity",
                "description": "如果A导致B，B导致C，则A间接导致C",
                "pattern": "A -> B -> C",
                "conclusion": "A indirectly causes C"
            },
            {
                "name": "common_cause",
                "description": "如果A导致B和C，则B和C可能相关但不互为因果",
                "pattern": "A -> B, A -> C",
                "conclusion": "B and C are correlated but not causal"
            },
            {
                "name": "necessary_condition",
                "description": "如果没有A就没有B，则A是B的必要条件",
                "pattern": "¬A -> ¬B",
                "conclusion": "A is necessary for B"
            },
            {
                "name": "sufficient_condition",
                "description": "如果有A就有B，则A是B的充分条件",
                "pattern": "A -> B (always)",
                "conclusion": "A is sufficient for B"
            }
        ]
        
    async def build_causal_model(self, domain_knowledge: Dict[str, Any]) -> CausalGraph:
        """构建因果模型"""
        logger.info("Building causal model from domain knowledge")
        
        # 提取实体和事件
        entities = domain_knowledge.get("entities", [])
        events = domain_knowledge.get("events", [])
        relationships = domain_knowledge.get("relationships", [])
        
        # 添加节点
        for entity in entities:
            node = CausalNode(
                id=entity["id"],
                name=entity["name"],
                description=entity.get("description", ""),
                node_type="entity",
                properties=entity.get("properties", {}),
                confidence=entity.get("confidence", 0.8)
            )
            self.causal_graph.add_node(node)
            
        for event in events:
            node = CausalNode(
                id=event["id"],
                name=event["name"],
                description=event.get("description", ""),
                node_type="event",
                properties=event.get("properties", {}),
                confidence=event.get("confidence", 0.8),
                temporal_info=event.get("temporal_info")
            )
            self.causal_graph.add_node(node)
            
        # 添加因果关系
        for rel in relationships:
            if rel.get("type") == "causal":
                edge = CausalEdge(
                    source=rel["source"],
                    target=rel["target"],
                    relation_type=CausalRelationType(rel.get("relation_type", "direct_cause")),
                    strength=rel.get("strength", 0.7),
                    confidence=rel.get("confidence", 0.8),
                    conditions=rel.get("conditions", []),
                    temporal_delay=rel.get("temporal_delay"),
                    evidence=rel.get("evidence", [])
                )
                self.causal_graph.add_edge(edge)
                
        return self.causal_graph
        
    async def analyze_causal_query(self, query: CausalQuery) -> CausalExplanation:
        """分析因果查询"""
        logger.info(f"Analyzing causal query: {query.query_type} for {query.target_event}")
        
        if query.query_type == "why":
            return await self._analyze_why_query(query)
        elif query.query_type == "what_if":
            return await self._analyze_what_if_query(query)
        elif query.query_type == "how":
            return await self._analyze_how_query(query)
        elif query.query_type == "counterfactual":
            return await self._analyze_counterfactual_query(query)
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}")
            
    async def _analyze_why_query(self, query: CausalQuery) -> CausalExplanation:
        """分析"为什么"查询"""
        target_event = query.target_event
        
        # 获取所有可能的原因
        direct_causes = self.causal_graph.get_direct_causes(target_event)
        root_causes = self.causal_graph.get_root_causes(target_event)
        
        # 构建因果链
        causal_chains = []
        for root_cause in root_causes:
            paths = self.causal_graph.get_causal_paths(root_cause, target_event)
            for path in paths:
                strength = self._calculate_path_strength(path)
                causal_chains.append({
                    "path": path,
                    "strength": strength,
                    "explanation": self._generate_path_explanation(path)
                })
                
        # 选择最强的因果链
        if causal_chains:
            best_chain = max(causal_chains, key=lambda x: x["strength"])
            causal_chain = best_chain["path"]
            confidence = best_chain["strength"]
            reasoning_steps = [best_chain["explanation"]]
        else:
            causal_chain = [target_event]
            confidence = 0.1
            reasoning_steps = ["No clear causal path found"]
            
        # 收集证据
        evidence = self._collect_evidence(causal_chain)
        
        return CausalExplanation(
            query_id=query.query_id,
            explanation_type="why",
            causal_chain=causal_chain,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            evidence=evidence,
            alternative_explanations=[
                {"path": chain["path"], "strength": chain["strength"]} 
                for chain in causal_chains[1:6]  # 前5个替代解释
            ]
        )
        
    async def _analyze_what_if_query(self, query: CausalQuery) -> CausalExplanation:
        """分析"如果...会怎样"查询"""
        intervention_node = query.target_event
        conditions = query.conditions
        
        # 获取所有受影响的节点
        affected_nodes = self._get_affected_nodes(intervention_node, conditions)
        
        # 计算影响程度
        effects = []
        for node in affected_nodes:
            paths = self.causal_graph.get_causal_paths(intervention_node, node)
            if paths:
                max_strength = max(self._calculate_path_strength(path) for path in paths)
                effects.append({
                    "node": node,
                    "effect_strength": max_strength,
                    "paths": paths[:3]  # 保留前3条路径
                })
                
        # 生成解释
        reasoning_steps = []
        causal_chain = [intervention_node]
        
        if effects:
            # 按影响强度排序
            effects.sort(key=lambda x: x["effect_strength"], reverse=True)
            
            for effect in effects[:5]:  # 前5个最强影响
                causal_chain.append(effect["node"])
                reasoning_steps.append(
                    f"If {intervention_node} occurs, {effect['node']} "
                    f"will be affected with strength {effect['effect_strength']:.2f}"
                )
                
        confidence = np.mean([e["effect_strength"] for e in effects]) if effects else 0.1
        evidence = self._collect_evidence(causal_chain)
        
        return CausalExplanation(
            query_id=query.query_id,
            explanation_type="what_if",
            causal_chain=causal_chain,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            evidence=evidence
        )
        
    async def _analyze_how_query(self, query: CausalQuery) -> CausalExplanation:
        """分析"如何"查询"""
        target_event = query.target_event
        conditions = query.conditions
        
        # 找到实现目标的可能路径
        possible_interventions = self._find_interventions(target_event, conditions)
        
        # 评估每个干预的可行性
        intervention_plans = []
        for intervention in possible_interventions:
            feasibility = self._evaluate_intervention_feasibility(intervention)
            intervention_plans.append({
                "intervention": intervention,
                "feasibility": feasibility,
                "steps": self._generate_intervention_steps(intervention)
            })
            
        # 选择最可行的方案
        if intervention_plans:
            best_plan = max(intervention_plans, key=lambda x: x["feasibility"])
            causal_chain = best_plan["intervention"]["path"]
            confidence = best_plan["feasibility"]
            reasoning_steps = best_plan["steps"]
        else:
            causal_chain = [target_event]
            confidence = 0.1
            reasoning_steps = ["No feasible intervention found"]
            
        evidence = self._collect_evidence(causal_chain)
        
        return CausalExplanation(
            query_id=query.query_id,
            explanation_type="how",
            causal_chain=causal_chain,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            evidence=evidence,
            alternative_explanations=[
                {"intervention": plan["intervention"], "feasibility": plan["feasibility"]}
                for plan in intervention_plans[1:4]  # 前3个替代方案
            ]
        )
        
    async def _analyze_counterfactual_query(self, query: CausalQuery) -> CausalExplanation:
        """分析反事实查询"""
        target_event = query.target_event
        counterfactual_conditions = query.conditions
        
        # 构建反事实场景
        counterfactual_graph = self._create_counterfactual_graph(counterfactual_conditions)
        
        # 分析在反事实条件下的结果
        counterfactual_outcomes = self._analyze_counterfactual_outcomes(
            counterfactual_graph, target_event
        )
        
        # 比较实际结果和反事实结果
        comparison = self._compare_factual_counterfactual(
            target_event, counterfactual_outcomes
        )
        
        causal_chain = comparison.get("causal_chain", [target_event])
        confidence = comparison.get("confidence", 0.5)
        reasoning_steps = comparison.get("reasoning_steps", [])
        evidence = self._collect_evidence(causal_chain)
        
        return CausalExplanation(
            query_id=query.query_id,
            explanation_type="counterfactual",
            causal_chain=causal_chain,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            evidence=evidence
        )
        
    def _calculate_path_strength(self, path: List[str]) -> float:
        """计算路径强度"""
        if len(path) < 2:
            return 0.0
            
        strength = 1.0
        for i in range(len(path) - 1):
            edge_key = (path[i], path[i + 1])
            if edge_key in self.causal_graph.edges:
                edge_strength = self.causal_graph.edges[edge_key].strength
                strength *= edge_strength
            else:
                return 0.0
                
        return strength
        
    def _generate_path_explanation(self, path: List[str]) -> str:
        """生成路径解释"""
        if len(path) < 2:
            return f"Event: {path[0]}"
            
        explanations = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            edge_key = (source, target)
            
            if edge_key in self.causal_graph.edges:
                edge = self.causal_graph.edges[edge_key]
                relation_desc = edge.relation_type.value.replace("_", " ")
                explanations.append(f"{source} {relation_desc} {target}")
            else:
                explanations.append(f"{source} leads to {target}")
                
        return " → ".join(explanations)
        
    def _get_affected_nodes(self, intervention_node: str, conditions: Dict[str, Any]) -> List[str]:
        """获取受干预影响的节点"""
        affected = set()
        
        # 使用BFS找到所有下游节点
        queue = [intervention_node]
        visited = set([intervention_node])
        
        while queue:
            current = queue.pop(0)
            successors = self.causal_graph.get_direct_effects(current)
            
            for successor in successors:
                if successor not in visited:
                    # 检查条件是否满足
                    edge_key = (current, successor)
                    if edge_key in self.causal_graph.edges:
                        edge = self.causal_graph.edges[edge_key]
                        if self._check_conditions(edge.conditions, conditions):
                            affected.add(successor)
                            queue.append(successor)
                            visited.add(successor)
                            
        return list(affected)
        
    def _check_conditions(self, edge_conditions: List[str], query_conditions: Dict[str, Any]) -> bool:
        """检查条件是否满足"""
        if not edge_conditions:
            return True
            
        for condition in edge_conditions:
            # 简化的条件检查逻辑
            if condition not in query_conditions:
                return False
                
        return True
        
    def _find_interventions(self, target_event: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """找到实现目标的干预方案"""
        interventions = []
        
        # 获取所有可能的根本原因
        root_causes = self.causal_graph.get_root_causes(target_event)
        
        for root_cause in root_causes:
            paths = self.causal_graph.get_causal_paths(root_cause, target_event)
            for path in paths:
                strength = self._calculate_path_strength(path)
                if strength > 0.3:  # 只考虑强度足够的路径
                    interventions.append({
                        "path": path,
                        "strength": strength,
                        "intervention_point": root_cause
                    })
                    
        return interventions
        
    def _evaluate_intervention_feasibility(self, intervention: Dict[str, Any]) -> float:
        """评估干预的可行性"""
        # 简化的可行性评估
        base_feasibility = intervention["strength"]
        
        # 考虑路径长度（越短越可行）
        path_length = len(intervention["path"])
        length_penalty = 0.1 * (path_length - 1)
        
        # 考虑干预点的可控性
        intervention_point = intervention["intervention_point"]
        controllability = self._get_node_controllability(intervention_point)
        
        feasibility = base_feasibility * controllability - length_penalty
        return max(0.0, min(1.0, feasibility))
        
    def _get_node_controllability(self, node_id: str) -> float:
        """获取节点的可控性"""
        if node_id in self.causal_graph.nodes:
            node = self.causal_graph.nodes[node_id]
            return node.properties.get("controllability", 0.5)
        return 0.5
        
    def _generate_intervention_steps(self, intervention: Dict[str, Any]) -> List[str]:
        """生成干预步骤"""
        path = intervention["path"]
        steps = []
        
        steps.append(f"1. Identify intervention point: {path[0]}")
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            steps.append(f"{i + 2}. Ensure {source} leads to {target}")
            
        steps.append(f"{len(path) + 1}. Monitor outcome: {path[-1]}")
        
        return steps
        
    def _create_counterfactual_graph(self, counterfactual_conditions: Dict[str, Any]) -> CausalGraph:
        """创建反事实图"""
        # 复制原图
        counterfactual_graph = CausalGraph()
        
        # 复制节点
        for node_id, node in self.causal_graph.nodes.items():
            counterfactual_graph.add_node(node)
            
        # 复制边，但根据反事实条件修改
        for edge_key, edge in self.causal_graph.edges.items():
            modified_edge = self._modify_edge_for_counterfactual(edge, counterfactual_conditions)
            if modified_edge:
                counterfactual_graph.add_edge(modified_edge)
                
        return counterfactual_graph
        
    def _modify_edge_for_counterfactual(self, edge: CausalEdge, conditions: Dict[str, Any]) -> Optional[CausalEdge]:
        """为反事实场景修改边"""
        # 检查是否需要修改或删除这条边
        for condition_key, condition_value in conditions.items():
            if condition_key == "remove_edge" and condition_value == f"{edge.source}->{edge.target}":
                return None  # 删除这条边
            elif condition_key == "weaken_edge" and condition_value == f"{edge.source}->{edge.target}":
                # 减弱边的强度
                modified_edge = CausalEdge(
                    source=edge.source,
                    target=edge.target,
                    relation_type=edge.relation_type,
                    strength=edge.strength * 0.5,
                    confidence=edge.confidence,
                    conditions=edge.conditions,
                    temporal_delay=edge.temporal_delay,
                    evidence=edge.evidence,
                    metadata=edge.metadata
                )
                return modified_edge
                
        return edge  # 保持原样
        
    def _analyze_counterfactual_outcomes(self, counterfactual_graph: CausalGraph, target_event: str) -> Dict[str, Any]:
        """分析反事实结果"""
        # 在反事实图中分析目标事件的可能性
        root_causes = counterfactual_graph.get_root_causes(target_event)
        
        outcomes = {
            "target_event": target_event,
            "root_causes": root_causes,
            "probability": 0.0,
            "causal_paths": []
        }
        
        total_strength = 0.0
        path_count = 0
        
        for root_cause in root_causes:
            paths = counterfactual_graph.get_causal_paths(root_cause, target_event)
            for path in paths:
                strength = self._calculate_path_strength_in_graph(path, counterfactual_graph)
                outcomes["causal_paths"].append({
                    "path": path,
                    "strength": strength
                })
                total_strength += strength
                path_count += 1
                
        if path_count > 0:
            outcomes["probability"] = total_strength / path_count
            
        return outcomes
        
    def _calculate_path_strength_in_graph(self, path: List[str], graph: CausalGraph) -> float:
        """在指定图中计算路径强度"""
        if len(path) < 2:
            return 0.0
            
        strength = 1.0
        for i in range(len(path) - 1):
            edge_key = (path[i], path[i + 1])
            if edge_key in graph.edges:
                edge_strength = graph.edges[edge_key].strength
                strength *= edge_strength
            else:
                return 0.0
                
        return strength
        
    def _compare_factual_counterfactual(self, target_event: str, counterfactual_outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """比较事实和反事实结果"""
        # 获取实际场景的结果
        factual_outcomes = self._analyze_counterfactual_outcomes(self.causal_graph, target_event)
        
        # 比较概率
        factual_prob = factual_outcomes["probability"]
        counterfactual_prob = counterfactual_outcomes["probability"]
        
        difference = abs(factual_prob - counterfactual_prob)
        
        reasoning_steps = [
            f"In the actual scenario, {target_event} has probability {factual_prob:.2f}",
            f"In the counterfactual scenario, {target_event} has probability {counterfactual_prob:.2f}",
            f"The difference is {difference:.2f}"
        ]
        
        if difference > 0.3:
            reasoning_steps.append("This represents a significant causal effect")
            confidence = 0.8
        elif difference > 0.1:
            reasoning_steps.append("This represents a moderate causal effect")
            confidence = 0.6
        else:
            reasoning_steps.append("This represents a weak causal effect")
            confidence = 0.3
            
        # 构建因果链
        causal_chain = [target_event]
        if counterfactual_outcomes["causal_paths"]:
            best_path = max(counterfactual_outcomes["causal_paths"], key=lambda x: x["strength"])
            causal_chain = best_path["path"]
            
        return {
            "causal_chain": causal_chain,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
            "factual_probability": factual_prob,
            "counterfactual_probability": counterfactual_prob,
            "causal_effect": difference
        }
        
    def _collect_evidence(self, causal_chain: List[str]) -> List[str]:
        """收集证据"""
        evidence = []
        
        for i in range(len(causal_chain) - 1):
            edge_key = (causal_chain[i], causal_chain[i + 1])
            if edge_key in self.causal_graph.edges:
                edge_evidence = self.causal_graph.edges[edge_key].evidence
                evidence.extend(edge_evidence)
                
        # 去重
        return list(set(evidence))
        
    async def explain_causal_mechanism(self, source: str, target: str) -> Dict[str, Any]:
        """解释因果机制"""
        paths = self.causal_graph.get_causal_paths(source, target)
        
        if not paths:
            return {
                "source": source,
                "target": target,
                "mechanism": "No causal relationship found",
                "confidence": 0.0
            }
            
        # 选择最强的路径
        best_path = max(paths, key=self._calculate_path_strength)
        strength = self._calculate_path_strength(best_path)
        
        # 生成机制解释
        mechanism_steps = []
        for i in range(len(best_path) - 1):
            source_node = best_path[i]
            target_node = best_path[i + 1]
            edge_key = (source_node, target_node)
            
            if edge_key in self.causal_graph.edges:
                edge = self.causal_graph.edges[edge_key]
                mechanism_steps.append({
                    "step": i + 1,
                    "from": source_node,
                    "to": target_node,
                    "relation": edge.relation_type.value,
                    "strength": edge.strength,
                    "conditions": edge.conditions,
                    "temporal_delay": edge.temporal_delay
                })
                
        return {
            "source": source,
            "target": target,
            "causal_path": best_path,
            "mechanism": mechanism_steps,
            "overall_strength": strength,
            "confidence": strength,
            "alternative_paths": [
                {"path": path, "strength": self._calculate_path_strength(path)}
                for path in paths[1:4]  # 前3个替代路径
            ]
        }


# 示例使用
async def main():
    """示例：因果推理"""
    
    # 创建因果推理引擎
    reasoning_engine = CausalReasoningEngine()
    
    # 构建示例领域知识
    domain_knowledge = {
        "entities": [
            {"id": "rain", "name": "Rain", "description": "Weather condition"},
            {"id": "wet_ground", "name": "Wet Ground", "description": "Ground condition"},
            {"id": "slippery_road", "name": "Slippery Road", "description": "Road condition"},
            {"id": "car_accident", "name": "Car Accident", "description": "Traffic incident"},
            {"id": "traffic_jam", "name": "Traffic Jam", "description": "Traffic condition"}
        ],
        "events": [
            {"id": "heavy_rain", "name": "Heavy Rain", "description": "Intense rainfall"},
            {"id": "driver_speeding", "name": "Driver Speeding", "description": "Excessive speed"}
        ],
        "relationships": [
            {
                "type": "causal",
                "source": "rain",
                "target": "wet_ground",
                "relation_type": "direct_cause",
                "strength": 0.9,
                "confidence": 0.95
            },
            {
                "type": "causal",
                "source": "wet_ground",
                "target": "slippery_road",
                "relation_type": "direct_cause",
                "strength": 0.8,
                "confidence": 0.9
            },
            {
                "type": "causal",
                "source": "slippery_road",
                "target": "car_accident",
                "relation_type": "contributory_cause",
                "strength": 0.6,
                "confidence": 0.8
            },
            {
                "type": "causal",
                "source": "driver_speeding",
                "target": "car_accident",
                "relation_type": "contributory_cause",
                "strength": 0.7,
                "confidence": 0.85
            },
            {
                "type": "causal",
                "source": "car_accident",
                "target": "traffic_jam",
                "relation_type": "direct_cause",
                "strength": 0.8,
                "confidence": 0.9
            }
        ]
    }
    
    # 构建因果模型
    causal_graph = await reasoning_engine.build_causal_model(domain_knowledge)
    print(f"Built causal graph with {len(causal_graph.nodes)} nodes and {len(causal_graph.edges)} edges")
    
    # 示例查询1：为什么发生交通事故？
    why_query = CausalQuery(
        query_id="why_001",
        query_type="why",
        target_event="car_accident"
    )
    
    why_explanation = await reasoning_engine.analyze_causal_query(why_query)
    print(f"\nWhy Query Result:")
    print(f"Causal Chain: {' -> '.join(why_explanation.causal_chain)}")
    print(f"Confidence: {why_explanation.confidence:.2f}")
    print(f"Reasoning: {why_explanation.reasoning_steps[0]}")
    
    # 示例查询2：如果下雨会怎样？
    what_if_query = CausalQuery(
        query_id="what_if_001",
        query_type="what_if",
        target_event="rain",
        conditions={"intensity": "heavy"}
    )
    
    what_if_explanation = await reasoning_engine.analyze_causal_query(what_if_query)
    print(f"\nWhat-if Query Result:")
    print(f"Causal Chain: {' -> '.join(what_if_explanation.causal_chain)}")
    print(f"Confidence: {what_if_explanation.confidence:.2f}")
    print(f"Effects: {what_if_explanation.reasoning_steps}")
    
    # 示例查询3：如何减少交通事故？
    how_query = CausalQuery(
        query_id="how_001",
        query_type="how",
        target_event="car_accident",
        conditions={"goal": "reduce"}
    )
    
    how_explanation = await reasoning_engine.analyze_causal_query(how_query)
    print(f"\nHow Query Result:")
    print(f"Intervention Plan: {' -> '.join(how_explanation.causal_chain)}")
    print(f"Confidence: {how_explanation.confidence:.2f}")
    print(f"Steps: {how_explanation.reasoning_steps}")
    
    # 解释因果机制
    mechanism = await reasoning_engine.explain_causal_mechanism("rain", "traffic_jam")
    print(f"\nCausal Mechanism (rain -> traffic_jam):")
    print(f"Path: {' -> '.join(mechanism['causal_path'])}")
    print(f"Overall Strength: {mechanism['overall_strength']:.2f}")
    print(f"Mechanism Steps:")
    for step in mechanism['mechanism']:
        print(f"  {step['step']}. {step['from']} --({step['relation']})-> {step['to']} (strength: {step['strength']:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
