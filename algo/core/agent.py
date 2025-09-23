"""
增强版Agent系统 - v1.3.0
支持多步推理、自主规划、工具组合调用
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint import MemorySaver
from loguru import logger

from core.llm import LLMService
from core.mcp import MCPToolRegistry
from core.retrieve import RAGService


class AgentCapability(Enum):
    """Agent能力枚举"""
    REASONING = "reasoning"          # 推理
    PLANNING = "planning"            # 规划
    TOOL_USE = "tool_use"           # 工具使用
    MEMORY = "memory"               # 记忆
    LEARNING = "learning"           # 学习
    COLLABORATION = "collaboration" # 协作


@dataclass
class AgentState:
    """Agent状态"""
    conversation_id: str
    messages: List[BaseMessage] = field(default_factory=list)
    current_task: Optional[str] = None
    plan: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_iterations: int = 10
    current_iteration: int = 0
    confidence_threshold: float = 0.7
    error_count: int = 0
    
    # 新增字段
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    sub_agents: List['EnhancedAgent'] = field(default_factory=list)
    learning_data: List[Dict[str, Any]] = field(default_factory=list)
    collaboration_requests: List[Dict[str, Any]] = field(default_factory=list)


class ReasoningEngine:
    """推理引擎"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.reasoning_templates = {
            "deductive": """
            基于以下前提进行演绎推理：
            前提：{premises}
            请推导出逻辑结论。
            """,
            "inductive": """
            基于以下观察进行归纳推理：
            观察：{observations}
            请总结出一般规律。
            """,
            "abductive": """
            基于以下现象进行溯因推理：
            现象：{phenomena}
            已知规则：{rules}
            请推测最可能的原因。
            """,
            "analogical": """
            基于以下类比进行推理：
            源领域：{source}
            目标领域：{target}
            请找出相似性并推理。
            """
        }
    
    async def reason(
        self,
        problem: str,
        reasoning_type: str = "deductive",
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """执行推理"""
        try:
            # 选择推理模板
            template = self.reasoning_templates.get(reasoning_type, self.reasoning_templates["deductive"])
            
            # 构建推理提示
            prompt = f"""
            问题：{problem}
            
            {template.format(**context or {})}
            
            请按以下格式输出：
            1. 推理步骤：
            2. 中间结论：
            3. 最终结论：
            4. 置信度（0-1）：
            5. 证据支持：
            """
            
            # 调用LLM进行推理
            response = await self.llm.generate(prompt)
            
            # 解析推理结果
            reasoning_result = self._parse_reasoning_result(response)
            
            return {
                "type": reasoning_type,
                "problem": problem,
                "steps": reasoning_result.get("steps", []),
                "conclusion": reasoning_result.get("conclusion", ""),
                "confidence": reasoning_result.get("confidence", 0.5),
                "evidence": reasoning_result.get("evidence", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return {
                "type": reasoning_type,
                "problem": problem,
                "error": str(e),
                "confidence": 0.0
            }
    
    def _parse_reasoning_result(self, response: str) -> Dict[str, Any]:
        """解析推理结果"""
        # 简化实现，实际应该用更复杂的解析逻辑
        lines = response.split("\n")
        result = {
            "steps": [],
            "conclusion": "",
            "confidence": 0.5,
            "evidence": []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if "推理步骤" in line:
                current_section = "steps"
            elif "最终结论" in line:
                current_section = "conclusion"
            elif "置信度" in line:
                try:
                    confidence = float(line.split("：")[-1].strip())
                    result["confidence"] = min(max(confidence, 0), 1)
                except:
                    pass
            elif "证据支持" in line:
                current_section = "evidence"
            elif line and current_section:
                if current_section == "steps":
                    result["steps"].append(line)
                elif current_section == "conclusion":
                    result["conclusion"] += line + " "
                elif current_section == "evidence":
                    result["evidence"].append(line)
        
        return result


class PlanningEngine:
    """规划引擎"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.planning_strategies = {
            "hierarchical": self._hierarchical_planning,
            "reactive": self._reactive_planning,
            "deliberative": self._deliberative_planning,
            "hybrid": self._hybrid_planning
        }
    
    async def create_plan(
        self,
        goal: str,
        constraints: List[str] = None,
        resources: List[str] = None,
        strategy: str = "hierarchical"
    ) -> List[Dict[str, Any]]:
        """创建执行计划"""
        planning_func = self.planning_strategies.get(strategy, self._hierarchical_planning)
        return await planning_func(goal, constraints, resources)
    
    async def _hierarchical_planning(
        self,
        goal: str,
        constraints: List[str] = None,
        resources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """层次化规划"""
        prompt = f"""
        目标：{goal}
        约束条件：{constraints or '无'}
        可用资源：{resources or '无'}
        
        请创建一个层次化的执行计划：
        1. 将目标分解为子目标
        2. 为每个子目标制定具体步骤
        3. 确定步骤之间的依赖关系
        4. 估计每个步骤的时间和资源需求
        
        输出JSON格式的计划。
        """
        
        response = await self.llm.generate(prompt)
        
        try:
            # 解析JSON响应
            plan_data = json.loads(response)
            
            # 转换为标准计划格式
            plan = []
            for item in plan_data.get("steps", []):
                plan.append({
                    "step_id": item.get("id", len(plan) + 1),
                    "action": item.get("action", ""),
                    "description": item.get("description", ""),
                    "dependencies": item.get("dependencies", []),
                    "estimated_time": item.get("time", "unknown"),
                    "required_resources": item.get("resources", []),
                    "priority": item.get("priority", "medium"),
                    "status": "pending"
                })
            
            return plan
            
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回简单计划
            return [{
                "step_id": 1,
                "action": "execute",
                "description": goal,
                "dependencies": [],
                "status": "pending"
            }]
    
    async def _reactive_planning(
        self,
        goal: str,
        constraints: List[str] = None,
        resources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """反应式规划（根据当前状态动态调整）"""
        # 简化实现
        return [{
            "step_id": 1,
            "action": "assess_situation",
            "description": "评估当前状态"
        }, {
            "step_id": 2,
            "action": "react",
            "description": f"根据状态执行: {goal}"
        }]
    
    async def _deliberative_planning(
        self,
        goal: str,
        constraints: List[str] = None,
        resources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """深思熟虑式规划（考虑长期影响）"""
        # 简化实现
        return [{
            "step_id": 1,
            "action": "analyze_long_term",
            "description": "分析长期影响"
        }, {
            "step_id": 2,
            "action": "optimize_path",
            "description": "优化执行路径"
        }, {
            "step_id": 3,
            "action": "execute",
            "description": goal
        }]
    
    async def _hybrid_planning(
        self,
        goal: str,
        constraints: List[str] = None,
        resources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """混合规划（结合多种策略）"""
        # 结合层次化和反应式
        hierarchical = await self._hierarchical_planning(goal, constraints, resources)
        reactive = await self._reactive_planning(goal, constraints, resources)
        
        # 合并计划
        return hierarchical[:2] + reactive[-1:]


class MemorySystem:
    """记忆系统"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: Dict[str, Any] = {}
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
    
    def store(self, memory_type: str, key: str, value: Any, metadata: Dict[str, Any] = None):
        """存储记忆"""
        memory_item = {
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "access_count": 0
        }
        
        if memory_type == "short_term":
            self.short_term_memory.append(memory_item)
            # 限制大小
            if len(self.short_term_memory) > 100:
                self.short_term_memory.pop(0)
                
        elif memory_type == "long_term":
            self.long_term_memory[key] = memory_item
            
        elif memory_type == "episodic":
            self.episodic_memory.append(memory_item)
            if len(self.episodic_memory) > self.max_size:
                self.episodic_memory.pop(0)
                
        elif memory_type == "semantic":
            self.semantic_memory[key] = memory_item
            
        elif memory_type == "working":
            self.working_memory[key] = memory_item
    
    def retrieve(self, memory_type: str, key: str = None, filter_func: Callable = None) -> Any:
        """检索记忆"""
        if memory_type == "short_term":
            if filter_func:
                return [m for m in self.short_term_memory if filter_func(m)]
            return self.short_term_memory
            
        elif memory_type == "long_term":
            if key:
                memory = self.long_term_memory.get(key)
                if memory:
                    memory["access_count"] += 1
                return memory
            return self.long_term_memory
            
        elif memory_type == "episodic":
            if filter_func:
                return [m for m in self.episodic_memory if filter_func(m)]
            return self.episodic_memory
            
        elif memory_type == "semantic":
            if key:
                return self.semantic_memory.get(key)
            return self.semantic_memory
            
        elif memory_type == "working":
            if key:
                return self.working_memory.get(key)
            return self.working_memory
        
        return None
    
    def consolidate(self):
        """记忆巩固（将短期记忆转为长期记忆）"""
        # 基于访问频率和重要性
        important_memories = [
            m for m in self.short_term_memory
            if m.get("metadata", {}).get("importance", 0) > 0.7
        ]
        
        for memory in important_memories:
            self.long_term_memory[memory["key"]] = memory
            
        # 清理短期记忆
        self.short_term_memory = [
            m for m in self.short_term_memory
            if m not in important_memories
        ]
    
    def forget(self, decay_rate: float = 0.1):
        """遗忘机制"""
        # 基于时间和访问频率的遗忘
        now = datetime.now()
        
        # 清理长期记忆中很久未访问的项
        keys_to_remove = []
        for key, memory in self.long_term_memory.items():
            last_access = datetime.fromisoformat(memory["timestamp"])
            days_passed = (now - last_access).days
            
            # 遗忘概率
            forget_prob = decay_rate * days_passed / (memory["access_count"] + 1)
            
            if forget_prob > 0.9:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.long_term_memory[key]


class LearningEngine:
    """学习引擎"""
    
    def __init__(self):
        self.learned_patterns: List[Dict[str, Any]] = []
        self.feedback_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
    
    async def learn_from_experience(
        self,
        experience: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """从经验中学习"""
        # 提取模式
        pattern = self._extract_pattern(experience)
        
        # 如果有反馈，更新学习
        if feedback:
            self.feedback_history.append({
                "experience": experience,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            
            # 调整模式权重
            self._adjust_pattern_weights(pattern, feedback)
        
        # 存储学习到的模式
        self.learned_patterns.append(pattern)
        
        # 更新性能指标
        self._update_performance_metrics(experience, feedback)
        
        return {
            "pattern": pattern,
            "confidence": pattern.get("confidence", 0.5),
            "improvements": self._suggest_improvements(pattern)
        }
    
    def _extract_pattern(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """提取模式"""
        return {
            "context": experience.get("context", {}),
            "action": experience.get("action", ""),
            "outcome": experience.get("outcome", ""),
            "confidence": 0.5,
            "frequency": 1
        }
    
    def _adjust_pattern_weights(self, pattern: Dict[str, Any], feedback: Dict[str, Any]):
        """调整模式权重"""
        if feedback.get("success", False):
            pattern["confidence"] = min(pattern["confidence"] * 1.1, 1.0)
        else:
            pattern["confidence"] = max(pattern["confidence"] * 0.9, 0.1)
    
    def _update_performance_metrics(self, experience: Dict[str, Any], feedback: Optional[Dict[str, Any]]):
        """更新性能指标"""
        metric_name = experience.get("metric_name", "general")
        
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = 0.5
        
        if feedback:
            success_rate = feedback.get("success_rate", 0.5)
            # 指数移动平均
            alpha = 0.1
            self.performance_metrics[metric_name] = (
                alpha * success_rate + 
                (1 - alpha) * self.performance_metrics[metric_name]
            )
    
    def _suggest_improvements(self, pattern: Dict[str, Any]) -> List[str]:
        """建议改进"""
        suggestions = []
        
        if pattern["confidence"] < 0.3:
            suggestions.append("需要更多训练数据")
        
        if pattern.get("frequency", 0) < 5:
            suggestions.append("模式出现频率较低，需要验证")
        
        return suggestions


class EnhancedAgent:
    """增强版Agent"""
    
    def __init__(
        self,
        agent_id: str,
        llm_service: LLMService,
        rag_service: RAGService,
        tool_registry: MCPToolRegistry,
        capabilities: List[AgentCapability] = None
    ):
        self.agent_id = agent_id
        self.llm = llm_service
        self.rag = rag_service
        self.tools = tool_registry
        
        # 初始化能力
        self.capabilities = capabilities or [
            AgentCapability.REASONING,
            AgentCapability.PLANNING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY
        ]
        
        # 初始化组件
        self.reasoning_engine = ReasoningEngine(llm_service)
        self.planning_engine = PlanningEngine(llm_service)
        self.memory_system = MemorySystem()
        self.learning_engine = LearningEngine()
        
        # 创建状态图
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
        
        logger.info(f"增强Agent初始化完成: {agent_id}, 能力: {[c.value for c in self.capabilities]}")
    
    def _build_graph(self) -> StateGraph:
        """构建Agent状态图"""
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("reason", self._reason_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("learn", self._learn_node)
        graph.add_node("collaborate", self._collaborate_node)
        graph.add_node("synthesize", self._synthesize_node)
        
        # 添加边
        graph.add_edge("analyze", "reason")
        graph.add_conditional_edges(
            "reason",
            self._should_plan,
            {
                True: "plan",
                False: "retrieve"
            }
        )
        graph.add_edge("plan", "execute")
        graph.add_edge("retrieve", "execute")
        graph.add_conditional_edges(
            "execute",
            self._should_collaborate,
            {
                True: "collaborate",
                False: "learn"
            }
        )
        graph.add_edge("collaborate", "learn")
        graph.add_edge("learn", "synthesize")
        graph.add_conditional_edges(
            "synthesize",
            self._should_continue,
            {
                True: "analyze",
                False: END
            }
        )
        
        # 设置入口
        graph.set_entry_point("analyze")
        
        return graph.compile()
    
    async def _analyze_node(self, state: AgentState) -> AgentState:
        """分析节点"""
        logger.info(f"[{self.agent_id}] 分析任务")
        
        # 分析当前任务
        last_message = state.messages[-1] if state.messages else None
        if last_message:
            state.current_task = last_message.content
            
            # 存储到工作记忆
            self.memory_system.store(
                "working",
                "current_task",
                state.current_task,
                {"timestamp": datetime.now().isoformat()}
            )
        
        return state
    
    async def _reason_node(self, state: AgentState) -> AgentState:
        """推理节点"""
        if AgentCapability.REASONING not in self.capabilities:
            return state
        
        logger.info(f"[{self.agent_id}] 执行推理")
        
        # 执行推理
        reasoning_result = await self.reasoning_engine.reason(
            problem=state.current_task,
            reasoning_type="deductive",
            context=state.context
        )
        
        # 添加到推理链
        state.reasoning_chain.append(reasoning_result)
        
        # 存储到记忆
        self.memory_system.store(
            "semantic",
            f"reasoning_{len(state.reasoning_chain)}",
            reasoning_result
        )
        
        return state
    
    async def _plan_node(self, state: AgentState) -> AgentState:
        """规划节点"""
        if AgentCapability.PLANNING not in self.capabilities:
            return state
        
        logger.info(f"[{self.agent_id}] 创建执行计划")
        
        # 创建计划
        plan = await self.planning_engine.create_plan(
            goal=state.current_task,
            constraints=state.context.get("constraints", []),
            resources=list(self.tools.list_tools().keys())
        )
        
        state.plan = plan
        
        # 存储到记忆
        self.memory_system.store(
            "episodic",
            f"plan_{state.conversation_id}",
            plan
        )
        
        return state
    
    async def _retrieve_node(self, state: AgentState) -> AgentState:
        """检索节点"""
        logger.info(f"[{self.agent_id}] 执行RAG检索")
        
        # RAG检索
        retrieval_result = await self.rag.retrieve(
            query=state.current_task,
            top_k=5
        )
        
        state.context["retrieval_results"] = retrieval_result
        
        return state
    
    async def _execute_node(self, state: AgentState) -> AgentState:
        """执行节点"""
        if AgentCapability.TOOL_USE not in self.capabilities:
            return state
        
        logger.info(f"[{self.agent_id}] 执行工具调用")
        
        # 根据计划执行工具
        for step in state.plan:
            if step["status"] == "pending":
                # 选择合适的工具
                tool_name = self._select_tool(step["action"])
                
                if tool_name:
                    # 执行工具
                    tool_result = await self.tools.execute_tool(
                        tool_name,
                        step.get("parameters", {})
                    )
                    
                    # 记录结果
                    state.tool_calls.append({
                        "tool": tool_name,
                        "parameters": step.get("parameters", {}),
                        "result": tool_result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    step["status"] = "completed"
                    step["result"] = tool_result
        
        return state
    
    async def _learn_node(self, state: AgentState) -> AgentState:
        """学习节点"""
        if AgentCapability.LEARNING not in self.capabilities:
            return state
        
        logger.info(f"[{self.agent_id}] 从经验中学习")
        
        # 构建经验
        experience = {
            "task": state.current_task,
            "plan": state.plan,
            "tool_calls": state.tool_calls,
            "context": state.context,
            "outcome": "success" if not state.error_count else "failure"
        }
        
        # 学习
        learning_result = await self.learning_engine.learn_from_experience(experience)
        
        state.learning_data.append(learning_result)
        
        # 记忆巩固
        self.memory_system.consolidate()
        
        return state
    
    async def _collaborate_node(self, state: AgentState) -> AgentState:
        """协作节点"""
        if AgentCapability.COLLABORATION not in self.capabilities:
            return state
        
        logger.info(f"[{self.agent_id}] 请求协作")
        
        # 判断是否需要其他Agent协助
        if self._needs_collaboration(state):
            # 创建协作请求
            collaboration_request = {
                "from_agent": self.agent_id,
                "task": state.current_task,
                "required_capabilities": self._identify_required_capabilities(state),
                "context": state.context,
                "timestamp": datetime.now().isoformat()
            }
            
            state.collaboration_requests.append(collaboration_request)
            
            # TODO: 实际的协作逻辑（调用其他Agent）
        
        return state
    
    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """综合节点"""
        logger.info(f"[{self.agent_id}] 综合结果")
        
        # 综合所有结果
        synthesis = {
            "task": state.current_task,
            "reasoning": state.reasoning_chain,
            "plan_execution": state.plan,
            "tool_results": state.tool_calls,
            "learning": state.learning_data,
            "collaborations": state.collaboration_requests
        }
        
        # 生成最终响应
        response = await self._generate_response(synthesis)
        
        # 添加到消息
        state.messages.append(AIMessage(content=response))
        
        # 更新迭代计数
        state.current_iteration += 1
        
        return state
    
    def _should_plan(self, state: AgentState) -> bool:
        """判断是否需要规划"""
        # 如果任务复杂或需要多步骤
        if state.reasoning_chain:
            last_reasoning = state.reasoning_chain[-1]
            if last_reasoning.get("confidence", 0) < state.confidence_threshold:
                return True
        
        # 如果有明确的步骤指示词
        task_lower = state.current_task.lower() if state.current_task else ""
        planning_keywords = ["步骤", "计划", "流程", "先", "然后", "最后"]
        
        return any(keyword in task_lower for keyword in planning_keywords)
    
    def _should_collaborate(self, state: AgentState) -> bool:
        """判断是否需要协作"""
        # 如果错误次数过多
        if state.error_count > 2:
            return True
        
        # 如果置信度过低
        if state.reasoning_chain:
            avg_confidence = sum(
                r.get("confidence", 0) for r in state.reasoning_chain
            ) / len(state.reasoning_chain)
            
            if avg_confidence < 0.5:
                return True
        
        return False
    
    def _should_continue(self, state: AgentState) -> bool:
        """判断是否继续"""
        # 达到最大迭代次数
        if state.current_iteration >= state.max_iterations:
            return False
        
        # 任务已完成
        if state.plan and all(step["status"] == "completed" for step in state.plan):
            return False
        
        # 有新的子任务
        if state.collaboration_requests:
            return True
        
        return False
    
    def _select_tool(self, action: str) -> Optional[str]:
        """选择合适的工具"""
        # 简单的关键词匹配
        tool_mapping = {
            "search": "web_search",
            "calculate": "calculator",
            "file": "filesystem",
            "database": "database",
            "api": "http"
        }
        
        for keyword, tool_name in tool_mapping.items():
            if keyword in action.lower():
                if tool_name in self.tools.list_tools():
                    return tool_name
        
        return None
    
    def _needs_collaboration(self, state: AgentState) -> bool:
        """判断是否需要协作"""
        # 基于任务复杂度和当前能力
        required_capabilities = self._identify_required_capabilities(state)
        missing_capabilities = [
            cap for cap in required_capabilities
            if cap not in self.capabilities
        ]
        
        return len(missing_capabilities) > 0
    
    def _identify_required_capabilities(self, state: AgentState) -> List[AgentCapability]:
        """识别所需能力"""
        required = []
        
        task_lower = state.current_task.lower() if state.current_task else ""
        
        if "推理" in task_lower or "分析" in task_lower:
            required.append(AgentCapability.REASONING)
        
        if "计划" in task_lower or "步骤" in task_lower:
            required.append(AgentCapability.PLANNING)
        
        if "工具" in task_lower or "执行" in task_lower:
            required.append(AgentCapability.TOOL_USE)
        
        if "记住" in task_lower or "记忆" in task_lower:
            required.append(AgentCapability.MEMORY)
        
        if "学习" in task_lower or "改进" in task_lower:
            required.append(AgentCapability.LEARNING)
        
        return required
    
    async def _generate_response(self, synthesis: Dict[str, Any]) -> str:
        """生成最终响应"""
        # 构建响应提示
        prompt = f"""
        基于以下执行结果生成响应：
        
        任务：{synthesis['task']}
        
        推理结果：{json.dumps(synthesis['reasoning'], ensure_ascii=False, indent=2)}
        
        执行计划：{json.dumps(synthesis['plan_execution'], ensure_ascii=False, indent=2)}
        
        工具结果：{json.dumps(synthesis['tool_results'], ensure_ascii=False, indent=2)}
        
        请生成一个清晰、准确、有帮助的响应。
        """
        
        response = await self.llm.generate(prompt)
        
        return response
    
    async def run(
        self,
        message: str,
        conversation_id: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """运行Agent"""
        # 初始化状态
        initial_state = AgentState(
            conversation_id=conversation_id,
            messages=[HumanMessage(content=message)],
            context=context or {},
            capabilities=self.capabilities
        )
        
        # 运行状态图
        config = {"configurable": {"thread_id": conversation_id}}
        final_state = await self.graph.ainvoke(initial_state, config)
        
        # 返回结果
        return {
            "response": final_state.messages[-1].content if final_state.messages else "",
            "reasoning": final_state.reasoning_chain,
            "plan": final_state.plan,
            "tool_calls": final_state.tool_calls,
            "learning": final_state.learning_data,
            "metadata": {
                "agent_id": self.agent_id,
                "iterations": final_state.current_iteration,
                "capabilities_used": [c.value for c in self.capabilities],
                "memory_stats": {
                    "short_term": len(self.memory_system.short_term_memory),
                    "long_term": len(self.memory_system.long_term_memory),
                    "episodic": len(self.memory_system.episodic_memory)
                },
                "performance_metrics": self.learning_engine.performance_metrics
            }
        }
