"""
VoiceHelper v1.22.0 - 增强Agent系统
实现多Agent协作、工具调用、记忆系统和自主学习
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Agent类型"""
    TASK_EXECUTOR = "task_executor"
    TOOL_SPECIALIST = "tool_specialist"
    MEMORY_MANAGER = "memory_manager"
    COORDINATOR = "coordinator"
    ANALYZER = "analyzer"

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MemoryType(Enum):
    """记忆类型"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

@dataclass
class AgentCapability:
    """Agent能力"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    confidence: float = 1.0
    execution_time: float = 0.0

@dataclass
class Task:
    """任务"""
    id: str
    description: str
    agent_type: AgentType
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Memory:
    """记忆"""
    id: str
    content: str
    memory_type: MemoryType
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Tool:
    """工具"""
    name: str
    description: str
    function: Callable
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    category: str = "general"
    version: str = "1.0.0"

class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, max_short_term: int = 1000, max_long_term: int = 10000):
        self.short_term_memory = deque(maxlen=max_short_term)
        self.long_term_memory = {}
        self.memory_index = defaultdict(list)
        
    def store_memory(self, content: str, memory_type: MemoryType, 
                    importance: float = 0.5, tags: List[str] = None) -> str:
        """存储记忆"""
        memory_id = str(uuid.uuid4())
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            created_at=time.time()
        )
        
        if memory_type == MemoryType.SHORT_TERM:
            self.short_term_memory.append(memory)
        else:
            self.long_term_memory[memory_id] = memory
        
        # 更新索引
        for tag in memory.tags:
            self.memory_index[tag].append(memory_id)
        
        logger.debug(f"Stored memory: {memory_id} ({memory_type.value})")
        return memory_id
    
    def retrieve_memory(self, query: str, memory_type: Optional[MemoryType] = None,
                        limit: int = 10) -> List[Memory]:
        """检索记忆"""
        results = []
        
        # 搜索短期记忆
        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            for memory in self.short_term_memory:
                if query.lower() in memory.content.lower():
                    memory.last_accessed = time.time()
                    memory.access_count += 1
                    results.append(memory)
        
        # 搜索长期记忆
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            for memory in self.long_term_memory.values():
                if query.lower() in memory.content.lower():
                    memory.last_accessed = time.time()
                    memory.access_count += 1
                    results.append(memory)
        
        # 按重要性和访问时间排序
        results.sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
        return results[:limit]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "total_memories": len(self.short_term_memory) + len(self.long_term_memory),
            "index_size": len(self.memory_index)
        }

class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools = {}
        self.categories = defaultdict(list)
        
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        self.categories[tool.category].append(tool.name)
        logger.info(f"Registered tool: {tool.name} ({tool.category})")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """按类别获取工具"""
        return [self.tools[name] for name in self.categories[category]]
    
    def search_tools(self, query: str) -> List[Tool]:
        """搜索工具"""
        results = []
        query_lower = query.lower()
        
        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append(tool)
        
        return results
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """获取工具统计"""
        return {
            "total_tools": len(self.tools),
            "categories": dict(self.categories),
            "tools_by_category": {
                category: len(tools) 
                for category, tools in self.categories.items()
            }
        }

class Agent:
    """智能Agent"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.memory_manager = MemoryManager()
        self.tool_registry = ToolRegistry()
        self.learning_data = []
        
        # 性能统计
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        
    async def execute_task(self, task: Task) -> Any:
        """执行任务"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = time.time()
        
        try:
            logger.info(f"Agent {self.agent_id} executing task: {task.id}")
            
            # 根据任务类型选择执行策略
            if task.agent_type == AgentType.TASK_EXECUTOR:
                result = await self._execute_general_task(task)
            elif task.agent_type == AgentType.TOOL_SPECIALIST:
                result = await self._execute_tool_task(task)
            elif task.agent_type == AgentType.MEMORY_MANAGER:
                result = await self._execute_memory_task(task)
            elif task.agent_type == AgentType.COORDINATOR:
                result = await self._execute_coordination_task(task)
            elif task.agent_type == AgentType.ANALYZER:
                result = await self._execute_analysis_task(task)
            else:
                result = await self._execute_general_task(task)
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            
            # 更新统计
            self.tasks_completed += 1
            execution_time = task.completed_at - task.started_at
            self.total_execution_time += execution_time
            
            # 学习
            await self._learn_from_task(task, result)
            
            logger.info(f"Task {task.id} completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.tasks_failed += 1
            logger.error(f"Task {task.id} failed: {e}")
            raise
    
    async def _execute_general_task(self, task: Task) -> Any:
        """执行通用任务"""
        # 模拟任务执行
        await asyncio.sleep(0.1)
        return f"General task result for: {task.description}"
    
    async def _execute_tool_task(self, task: Task) -> Any:
        """执行工具任务"""
        # 模拟工具调用
        await asyncio.sleep(0.05)
        return f"Tool task result for: {task.description}"
    
    async def _execute_memory_task(self, task: Task) -> Any:
        """执行记忆任务"""
        # 模拟记忆操作
        await asyncio.sleep(0.02)
        return f"Memory task result for: {task.description}"
    
    async def _execute_coordination_task(self, task: Task) -> Any:
        """执行协调任务"""
        # 模拟协调操作
        await asyncio.sleep(0.08)
        return f"Coordination task result for: {task.description}"
    
    async def _execute_analysis_task(self, task: Task) -> Any:
        """执行分析任务"""
        # 模拟分析操作
        await asyncio.sleep(0.12)
        return f"Analysis task result for: {task.description}"
    
    async def _learn_from_task(self, task: Task, result: Any):
        """从任务中学习"""
        learning_data = {
            "task_id": task.id,
            "task_type": task.agent_type.value,
            "description": task.description,
            "execution_time": task.completed_at - task.started_at,
            "result": result,
            "timestamp": time.time()
        }
        
        self.learning_data.append(learning_data)
        
        # 存储到记忆中
        self.memory_manager.store_memory(
            content=f"Task: {task.description} -> Result: {result}",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            tags=["task_execution", task.agent_type.value]
        )
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """获取Agent统计"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.tasks_completed / (self.tasks_completed + self.tasks_failed) if (self.tasks_completed + self.tasks_failed) > 0 else 0,
            "avg_execution_time": self.total_execution_time / self.tasks_completed if self.tasks_completed > 0 else 0,
            "learning_data_count": len(self.learning_data),
            "memory_stats": self.memory_manager.get_memory_stats()
        }

class MultiAgentSystem:
    """多Agent系统"""
    
    def __init__(self):
        self.agents = {}
        self.task_queue = deque()
        self.completed_tasks = []
        self.failed_tasks = []
        
        # 初始化不同类型的Agent
        self._initialize_agents()
        
    def _initialize_agents(self):
        """初始化Agent"""
        # 任务执行Agent
        task_executor = Agent(
            agent_id="task_executor_001",
            agent_type=AgentType.TASK_EXECUTOR,
            capabilities=[
                AgentCapability("general_execution", "执行通用任务", ["text"], ["result"]),
                AgentCapability("data_processing", "数据处理", ["data"], ["processed_data"])
            ]
        )
        self.agents["task_executor_001"] = task_executor
        
        # 工具专家Agent
        tool_specialist = Agent(
            agent_id="tool_specialist_001",
            agent_type=AgentType.TOOL_SPECIALIST,
            capabilities=[
                AgentCapability("tool_calling", "工具调用", ["tool_name", "parameters"], ["tool_result"]),
                AgentCapability("tool_optimization", "工具优化", ["tool_usage"], ["optimized_usage"])
            ]
        )
        self.agents["tool_specialist_001"] = tool_specialist
        
        # 记忆管理Agent
        memory_manager = Agent(
            agent_id="memory_manager_001",
            agent_type=AgentType.MEMORY_MANAGER,
            capabilities=[
                AgentCapability("memory_storage", "记忆存储", ["content"], ["memory_id"]),
                AgentCapability("memory_retrieval", "记忆检索", ["query"], ["memories"])
            ]
        )
        self.agents["memory_manager_001"] = memory_manager
        
        # 协调Agent
        coordinator = Agent(
            agent_id="coordinator_001",
            agent_type=AgentType.COORDINATOR,
            capabilities=[
                AgentCapability("task_coordination", "任务协调", ["tasks"], ["coordination_plan"]),
                AgentCapability("resource_management", "资源管理", ["resources"], ["allocation"])
            ]
        )
        self.agents["coordinator_001"] = coordinator
        
        # 分析Agent
        analyzer = Agent(
            agent_id="analyzer_001",
            agent_type=AgentType.ANALYZER,
            capabilities=[
                AgentCapability("data_analysis", "数据分析", ["data"], ["analysis_result"]),
                AgentCapability("pattern_recognition", "模式识别", ["patterns"], ["recognized_patterns"])
            ]
        )
        self.agents["analyzer_001"] = analyzer
    
    async def submit_task(self, description: str, agent_type: AgentType, 
                         priority: int = 1, dependencies: List[str] = None) -> str:
        """提交任务"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            description=description,
            agent_type=agent_type,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.task_queue.append(task)
        logger.info(f"Task submitted: {task_id} ({agent_type.value})")
        return task_id
    
    async def execute_tasks(self, max_concurrent: int = 5):
        """执行任务"""
        active_tasks = []
        
        while self.task_queue or active_tasks:
            # 启动新任务
            while len(active_tasks) < max_concurrent and self.task_queue:
                task = self.task_queue.popleft()
                
                # 检查依赖
                if await self._check_dependencies(task):
                    # 选择最适合的Agent
                    agent = self._select_agent(task.agent_type)
                    
                    if agent:
                        # 异步执行任务
                        task_coroutine = agent.execute_task(task)
                        active_tasks.append((task, task_coroutine))
                    else:
                        logger.warning(f"No available agent for task: {task.id}")
                        self.failed_tasks.append(task)
            
            # 等待任务完成
            if active_tasks:
                completed_tasks = []
                for task, coroutine in active_tasks:
                    try:
                        result = await coroutine
                        self.completed_tasks.append(task)
                        completed_tasks.append((task, coroutine))
                    except Exception as e:
                        logger.error(f"Task {task.id} failed: {e}")
                        self.failed_tasks.append(task)
                        completed_tasks.append((task, coroutine))
                
                # 移除已完成的任务
                for completed_task in completed_tasks:
                    active_tasks.remove(completed_task)
        
        logger.info(f"Task execution completed. Completed: {len(self.completed_tasks)}, Failed: {len(self.failed_tasks)}")
    
    async def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            # 检查依赖任务是否已完成
            if not any(t.id == dep_id and t.status == TaskStatus.COMPLETED for t in self.completed_tasks):
                return False
        return True
    
    def _select_agent(self, agent_type: AgentType) -> Optional[Agent]:
        """选择Agent"""
        for agent in self.agents.values():
            if agent.agent_type == agent_type:
                return agent
        return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.get_agent_stats()
        
        return {
            "total_agents": len(self.agents),
            "pending_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "success_rate": len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) if (len(self.completed_tasks) + len(self.failed_tasks)) > 0 else 0,
            "agent_stats": agent_stats
        }

# 全局多Agent系统实例
multi_agent_system = MultiAgentSystem()

async def submit_agent_task(description: str, agent_type: AgentType, 
                           priority: int = 1, dependencies: List[str] = None) -> str:
    """提交Agent任务"""
    return await multi_agent_system.submit_task(description, agent_type, priority, dependencies)

async def execute_agent_tasks(max_concurrent: int = 5):
    """执行Agent任务"""
    await multi_agent_system.execute_tasks(max_concurrent)

def get_agent_system_stats() -> Dict[str, Any]:
    """获取Agent系统统计"""
    return multi_agent_system.get_system_stats()

if __name__ == "__main__":
    # 测试代码
    async def test_agent_system():
        # 提交一些测试任务
        task1 = await submit_agent_task("处理用户查询", AgentType.TASK_EXECUTOR, priority=1)
        task2 = await submit_agent_task("调用搜索工具", AgentType.TOOL_SPECIALIST, priority=2)
        task3 = await submit_agent_task("存储对话记忆", AgentType.MEMORY_MANAGER, priority=1)
        task4 = await submit_agent_task("协调多个任务", AgentType.COORDINATOR, priority=3)
        task5 = await submit_agent_task("分析用户行为", AgentType.ANALYZER, priority=2)
        
        # 执行任务
        await execute_agent_tasks(max_concurrent=3)
        
        # 获取统计信息
        stats = get_agent_system_stats()
        print("Agent系统统计:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    asyncio.run(test_agent_system())
