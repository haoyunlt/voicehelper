"""
LangGraph Agent实现
基于LangGraph的状态机Agent，支持规划、工具调用、记忆等功能
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint import MemorySaver
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from loguru import logger

from core.llm import LLMService
from core.bge_faiss_rag import BGEFAISSRAGService
from core.rag import FaissRetriever, BgeEmbedder
from common.errors import VoiceHelperError


class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    TOOL_CALLING = "tool_calling"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentContext:
    """Agent上下文"""
    conversation_id: str
    user_id: str
    session_id: str
    messages: List[BaseMessage] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    current_step: int = 0
    state: AgentState = AgentState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class AgentEvent:
    """Agent事件"""
    
    def __init__(self, event_type: str, data: Dict[str, Any], context: AgentContext):
        self.event_type = event_type
        self.data = data
        self.context = context
        self.timestamp = datetime.now()
        self.event_id = str(uuid.uuid4())


class LangGraphAgent:
    """基于LangGraph的Agent实现"""
    
    def __init__(
        self,
        llm_service: LLMService,
        rag_service: Optional[BGEFAISSRAGService] = None,
        retriever: Optional[FaissRetriever] = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        enable_planning: bool = True,
        enable_memory: bool = True
    ):
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.retriever = retriever
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.enable_planning = enable_planning
        self.enable_memory = enable_memory
        
        # 工具执行器
        self.tool_executor = ToolExecutor(self.tools) if self.tools else None
        
        # 检查点保存器（用于记忆）
        self.checkpointer = MemorySaver() if enable_memory else None
        
        # 构建状态图
        self.graph = self._build_graph()
        
        # 事件回调
        self.event_callbacks: List[Callable[[AgentEvent], None]] = []
        
        logger.info(f"LangGraph Agent初始化完成，工具数: {len(self.tools)}")
    
    def _build_graph(self) -> StateGraph:
        """构建Agent状态图"""
        # 定义状态图
        workflow = StateGraph(AgentContext)
        
        # 添加节点
        workflow.add_node("start", self._start_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("tool_call", self._tool_call_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("end", self._end_node)
        
        # 设置入口点
        workflow.set_entry_point("start")
        
        # 添加边
        workflow.add_edge("start", "plan")
        workflow.add_conditional_edges(
            "plan",
            self._should_retrieve,
            {
                "retrieve": "retrieve",
                "tool_call": "tool_call",
                "synthesize": "synthesize"
            }
        )
        workflow.add_edge("retrieve", "tool_call")
        workflow.add_conditional_edges(
            "tool_call",
            self._should_continue,
            {
                "continue": "tool_call",
                "synthesize": "synthesize",
                "end": "end"
            }
        )
        workflow.add_edge("synthesize", "end")
        workflow.add_edge("end", END)
        
        # 编译图
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def process_message(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """处理用户消息"""
        try:
            # 创建上下文
            context = AgentContext(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id or str(uuid.uuid4()),
                messages=[HumanMessage(content=message)]
            )
            
            # 发送开始事件
            yield AgentEvent("agent_start", {
                "message": message,
                "conversation_id": conversation_id
            }, context)
            
            # 运行状态图
            config = {"configurable": {"thread_id": context.session_id}}
            
            async for chunk in self.graph.astream(context, config=config):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, AgentContext):
                        context = node_output
                        
                        # 发送状态更新事件
                        yield AgentEvent("agent_state_update", {
                            "node": node_name,
                            "state": context.state.value,
                            "step": context.current_step
                        }, context)
            
            # 发送完成事件
            yield AgentEvent("agent_complete", {
                "final_state": context.state.value,
                "total_steps": context.current_step,
                "tool_results": context.tool_results
            }, context)
            
        except Exception as e:
            logger.error(f"Agent处理消息失败: {e}")
            error_context = AgentContext(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id or str(uuid.uuid4()),
                state=AgentState.ERROR
            )
            
            yield AgentEvent("agent_error", {
                "error": str(e),
                "error_type": type(e).__name__
            }, error_context)
    
    async def _start_node(self, context: AgentContext) -> AgentContext:
        """开始节点"""
        context.state = AgentState.PLANNING
        context.current_step = 1
        context.updated_at = datetime.now()
        
        logger.info(f"Agent开始处理: {context.conversation_id}")
        return context
    
    async def _plan_node(self, context: AgentContext) -> AgentContext:
        """规划节点"""
        if not self.enable_planning:
            context.state = AgentState.RETRIEVING
            return context
        
        try:
            context.state = AgentState.PLANNING
            
            # 获取最后一条用户消息
            user_message = None
            for msg in reversed(context.messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if not user_message:
                raise VoiceHelperError("NO_USER_MESSAGE", "No user message found")
            
            # 生成执行计划
            plan_prompt = f"""
分析用户问题并制定执行计划。

用户问题：{user_message}

可用工具：{[tool.name for tool in self.tools]}

请分析这个问题需要：
1. 是否需要检索相关文档？
2. 需要调用哪些工具？
3. 执行步骤是什么？

以JSON格式返回计划：
{{
    "need_retrieval": true/false,
    "tools_needed": ["tool1", "tool2"],
    "steps": ["步骤1", "步骤2"],
    "reasoning": "规划理由"
}}
"""
            
            response = await self.llm_service.generate(
                prompt=plan_prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # 解析计划
            try:
                plan = json.loads(response.get("content", "{}"))
            except json.JSONDecodeError:
                plan = {
                    "need_retrieval": True,
                    "tools_needed": [],
                    "steps": ["分析问题", "生成回答"],
                    "reasoning": "使用默认计划"
                }
            
            context.plan = plan
            context.current_step += 1
            
            # 触发事件
            self._emit_event(AgentEvent("agent_plan", {
                "plan": plan,
                "user_message": user_message
            }, context))
            
            logger.info(f"Agent规划完成: {plan}")
            
        except Exception as e:
            logger.error(f"Agent规划失败: {e}")
            context.state = AgentState.ERROR
            context.metadata["error"] = str(e)
        
        return context
    
    async def _retrieve_node(self, context: AgentContext) -> AgentContext:
        """检索节点"""
        try:
            context.state = AgentState.RETRIEVING
            
            # 获取用户问题
            user_message = None
            for msg in reversed(context.messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if not user_message:
                logger.warning("未找到用户消息")
                return context
            
            retrieved_docs = []
            
            # 优先使用新的FaissRetriever
            if self.retriever:
                try:
                    retrieved_docs = self.retriever.retrieve(
                        query=user_message,
                        top_k=5
                    )
                    logger.info(f"使用FaissRetriever检索完成，结果数: {len(retrieved_docs)}")
                except Exception as e:
                    logger.error(f"FaissRetriever检索失败: {e}")
            
            # 回退到原有的RAG服务
            elif self.rag_service:
                try:
                    retrieved_docs = await self.rag_service.retrieve(
                        query=user_message,
                        top_k=5
                    )
                    logger.info(f"使用RAG服务检索完成，结果数: {len(retrieved_docs)}")
                except Exception as e:
                    logger.error(f"RAG服务检索失败: {e}")
            
            context.retrieved_docs = retrieved_docs
            context.current_step += 1
            
            # 触发事件
            self._emit_event(AgentEvent("agent_retrieve", {
                "query": user_message,
                "results_count": len(retrieved_docs),
                "results": retrieved_docs[:3],  # 只发送前3个结果
                "retriever_type": "faiss" if self.retriever else "rag_service"
            }, context))
            
            logger.info(f"Agent检索完成，结果数: {len(retrieved_docs)}")
            
        except Exception as e:
            logger.error(f"Agent检索失败: {e}")
            context.state = AgentState.ERROR
            context.metadata["error"] = str(e)
        
        return context
    
    async def _tool_call_node(self, context: AgentContext) -> AgentContext:
        """工具调用节点"""
        try:
            context.state = AgentState.TOOL_CALLING
            
            if not self.tools or not self.tool_executor:
                # 没有工具，直接跳到合成
                context.state = AgentState.SYNTHESIZING
                return context
            
            # 确定需要调用的工具
            tools_to_call = []
            if context.plan and context.plan.get("tools_needed"):
                tools_to_call = context.plan["tools_needed"]
            
            # 执行工具调用
            for tool_name in tools_to_call:
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    continue
                
                try:
                    # 准备工具输入
                    tool_input = self._prepare_tool_input(tool, context)
                    
                    # 执行工具
                    start_time = datetime.now()
                    result = await tool.arun(tool_input)
                    end_time = datetime.now()
                    
                    # 保存结果
                    tool_result = {
                        "tool_name": tool_name,
                        "input": tool_input,
                        "output": result,
                        "success": True,
                        "duration_ms": int((end_time - start_time).total_seconds() * 1000),
                        "timestamp": start_time.isoformat()
                    }
                    
                    context.tool_results.append(tool_result)
                    
                    # 触发事件
                    self._emit_event(AgentEvent("agent_tool_result", tool_result, context))
                    
                    logger.info(f"工具调用成功: {tool_name}")
                    
                except Exception as tool_error:
                    tool_result = {
                        "tool_name": tool_name,
                        "input": tool_input,
                        "output": None,
                        "success": False,
                        "error": str(tool_error),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    context.tool_results.append(tool_result)
                    logger.error(f"工具调用失败: {tool_name}, 错误: {tool_error}")
            
            context.current_step += 1
            
        except Exception as e:
            logger.error(f"Agent工具调用失败: {e}")
            context.state = AgentState.ERROR
            context.metadata["error"] = str(e)
        
        return context
    
    async def _synthesize_node(self, context: AgentContext) -> AgentContext:
        """合成节点"""
        try:
            context.state = AgentState.SYNTHESIZING
            
            # 获取用户问题
            user_message = None
            for msg in reversed(context.messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            # 构建上下文信息
            context_parts = []
            
            # 添加检索到的文档
            if context.retrieved_docs:
                docs_text = "\n".join([
                    f"文档{i+1}: {doc['content'][:300]}..."
                    for i, doc in enumerate(context.retrieved_docs[:3])
                ])
                context_parts.append(f"相关文档：\n{docs_text}")
            
            # 添加工具结果
            if context.tool_results:
                tools_text = "\n".join([
                    f"工具{result['tool_name']}结果: {str(result['output'])[:200]}..."
                    for result in context.tool_results
                    if result['success']
                ])
                if tools_text:
                    context_parts.append(f"工具执行结果：\n{tools_text}")
            
            # 构建最终提示词
            context_text = "\n\n".join(context_parts)
            
            final_prompt = f"""
基于以下信息回答用户问题：

{context_text}

用户问题：{user_message}

请提供准确、有用的回答。如果信息不足，请说明。
"""
            
            # 生成最终回答
            response = await self.llm_service.generate(
                prompt=final_prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 添加AI回复到消息历史
            ai_message = AIMessage(content=response.get("content", ""))
            context.messages.append(ai_message)
            
            context.current_step += 1
            
            # 触发事件
            self._emit_event(AgentEvent("agent_synthesize", {
                "response": response.get("content", ""),
                "context_used": len(context_parts) > 0,
                "references": context.retrieved_docs,
                "tool_results": context.tool_results
            }, context))
            
            logger.info("Agent合成完成")
            
        except Exception as e:
            logger.error(f"Agent合成失败: {e}")
            context.state = AgentState.ERROR
            context.metadata["error"] = str(e)
        
        return context
    
    async def _end_node(self, context: AgentContext) -> AgentContext:
        """结束节点"""
        context.state = AgentState.COMPLETED
        context.updated_at = datetime.now()
        
        logger.info(f"Agent处理完成: {context.conversation_id}")
        return context
    
    def _should_retrieve(self, context: AgentContext) -> str:
        """判断是否需要检索"""
        if context.plan and context.plan.get("need_retrieval", True):
            return "retrieve"
        elif context.plan and context.plan.get("tools_needed"):
            return "tool_call"
        else:
            return "synthesize"
    
    def _should_continue(self, context: AgentContext) -> str:
        """判断是否继续工具调用"""
        if context.current_step >= self.max_iterations:
            return "end"
        
        # 检查是否还有工具需要调用
        if context.plan and context.plan.get("tools_needed"):
            called_tools = {result["tool_name"] for result in context.tool_results}
            needed_tools = set(context.plan["tools_needed"])
            
            if needed_tools - called_tools:
                return "continue"
        
        return "synthesize"
    
    def _prepare_tool_input(self, tool: BaseTool, context: AgentContext) -> str:
        """准备工具输入"""
        # 获取用户消息
        user_message = ""
        for msg in reversed(context.messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        # 简单的输入准备，实际应用中可能需要更复杂的逻辑
        return user_message
    
    def _emit_event(self, event: AgentEvent):
        """触发事件"""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"事件回调失败: {e}")
    
    def add_event_callback(self, callback: Callable[[AgentEvent], None]):
        """添加事件回调"""
        self.event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[AgentEvent], None]):
        """移除事件回调"""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """获取会话历史"""
        if not self.checkpointer:
            return []
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.graph.get_state(config)
            if state and hasattr(state, 'messages'):
                return state.messages
        except Exception as e:
            logger.error(f"获取会话历史失败: {e}")
        
        return []
