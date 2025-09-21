"""
LangGraph Agent 实现
基于状态机的智能代理，支持意图识别、RAG检索、工具调用等
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import Tool
from langchain_core.runnables import RunnableConfig

from core.config import config
from core.retrieve import RetrieveService
from core.models import QueryRequest, Reference

class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Sequence[BaseMessage]
    conversation_id: str
    tenant_id: str
    user_query: str
    intent: Optional[str]
    plan: Optional[List[str]]
    retrieved_docs: Optional[List[Dict]]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]
    final_answer: Optional[str]
    references: Optional[List[Reference]]
    error: Optional[str]
    metadata: Dict[str, Any]

class IntentType(Enum):
    """意图类型"""
    QA = "qa"  # 问答
    SEARCH = "search"  # 检索
    TOOL_USE = "tool_use"  # 工具调用
    CHITCHAT = "chitchat"  # 闲聊
    UNKNOWN = "unknown"

class LangGraphAgent:
    """基于LangGraph的智能代理"""
    
    def __init__(self, retrieve_service: RetrieveService):
        self.retrieve_service = retrieve_service
        self.graph = self._build_graph()
        self.tools = self._init_tools()
        self.tool_executor = ToolExecutor(self.tools)
        
    def _build_graph(self) -> StateGraph:
        """构建状态图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("intent_recognition", self.intent_recognition)
        workflow.add_node("make_plan", self.make_plan)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("execute_tools", self.execute_tools)
        workflow.add_node("synthesize", self.synthesize_answer)
        workflow.add_node("error_handler", self.handle_error)
        
        # 设置入口
        workflow.set_entry_point("intent_recognition")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "intent_recognition",
            self._route_by_intent,
            {
                IntentType.QA.value: "retrieve",
                IntentType.SEARCH.value: "retrieve",
                IntentType.TOOL_USE.value: "make_plan",
                IntentType.CHITCHAT.value: "synthesize",
                IntentType.UNKNOWN.value: "error_handler",
            }
        )
        
        workflow.add_edge("make_plan", "execute_tools")
        workflow.add_edge("retrieve", "synthesize")
        workflow.add_edge("execute_tools", "synthesize")
        workflow.add_edge("synthesize", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    def _init_tools(self) -> List[Tool]:
        """初始化工具"""
        tools = []
        
        # 文件系统工具（只读）
        fs_tool = Tool(
            name="read_file",
            func=self._read_file,
            description="Read content from a file"
        )
        tools.append(fs_tool)
        
        # HTTP工具（只读）
        http_tool = Tool(
            name="fetch_url",
            func=self._fetch_url,
            description="Fetch content from a URL"
        )
        tools.append(http_tool)
        
        # 数据库查询工具（只读）
        db_tool = Tool(
            name="query_database",
            func=self._query_database,
            description="Query database with SQL (read-only)"
        )
        tools.append(db_tool)
        
        return tools
    
    async def intent_recognition(self, state: AgentState) -> AgentState:
        """意图识别节点"""
        try:
            user_query = state["user_query"]
            
            # 简化的意图识别逻辑
            if any(kw in user_query for kw in ["查询", "搜索", "找", "检索"]):
                intent = IntentType.SEARCH.value
            elif any(kw in user_query for kw in ["读取", "获取", "查看", "文件"]):
                intent = IntentType.TOOL_USE.value
            elif any(kw in user_query for kw in ["你好", "谢谢", "再见"]):
                intent = IntentType.CHITCHAT.value
            else:
                intent = IntentType.QA.value
            
            state["intent"] = intent
            
            # 发送Agent事件
            await self._emit_event({
                "type": "agent_intent",
                "intent": intent,
                "query": user_query
            })
            
            return state
            
        except Exception as e:
            state["error"] = f"Intent recognition error: {str(e)}"
            return state
    
    async def make_plan(self, state: AgentState) -> AgentState:
        """制定执行计划"""
        try:
            intent = state["intent"]
            user_query = state["user_query"]
            
            # 根据意图制定计划
            if intent == IntentType.TOOL_USE.value:
                # 分析需要调用哪些工具
                plan = []
                if "文件" in user_query:
                    plan.append("read_file")
                if "网址" in user_query or "URL" in user_query:
                    plan.append("fetch_url")
                if "数据库" in user_query or "SQL" in user_query:
                    plan.append("query_database")
                
                state["plan"] = plan
                
                # 发送计划事件
                await self._emit_event({
                    "type": "agent_plan",
                    "items": plan
                })
            
            return state
            
        except Exception as e:
            state["error"] = f"Planning error: {str(e)}"
            return state
    
    async def retrieve_documents(self, state: AgentState) -> AgentState:
        """检索文档节点"""
        try:
            user_query = state["user_query"]
            
            # 调用检索服务
            from core.models import QueryRequest, Message
            query_request = QueryRequest(
                messages=[Message(role="user", content=user_query)],
                top_k=5,
                temperature=0.3
            )
            
            retrieved_docs = []
            references = []
            
            async for response in self.retrieve_service.stream_query(query_request):
                response_data = json.loads(response)
                if response_data["type"] == "refs" and response_data.get("refs"):
                    for ref in response_data["refs"]:
                        references.append(Reference(**ref))
                        retrieved_docs.append({
                            "content": ref.get("content", ""),
                            "source": ref.get("source", ""),
                            "score": ref.get("score", 0)
                        })
            
            state["retrieved_docs"] = retrieved_docs
            state["references"] = references
            
            # 发送检索事件
            await self._emit_event({
                "type": "agent_retrieve",
                "doc_count": len(retrieved_docs),
                "top_score": retrieved_docs[0]["score"] if retrieved_docs else 0
            })
            
            return state
            
        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            return state
    
    async def execute_tools(self, state: AgentState) -> AgentState:
        """执行工具调用"""
        try:
            plan = state.get("plan", [])
            tool_results = []
            
            for tool_name in plan:
                # 发送工具调用事件
                await self._emit_event({
                    "type": "agent_tool_start",
                    "tool": tool_name
                })
                
                # 执行工具（这里简化处理）
                if tool_name == "read_file":
                    result = await self._read_file("/example/file.txt")
                elif tool_name == "fetch_url":
                    result = await self._fetch_url("https://example.com")
                elif tool_name == "query_database":
                    result = await self._query_database("SELECT * FROM users LIMIT 1")
                else:
                    result = f"Unknown tool: {tool_name}"
                
                tool_results.append({
                    "tool": tool_name,
                    "result": result
                })
                
                # 发送工具结果事件
                await self._emit_event({
                    "type": "agent_tool_result",
                    "tool": tool_name,
                    "success": True
                })
            
            state["tool_results"] = tool_results
            return state
            
        except Exception as e:
            state["error"] = f"Tool execution error: {str(e)}"
            return state
    
    async def synthesize_answer(self, state: AgentState) -> AgentState:
        """综合生成答案"""
        try:
            intent = state.get("intent")
            retrieved_docs = state.get("retrieved_docs", [])
            tool_results = state.get("tool_results", [])
            user_query = state["user_query"]
            
            # 构建上下文
            context = ""
            if retrieved_docs:
                context = "\n".join([doc["content"] for doc in retrieved_docs[:3]])
            
            if tool_results:
                context += "\n\n工具调用结果：\n"
                for result in tool_results:
                    context += f"- {result['tool']}: {result['result']}\n"
            
            # 生成答案（这里简化处理）
            if intent == IntentType.CHITCHAT.value:
                final_answer = self._generate_chitchat_response(user_query)
            else:
                final_answer = self._generate_contextual_answer(user_query, context)
            
            state["final_answer"] = final_answer
            
            # 发送综合事件
            await self._emit_event({
                "type": "agent_summary",
                "answer_length": len(final_answer),
                "has_refs": bool(state.get("references"))
            })
            
            return state
            
        except Exception as e:
            state["error"] = f"Synthesis error: {str(e)}"
            return state
    
    async def handle_error(self, state: AgentState) -> AgentState:
        """错误处理节点"""
        error = state.get("error", "Unknown error")
        state["final_answer"] = f"抱歉，处理您的请求时出现错误：{error}"
        
        # 发送错误事件
        await self._emit_event({
            "type": "agent_error",
            "error": error
        })
        
        return state
    
    def _route_by_intent(self, state: AgentState) -> str:
        """根据意图路由到下一个节点"""
        intent = state.get("intent", IntentType.UNKNOWN.value)
        return intent
    
    async def _read_file(self, filepath: str) -> str:
        """读取文件工具"""
        # 安全检查
        if ".." in filepath or filepath.startswith("/"):
            return "Access denied: Invalid file path"
        
        # 模拟文件读取
        return f"File content of {filepath}: [示例内容]"
    
    async def _fetch_url(self, url: str) -> str:
        """获取URL内容工具"""
        # URL白名单检查
        allowed_domains = ["example.com", "docs.example.com"]
        if not any(domain in url for domain in allowed_domains):
            return "Access denied: URL not in whitelist"
        
        # 模拟HTTP请求
        return f"Content from {url}: [示例网页内容]"
    
    async def _query_database(self, sql: str) -> str:
        """数据库查询工具"""
        # SQL注入防护
        if any(kw in sql.upper() for kw in ["DROP", "DELETE", "UPDATE", "INSERT"]):
            return "Access denied: Only SELECT queries allowed"
        
        # 模拟数据库查询
        return f"Query result: [示例查询结果]"
    
    def _generate_chitchat_response(self, query: str) -> str:
        """生成闲聊回复"""
        responses = {
            "你好": "您好！有什么可以帮助您的吗？",
            "谢谢": "不客气，很高兴能帮助到您！",
            "再见": "再见！祝您有美好的一天！"
        }
        
        for key, response in responses.items():
            if key in query:
                return response
        
        return "我理解您的意思，请问有什么具体问题需要帮助吗？"
    
    def _generate_contextual_answer(self, query: str, context: str) -> str:
        """基于上下文生成答案"""
        if context:
            return f"根据相关资料，{context[:200]}..."
        else:
            return "抱歉，我没有找到相关的信息来回答您的问题。"
    
    async def _emit_event(self, event: Dict[str, Any]):
        """发送Agent事件"""
        print(f"Agent Event: {json.dumps(event, ensure_ascii=False)}")
    
    async def run(self, query: str, conversation_id: str, tenant_id: str) -> Dict[str, Any]:
        """运行Agent"""
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_query=query,
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # 执行图
        config = RunnableConfig(configurable={"thread_id": conversation_id})
        result = await self.graph.ainvoke(initial_state, config)
        
        return {
            "answer": result.get("final_answer", ""),
            "references": result.get("references", []),
            "intent": result.get("intent", ""),
            "error": result.get("error")
        }
