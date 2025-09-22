"""
聊天语音Agent图
基于V2架构的BaseAgentGraph实现多模态对话流程
"""

import json
from typing import Iterator, Dict, Any, Optional, List
from loguru import logger

from ..base.runnable import BaseAgentGraph
from ..base.protocols import StreamCallback


class ChatVoiceAgentGraph(BaseAgentGraph):
    """聊天语音 Agent 图"""
    
    def __init__(
        self,
        retriever,
        tools=None,
        llm_service=None,
        tts_service=None,
        **kwargs
    ):
        super().__init__(retriever=retriever, tools=tools or [], **kwargs)
        self.llm_service = llm_service
        self.tts_service = tts_service
    
    def stream(self, query: str, *, cb: Optional[StreamCallback] = None) -> Iterator[Dict[str, Any]]:
        """
        流式处理对话
        
        Args:
            query: 用户查询
            cb: 回调函数
            
        Yields:
            流式结果事件
        """
        try:
            logger.info(f"开始处理对话: {query[:100]}...")
            
            # 1. 意图识别
            self.emit(cb, "agent_plan", {"step": "intent", "query": query})
            intent = self._analyze_intent(query)
            yield {"event": "intent", "data": intent}
            
            # 2. 检索（如需要）
            docs = []
            if intent.get("need_retrieval", True):
                self.emit(cb, "agent_step", {"step": "retrieve", "query": query})
                try:
                    docs = self.retriever.retrieve_with_callback(query, cb, top_k=5)
                    yield {"event": "retrieve", "data": {"docs": docs, "count": len(docs)}}
                except Exception as e:
                    logger.warning(f"检索失败，继续处理: {e}")
                    yield {"event": "retrieve_error", "data": {"error": str(e)}}
            
            # 3. 计划生成
            self.emit(cb, "agent_step", {"step": "plan"})
            plan = self._generate_plan(query, intent, docs)
            yield {"event": "plan", "data": plan}
            
            # 4. 工具执行
            tool_results = []
            for step in plan.get("steps", []):
                if step.get("tool"):
                    tool_name = step["tool"]
                    tool = self._get_tool_by_name(tool_name)
                    if tool:
                        try:
                            self.emit(cb, "agent_step", {"step": "tool_execute", "tool": tool_name})
                            result = tool.run_with_callback(cb, **step.get("args", {}))
                            tool_results.append({"tool": tool_name, "result": result})
                            yield {"event": "tool_result", "data": {"tool": tool_name, "result": result}}
                        except Exception as e:
                            logger.error(f"工具执行失败: {tool_name}, {e}")
                            yield {"event": "tool_error", "data": {"tool": tool_name, "error": str(e)}}
            
            # 5. 综合回答
            self.emit(cb, "agent_step", {"step": "synthesize"})
            answer = self._synthesize_answer(query, intent, docs, tool_results)
            yield {"event": "answer", "data": answer}
            
            # 6. TTS（如需要）
            if intent.get("need_tts", False) and self.tts_service:
                self.emit(cb, "agent_step", {"step": "tts"})
                try:
                    for audio_chunk in self._text_to_speech(answer["text"]):
                        yield {"event": "audio", "data": audio_chunk}
                except Exception as e:
                    logger.error(f"TTS失败: {e}")
                    yield {"event": "tts_error", "data": {"error": str(e)}}
            
            self.emit(cb, "agent_summary", {"status": "completed", "query": query})
            yield {"event": "completed", "data": {"status": "success"}}
            
        except Exception as e:
            logger.error(f"Agent处理失败: {e}")
            self.emit(cb, "agent_error", {"error": str(e)})
            yield {"event": "error", "data": {"error": str(e)}}
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        意图分析
        
        Args:
            query: 用户查询
            
        Returns:
            意图分析结果
        """
        # 简单的意图分析逻辑
        intent = {
            "need_retrieval": True,
            "need_tts": False,
            "category": "general",
            "confidence": 0.8
        }
        
        # 检查是否需要检索
        no_retrieval_keywords = ["你好", "再见", "谢谢", "时间", "天气"]
        if any(keyword in query for keyword in no_retrieval_keywords):
            intent["need_retrieval"] = False
            intent["category"] = "greeting"
        
        # 检查是否需要TTS
        if "语音" in query or "说出来" in query:
            intent["need_tts"] = True
        
        # 检查是否需要工具
        intent["need_tools"] = []
        if "网址" in query or "链接" in query or "http" in query:
            intent["need_tools"].append("fetch_url")
        if "文件" in query or "读取" in query:
            intent["need_tools"].append("read_file")
        if "github" in query.lower() or "代码" in query:
            intent["need_tools"].append("read_github")
        
        logger.info(f"意图分析结果: {intent}")
        return intent
    
    def _generate_plan(self, query: str, intent: Dict[str, Any], docs: List[Dict]) -> Dict[str, Any]:
        """
        生成执行计划
        
        Args:
            query: 用户查询
            intent: 意图分析结果
            docs: 检索到的文档
            
        Returns:
            执行计划
        """
        plan = {
            "query": query,
            "strategy": "rag_enhanced" if docs else "direct_answer",
            "steps": []
        }
        
        # 根据意图添加工具执行步骤
        for tool_name in intent.get("need_tools", []):
            if tool_name == "fetch_url":
                # 尝试从查询中提取URL
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', query)
                if urls:
                    plan["steps"].append({
                        "tool": "fetch_url",
                        "args": {"url": urls[0]}
                    })
            elif tool_name == "read_file":
                # 尝试从查询中提取文件路径
                import re
                files = re.findall(r'["\']([^"\']+\.[a-zA-Z]+)["\']', query)
                if files:
                    plan["steps"].append({
                        "tool": "read_file", 
                        "args": {"file_path": files[0]}
                    })
            elif tool_name == "read_github":
                # 尝试从查询中提取GitHub信息
                import re
                github_pattern = r'github\.com/([^/]+/[^/]+)(?:/[^/]+/[^/]+/(.+))?'
                matches = re.findall(github_pattern, query)
                if matches:
                    repo, path = matches[0]
                    if path:
                        plan["steps"].append({
                            "tool": "read_github",
                            "args": {"repo": repo, "path": path}
                        })
        
        logger.info(f"执行计划: {plan}")
        return plan
    
    def _synthesize_answer(
        self, 
        query: str, 
        intent: Dict[str, Any], 
        docs: List[Dict], 
        tool_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        综合答案
        
        Args:
            query: 用户查询
            intent: 意图分析结果
            docs: 检索文档
            tool_results: 工具执行结果
            
        Returns:
            综合答案
        """
        # 构建上下文
        context_parts = []
        
        # 添加检索文档
        if docs:
            context_parts.append("相关文档:")
            for i, doc in enumerate(docs[:3], 1):
                context_parts.append(f"{i}. {doc.get('content', '')[:200]}...")
        
        # 添加工具结果
        if tool_results:
            context_parts.append("工具执行结果:")
            for result in tool_results:
                tool_name = result["tool"]
                tool_data = result["result"]
                if tool_data.get("status") == "success":
                    content = tool_data.get("text") or tool_data.get("content", "")
                    context_parts.append(f"{tool_name}: {content[:300]}...")
        
        context = "\n".join(context_parts)
        
        # 使用LLM生成答案
        if self.llm_service:
            try:
                prompt = self._build_prompt(query, context, intent)
                response = self.llm_service.generate(prompt)
                answer_text = response.get("text", "抱歉，我无法生成回答。")
            except Exception as e:
                logger.error(f"LLM生成失败: {e}")
                answer_text = "抱歉，我遇到了一些技术问题，无法生成回答。"
        else:
            # 简单的模板回答
            if docs or tool_results:
                answer_text = f"根据相关信息，关于'{query}'的回答是：\n\n{context[:500]}..."
            else:
                answer_text = f"关于'{query}'，我需要更多信息才能给出准确回答。"
        
        answer = {
            "text": answer_text,
            "references": [doc.get("source", "") for doc in docs if doc.get("source")],
            "tools_used": [result["tool"] for result in tool_results],
            "confidence": 0.8 if docs or tool_results else 0.5
        }
        
        logger.info(f"生成答案: {len(answer_text)} 字符")
        return answer
    
    def _build_prompt(self, query: str, context: str, intent: Dict[str, Any]) -> str:
        """
        构建LLM提示词
        
        Args:
            query: 用户查询
            context: 上下文信息
            intent: 意图信息
            
        Returns:
            提示词
        """
        prompt_parts = [
            "你是一个智能助手，请根据提供的上下文信息回答用户问题。",
            "",
            f"用户问题: {query}",
            "",
        ]
        
        if context.strip():
            prompt_parts.extend([
                "相关上下文:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "请提供准确、有用的回答。如果上下文信息不足，请说明需要更多信息。",
            "回答应该简洁明了，重点突出。"
        ])
        
        return "\n".join(prompt_parts)
    
    def _text_to_speech(self, text: str) -> Iterator[bytes]:
        """
        文本转语音
        
        Args:
            text: 要转换的文本
            
        Yields:
            音频数据块
        """
        if not self.tts_service:
            logger.warning("TTS服务未配置")
            return
        
        try:
            # 分段处理长文本
            max_chunk_size = 200
            text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            for chunk in text_chunks:
                if chunk.strip():
                    audio_data = self.tts_service.synthesize(chunk)
                    if audio_data:
                        yield audio_data
                        
        except Exception as e:
            logger.error(f"TTS处理失败: {e}")
            raise
