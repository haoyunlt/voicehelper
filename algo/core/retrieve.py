import json
import asyncio
from typing import AsyncGenerator, List, Dict, Any
from datetime import datetime

import requests
from langchain_milvus import Milvus
from pymilvus import Collection

from core.config import config
from core.embeddings import get_embeddings
from core.models import QueryRequest, QueryResponse, Reference, Message

class RetrieveService:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.milvus = Milvus(
            embedding_function=self.embeddings,
            collection_name=config.DEFAULT_COLLECTION_NAME,
            connection_args={
                "host": config.MILVUS_HOST,
                "port": config.MILVUS_PORT,
                "user": config.MILVUS_USER,
                "password": config.MILVUS_PASSWORD,
            }
        )
    
    async def stream_query(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        """流式查询处理"""
        try:
            # 1. 提取用户查询
            user_query = self._extract_user_query(request.messages)
            if not user_query:
                yield self._format_response("error", "No user query found")
                return
            
            # 2. 检索相关文档
            references = await self._retrieve_documents(
                user_query, 
                request.top_k,
                request.filters
            )
            
            # 3. 发送引用信息
            if references:
                yield self._format_response("refs", refs=references)
            
            # 4. 构建提示词
            prompt = self._build_prompt(request.messages, references)
            
            # 5. 调用大模型流式生成
            async for response in self._stream_llm_response(prompt, request):
                yield response
            
            # 6. 发送结束信号
            yield self._format_response("end")
            
        except Exception as e:
            print(f"Error in stream_query: {e}")
            yield self._format_response("error", str(e))
    
    def _extract_user_query(self, messages: List[Message]) -> str:
        """提取用户查询"""
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""
    
    async def _retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[Reference]:
        """检索相关文档"""
        try:
            # 构建过滤表达式
            expr = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(f'{key} == "{value}"')
                    else:
                        conditions.append(f'{key} == {value}')
                if conditions:
                    expr = " and ".join(conditions)
            
            # 执行相似性搜索
            results = self.milvus.similarity_search_with_score(
                query=query,
                k=top_k,
                expr=expr
            )
            
            # 转换为引用格式
            references = []
            for doc, score in results:
                if score >= config.DEFAULT_SIMILARITY_THRESHOLD:
                    references.append(Reference(
                        chunk_id=doc.metadata.get('chunk_id', ''),
                        source=doc.metadata.get('source', ''),
                        score=float(score),
                        content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    ))
            
            return references
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def _build_prompt(self, messages: List[Message], references: List[Reference]) -> List[Dict[str, str]]:
        """构建提示词"""
        # 系统提示词
        system_prompt = """你是企业知识助手，只基于检索到的片段回答用户问题。

回答要求：
- 使用中文回答
- 基于提供的片段内容回答，给出引用编号，例如[1][2]
- 先给出结论，再提供依据，简洁分点说明
- 如需步骤说明，使用1/2/3编号
- 如果片段中没有找到相关信息，明确告知"没有找到相关依据"，并提出可行的下一步建议"""
        
        # 构建上下文
        context_parts = []
        if references:
            context_parts.append("可用片段（带编号与来源）：")
            for i, ref in enumerate(references, 1):
                context_parts.append(f"[{i}] 来源：{ref.source}")
                context_parts.append(f"内容：{ref.content}")
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # 构建消息列表
        prompt_messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史消息（保留最近几轮对话）
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        for msg in recent_messages[:-1]:  # 除了最后一条用户消息
            if msg.role != "system":
                prompt_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # 添加当前用户查询和上下文
        user_content = recent_messages[-1].content
        if context:
            user_content = f"{context}\n\n用户问题：{user_content}"
        
        prompt_messages.append({
            "role": "user",
            "content": user_content
        })
        
        return prompt_messages
    
    async def _stream_llm_response(
        self, 
        messages: List[Dict[str, str]], 
        request: QueryRequest
    ) -> AsyncGenerator[str, None]:
        """调用大模型流式生成"""
        try:
            headers = {
                "Authorization": f"Bearer {config.ARK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config.ARK_MODEL,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True
            }
            
            # 发送请求
            response = requests.post(
                f"{config.ARK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                yield self._format_response("error", f"LLM API error: {response.status_code}")
                return
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 去掉 'data: ' 前缀
                        
                        if data.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    if content:
                                        yield self._format_response("delta", content)
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            yield self._format_response("error", str(e))
    
    def _format_response(
        self, 
        response_type: str, 
        content: str = None, 
        refs: List[Reference] = None
    ) -> str:
        """格式化响应"""
        response = QueryResponse(
            type=response_type,
            content=content,
            refs=refs
        )
        return json.dumps(response.dict(), ensure_ascii=False) + "\n"
