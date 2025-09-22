import json
import asyncio
import pickle
import numpy as np
from typing import AsyncGenerator, List, Dict, Any
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

from core.config import config
from core.embeddings import get_embeddings
from core.models import QueryRequest, QueryResponse, Reference, Message

class RetrieveService:
    def __init__(self):
        self.embeddings = get_embeddings()
        # 使用本地向量存储替代 Milvus
        self.vector_store = {}
        self.documents_store = {}
        self.storage_path = Path("/app/data")
        self.storage_path.mkdir(exist_ok=True)
        self._load_local_storage()
    
    def _load_local_storage(self):
        """加载本地存储的向量数据"""
        try:
            vector_file = self.storage_path / "vectors.pkl"
            docs_file = self.storage_path / "documents.pkl"
            
            if vector_file.exists():
                with open(vector_file, 'rb') as f:
                    self.vector_store = pickle.load(f)
                print(f"Loaded {len(self.vector_store)} vectors from local storage")
            
            if docs_file.exists():
                with open(docs_file, 'rb') as f:
                    self.documents_store = pickle.load(f)
                print(f"Loaded {len(self.documents_store)} documents from local storage")
                    
        except Exception as e:
            print(f"Failed to load local storage: {e}")
            self.vector_store = {}
            self.documents_store = {}
    
    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """使用本地向量存储进行相似度搜索"""
        if not self.vector_store:
            print("No vectors in local storage")
            return []
        
        # 获取查询向量
        query_vector = self.embeddings.embed_query(query)
        
        # 计算相似度
        similarities = []
        for chunk_id, data in self.vector_store.items():
            similarity = self._cosine_similarity(query_vector, data["vector"])
            similarities.append((chunk_id, similarity, data))
        
        # 按相似度排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score, data in similarities[:k]:
            # 创建类似 langchain Document 的对象
            class MockDocument:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata
            
            doc = MockDocument(
                page_content=data["text"],
                metadata=data["metadata"]
            )
            results.append((doc, score))
        
        return results
    
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
            results = self.similarity_search_with_score(
                query=query,
                k=top_k
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
            # 使用多模型服务
            from core.multi_model_config import MultiModelConfig
            from core.multi_model_service import MultiModelService
            
            multi_config = MultiModelConfig()
            multi_service = MultiModelService(multi_config)
            
            # 获取首选模型
            preferred_model = getattr(request, 'model', None) or config.PRIMARY_MODEL
            
            # 调用多模型服务进行流式生成
            async for chunk_type, chunk_content in multi_service.stream_model(
                messages=messages,
                preferred_model=preferred_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                if chunk_type == "error":
                    yield self._format_response("error", f"LLM API error: {chunk_content}")
                    return
                elif chunk_type == "content":
                    yield self._format_response("content", chunk_content)
                elif chunk_type == "done":
                    break
            
            return
            
        except Exception as e:
            logger.error(f"LLM调用异常: {str(e)}")
            yield self._format_response("error", f"LLM调用异常: {str(e)}")
        
        # 发送结束标记
        yield self._format_response("end", None)
    
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
