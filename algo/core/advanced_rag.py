"""
高级RAG系统 - v1.4.0
实现HyDE、重排序、多路召回、查询改写等高级特性
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from loguru import logger

from core.embeddings import EmbeddingService
from core.llm import LLMService
# from pymilvus import Collection  # 已移除 Milvus 支持


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    retrieval_method: str  # dense, sparse, hybrid, hyde


class HyDEGenerator:
    """假设性文档嵌入(Hypothetical Document Embeddings)生成器"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.templates = {
            "qa": """
            问题：{query}
            
            请生成一个详细的、准确的答案文档，包含以下要素：
            1. 直接回答问题
            2. 提供相关背景信息
            3. 包含具体的事实和数据
            4. 使用专业术语
            
            生成的文档：
            """,
            "definition": """
            查询：{query}
            
            请生成一个类似百科全书的定义文档，包含：
            1. 标准定义
            2. 详细解释
            3. 相关概念
            4. 实际应用
            
            定义文档：
            """,
            "tutorial": """
            主题：{query}
            
            请生成一个教程文档，包含：
            1. 概述
            2. 步骤说明
            3. 注意事项
            4. 最佳实践
            
            教程文档：
            """
        }
    
    async def generate_hypothetical_documents(
        self,
        query: str,
        num_documents: int = 3,
        doc_type: str = "qa"
    ) -> List[str]:
        """生成假设性文档"""
        template = self.templates.get(doc_type, self.templates["qa"])
        prompt = template.format(query=query)
        
        documents = []
        for i in range(num_documents):
            # 添加变化以生成不同的文档
            varied_prompt = f"{prompt}\n\n版本 {i+1}（从不同角度回答）："
            
            response = await self.llm.generate(
                varied_prompt,
                temperature=0.7,  # 增加多样性
                max_tokens=300
            )
            
            documents.append(response)
            
        logger.info(f"生成了 {len(documents)} 个假设性文档")
        return documents


class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.rewrite_strategies = {
            "expansion": self._expand_query,
            "decomposition": self._decompose_query,
            "clarification": self._clarify_query,
            "translation": self._translate_concepts
        }
    
    async def rewrite_query(
        self,
        query: str,
        strategy: str = "expansion",
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """改写查询"""
        rewrite_func = self.rewrite_strategies.get(strategy, self._expand_query)
        return await rewrite_func(query, context)
    
    async def _expand_query(self, query: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """扩展查询"""
        prompt = f"""
        原始查询：{query}
        
        请扩展这个查询，生成3个相关但更具体的查询变体：
        1. 添加相关关键词
        2. 使用同义词
        3. 包含上下文信息
        
        输出格式（每行一个查询）：
        """
        
        response = await self.llm.generate(prompt)
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        queries.insert(0, query)  # 保留原始查询
        
        return queries[:4]  # 返回最多4个查询
    
    async def _decompose_query(self, query: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """分解查询"""
        prompt = f"""
        复杂查询：{query}
        
        将这个查询分解为多个简单的子查询，每个子查询关注一个方面：
        """
        
        response = await self.llm.generate(prompt)
        sub_queries = [q.strip() for q in response.split('\n') if q.strip()]
        
        return sub_queries
    
    async def _clarify_query(self, query: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """澄清查询"""
        prompt = f"""
        模糊查询：{query}
        上下文：{context or '无'}
        
        这个查询可能有多种理解方式，请生成更明确的查询版本：
        """
        
        response = await self.llm.generate(prompt)
        clarified = [q.strip() for q in response.split('\n') if q.strip()]
        
        return clarified
    
    async def _translate_concepts(self, query: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """概念转换"""
        prompt = f"""
        查询：{query}
        
        将查询中的通俗用语转换为专业术语，或反之：
        """
        
        response = await self.llm.generate(prompt)
        translated = [q.strip() for q in response.split('\n') if q.strip()]
        
        return translated


class CrossEncoder:
    """交叉编码器用于重排序"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # 如果有GPU则使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """重排序文档"""
        scores = []
        
        with torch.no_grad():
            for doc in documents:
                # 编码查询-文档对
                inputs = self.tokenizer(
                    query,
                    doc,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取相关性分数
                outputs = self.model(**inputs)
                score = outputs.logits[0].cpu().numpy()
                scores.append(float(score))
        
        # 排序并返回top_k
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_scores = [scores[i] for i in ranked_indices]
        
        return list(zip(ranked_indices.tolist(), ranked_scores))


class MultiPathRetriever:
    """多路径检索器"""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        # milvus_collection: Collection,  # 已移除 Milvus 支持
        sparse_index=None  # BM25或其他稀疏检索索引
    ):
        self.embedding_service = embedding_service
        # self.collection = milvus_collection  # 已移除 Milvus 支持
        self.sparse_index = sparse_index
    
    async def retrieve_dense(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """密集向量检索"""
        # 获取查询向量
        query_vector = await self.embedding_service.embed_text(query)
        
        # 本地向量检索（替代 Milvus）
        logger.warning("密集向量检索功能需要实现本地向量存储")
        return []
    
    async def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """稀疏检索（BM25等）"""
        if not self.sparse_index:
            return []
        
        # 使用BM25或其他稀疏检索方法
        results = self.sparse_index.search(query, top_k)
        
        retrieval_results = []
        for doc_id, score in results:
            # 从数据库获取文档内容
            doc = self._get_document(doc_id)
            if doc:
                retrieval_results.append(RetrievalResult(
                    chunk_id=doc["chunk_id"],
                    content=doc["content"],
                    score=score,
                    source=doc["source"],
                    metadata=doc.get("metadata", {}),
                    retrieval_method="sparse"
                ))
        
        return retrieval_results
    
    async def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.7
    ) -> List[RetrievalResult]:
        """混合检索"""
        # 并行执行密集和稀疏检索
        dense_task = self.retrieve_dense(query, top_k * 2)
        sparse_task = self.retrieve_sparse(query, top_k * 2)
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # 合并和重新评分
        combined_results = {}
        
        # 处理密集检索结果
        for result in dense_results:
            combined_results[result.chunk_id] = {
                "result": result,
                "dense_score": result.score * dense_weight,
                "sparse_score": 0
            }
        
        # 处理稀疏检索结果
        sparse_weight = 1 - dense_weight
        for result in sparse_results:
            if result.chunk_id in combined_results:
                combined_results[result.chunk_id]["sparse_score"] = result.score * sparse_weight
            else:
                combined_results[result.chunk_id] = {
                    "result": result,
                    "dense_score": 0,
                    "sparse_score": result.score * sparse_weight
                }
        
        # 计算最终分数并排序
        final_results = []
        for chunk_id, scores in combined_results.items():
            result = scores["result"]
            result.score = scores["dense_score"] + scores["sparse_score"]
            result.retrieval_method = "hybrid"
            final_results.append(result)
        
        # 排序并返回top_k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
    
    def _get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """从数据库获取文档"""
        # 实现文档获取逻辑
        pass


class AdvancedRAG:
    """高级RAG系统"""
    
    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        # milvus_collection: Collection  # 已移除 Milvus 支持
    ):
        self.llm = llm_service
        self.embedding_service = embedding_service
        # self.collection = milvus_collection  # 已移除 Milvus 支持
        
        # 初始化组件
        self.hyde_generator = HyDEGenerator(llm_service)
        self.query_rewriter = QueryRewriter(llm_service)
        self.cross_encoder = CrossEncoder()
        # self.multi_retriever = MultiPathRetriever(
        #     embedding_service,
        #     milvus_collection  # 已移除 Milvus 支持
        # )
        
        logger.info("高级RAG系统初始化完成")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_rewrite: bool = True,
        use_rerank: bool = True,
        retrieval_method: str = "hybrid"
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """执行高级检索"""
        metadata = {
            "original_query": query,
            "techniques_used": []
        }
        
        all_results = []
        
        # 1. 查询改写
        queries = [query]
        if use_rewrite:
            rewritten_queries = await self.query_rewriter.rewrite_query(query)
            queries.extend(rewritten_queries)
            metadata["techniques_used"].append("query_rewriting")
            metadata["rewritten_queries"] = rewritten_queries
        
        # 2. HyDE
        if use_hyde:
            hypothetical_docs = await self.hyde_generator.generate_hypothetical_documents(query)
            metadata["techniques_used"].append("hyde")
            metadata["hypothetical_docs_count"] = len(hypothetical_docs)
            
            # 为假设性文档生成向量并检索
            for doc in hypothetical_docs:
                if retrieval_method == "dense":
                    results = await self.multi_retriever.retrieve_dense(doc, top_k * 2)
                elif retrieval_method == "sparse":
                    results = await self.multi_retriever.retrieve_sparse(doc, top_k * 2)
                else:  # hybrid
                    results = await self.multi_retriever.retrieve_hybrid(doc, top_k * 2)
                
                all_results.extend(results)
        
        # 3. 多查询检索
        for q in queries:
            if retrieval_method == "dense":
                results = await self.multi_retriever.retrieve_dense(q, top_k * 2)
            elif retrieval_method == "sparse":
                results = await self.multi_retriever.retrieve_sparse(q, top_k * 2)
            else:  # hybrid
                results = await self.multi_retriever.retrieve_hybrid(q, top_k * 2)
            
            all_results.extend(results)
        
        # 4. 去重
        unique_results = {}
        for result in all_results:
            if result.chunk_id not in unique_results:
                unique_results[result.chunk_id] = result
            else:
                # 保留分数更高的
                if result.score > unique_results[result.chunk_id].score:
                    unique_results[result.chunk_id] = result
        
        results_list = list(unique_results.values())
        
        # 5. 重排序
        if use_rerank and len(results_list) > 0:
            metadata["techniques_used"].append("reranking")
            
            # 准备文档内容
            documents = [r.content for r in results_list]
            
            # 使用交叉编码器重排序
            reranked_indices = self.cross_encoder.rerank(query, documents, top_k)
            
            # 重新排列结果
            reranked_results = []
            for idx, score in reranked_indices:
                result = results_list[idx]
                result.score = score  # 使用重排序分数
                reranked_results.append(result)
            
            results_list = reranked_results
        else:
            # 简单排序
            results_list.sort(key=lambda x: x.score, reverse=True)
            results_list = results_list[:top_k]
        
        metadata["total_retrieved"] = len(all_results)
        metadata["unique_retrieved"] = len(unique_results)
        metadata["final_count"] = len(results_list)
        
        return results_list, metadata
    
    async def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """基于检索结果生成答案"""
        # 构建上下文
        context_parts = []
        references = []
        
        for i, result in enumerate(retrieval_results):
            context_parts.append(f"[文档{i+1}] {result.content}")
            references.append({
                "chunk_id": result.chunk_id,
                "source": result.source,
                "score": result.score,
                "retrieval_method": result.retrieval_method
            })
        
        context = "\n\n".join(context_parts)
        
        # 构建提示
        prompt = f"""
        基于以下检索到的文档回答用户问题。
        
        检索文档：
        {context}
        
        用户问题：{query}
        
        请提供准确、全面的回答，并在回答中引用相关文档编号。
        """
        
        # 添加会话历史
        if conversation_history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-5:]  # 最近5轮对话
            ])
            prompt = f"对话历史：\n{history_text}\n\n{prompt}"
        
        # 生成回答
        answer = await self.llm.generate(prompt)
        
        return answer, references
    
    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        ground_truth: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """评估检索质量"""
        metrics = {}
        
        # 1. 相关性评分
        relevance_prompt = f"""
        查询：{query}
        
        请评估以下文档与查询的相关性（0-1分）：
        {retrieved_docs[0].content if retrieved_docs else '无'}
        
        只输出分数：
        """
        
        relevance_score = await self.llm.generate(relevance_prompt)
        try:
            metrics["relevance"] = float(relevance_score.strip())
        except:
            metrics["relevance"] = 0.0
        
        # 2. 覆盖度评分
        if ground_truth:
            # 计算召回率
            retrieved_ids = set(r.chunk_id for r in retrieved_docs)
            truth_ids = set(ground_truth)
            recall = len(retrieved_ids & truth_ids) / len(truth_ids) if truth_ids else 0
            metrics["recall"] = recall
        
        # 3. 多样性评分
        if len(retrieved_docs) > 1:
            # 计算文档之间的平均相似度
            embeddings = []
            for doc in retrieved_docs[:5]:
                embedding = await self.embedding_service.embed_text(doc.content)
                embeddings.append(embedding)
            
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            metrics["diversity"] = 1 - avg_similarity  # 多样性是相似度的反向
        else:
            metrics["diversity"] = 0.0
        
        return metrics
