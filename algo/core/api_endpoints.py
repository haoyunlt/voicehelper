"""
BGE + FAISS RAG API 端点
提供文档入库、检索、指标查询等功能
"""

from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from pydantic import BaseModel
from loguru import logger

from core.rag_factory import get_rag_factory
from core.rag.metrics import get_metrics_collector


class DocumentInput(BaseModel):
    """文档输入模型"""
    id: str
    title: Optional[str] = ""
    content: str
    source: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}


class IngestRequest(BaseModel):
    """入库请求模型"""
    documents: List[DocumentInput]
    tenant_id: str = "default"
    dataset_id: str = "default"


class RetrieveRequest(BaseModel):
    """检索请求模型"""
    query: str
    tenant_id: str = "default"
    dataset_id: str = "default"
    top_k: int = 5
    score_threshold: float = 0.0


class RAGEndpoints:
    """RAG API端点类"""
    
    def __init__(self):
        self.factory = get_rag_factory()
        self.metrics_collector = get_metrics_collector()
    
    async def ingest_documents(self, request: IngestRequest) -> Dict[str, Any]:
        """文档入库"""
        try:
            logger.info(f"开始文档入库: 租户={request.tenant_id}, 数据集={request.dataset_id}, 文档数={len(request.documents)}")
            
            # 转换文档格式
            documents = [doc.dict() for doc in request.documents]
            
            # 构建索引
            result = await self.factory.build_index_from_documents(
                documents=documents,
                tenant_id=request.tenant_id,
                dataset_id=request.dataset_id
            )
            
            logger.info(f"文档入库完成: {result}")
            return {
                "status": "success",
                "message": "Documents ingested successfully",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"文档入库失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to ingest documents: {str(e)}")
    
    async def retrieve_documents(self, request: RetrieveRequest) -> Dict[str, Any]:
        """文档检索"""
        try:
            logger.info(f"开始文档检索: 查询={request.query[:50]}..., 租户={request.tenant_id}, 数据集={request.dataset_id}")
            
            # 获取检索器
            retriever = self.factory.create_retriever(
                tenant_id=request.tenant_id,
                dataset_id=request.dataset_id
            )
            
            # 执行检索
            results = retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )
            
            logger.info(f"文档检索完成: 结果数={len(results)}")
            return {
                "status": "success",
                "message": "Documents retrieved successfully",
                "data": {
                    "query": request.query,
                    "results": results,
                    "total_count": len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")
    
    def get_retrieval_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """获取检索指标"""
        try:
            stats = self.metrics_collector.get_retrieval_stats(window_minutes=window_minutes)
            return {
                "status": "success",
                "message": "Retrieval metrics retrieved successfully",
                "data": stats
            }
        except Exception as e:
            logger.error(f"获取检索指标失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get retrieval metrics: {str(e)}")
    
    def get_index_metrics(self) -> Dict[str, Any]:
        """获取索引指标"""
        try:
            stats = self.metrics_collector.get_index_stats()
            return {
                "status": "success",
                "message": "Index metrics retrieved successfully", 
                "data": stats
            }
        except Exception as e:
            logger.error(f"获取索引指标失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get index metrics: {str(e)}")
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """获取缓存指标"""
        try:
            stats = self.metrics_collector.get_cache_stats()
            return {
                "status": "success",
                "message": "Cache metrics retrieved successfully",
                "data": stats
            }
        except Exception as e:
            logger.error(f"获取缓存指标失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get cache metrics: {str(e)}")
    
    def get_retriever_stats(self, tenant_id: str = "default", dataset_id: str = "default") -> Dict[str, Any]:
        """获取检索器统计"""
        try:
            stats = self.factory.get_retriever_stats(
                tenant_id=tenant_id,
                dataset_id=dataset_id
            )
            return {
                "status": "success",
                "message": "Retriever stats retrieved successfully",
                "data": stats
            }
        except Exception as e:
            logger.error(f"获取检索器统计失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get retriever stats: {str(e)}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        try:
            all_metrics = self.metrics_collector.export_metrics()
            return {
                "status": "success",
                "message": "All metrics retrieved successfully",
                "data": all_metrics
            }
        except Exception as e:
            logger.error(f"获取所有指标失败: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get all metrics: {str(e)}")


# 全局端点实例
rag_endpoints = RAGEndpoints()
