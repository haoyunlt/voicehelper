"""
V2架构核心父类
定义BaseTool、BaseRetriever、BaseAgentGraph等核心抽象类
"""

from typing import Iterator, Optional, Any, Dict, List
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

from .mixins import RetryableMixin, ObservableMixin, CacheableMixin
from .protocols import StreamCallback


class BaseTool(BaseModel, RetryableMixin, ObservableMixin):
    """工具基类"""
    
    name: str
    description: str
    args_schema: dict = Field(default_factory=dict)  # JSONSchema
    
    class Config:
        arbitrary_types_allowed = True
    
    def validate_args(self, **kwargs) -> dict:
        """
        参数校验
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            验证后的参数
        """
        # TODO: 基于 args_schema 进行 JSONSchema 校验
        return kwargs
    
    @abstractmethod
    def run(self, **kwargs) -> dict:
        """
        执行工具 - 子类必须实现
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        raise NotImplementedError
    
    def run_with_callback(self, cb: Optional[StreamCallback] = None, **kwargs) -> dict:
        """
        带回调的执行
        
        Args:
            cb: 回调函数
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        self.emit(cb, "tool_start", {"tool": self.name, "args": kwargs})
        try:
            result = self._retry(lambda: self.run(**kwargs))
            self.emit(cb, "tool_result", {"tool": self.name, "result": result})
            return result
        except Exception as e:
            self.emit(cb, "tool_error", {"tool": self.name, "error": str(e)})
            raise


class BaseRetriever(BaseModel, RetryableMixin, ObservableMixin, CacheableMixin):
    """检索器基类"""
    
    top_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        检索 - 子类必须实现
        
        Args:
            query: 查询文本
            **kwargs: 检索参数
            
        Returns:
            检索结果列表
        """
        raise NotImplementedError
    
    def retrieve_with_callback(self, query: str, cb: Optional[StreamCallback] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        带回调的检索
        
        Args:
            query: 查询文本
            cb: 回调函数
            **kwargs: 检索参数
            
        Returns:
            检索结果列表
        """
        self.emit(cb, "retrieve_start", {"query": query})
        try:
            # 尝试从缓存获取
            result = self._cached_call(self.retrieve, query, **kwargs)
            self.emit(cb, "retrieve_result", {"query": query, "count": len(result)})
            return result
        except Exception as e:
            self.emit(cb, "retrieve_error", {"query": query, "error": str(e)})
            raise


class BaseAgentGraph(BaseModel, ObservableMixin):
    """Agent 图基类"""
    
    retriever: BaseRetriever
    tools: List[BaseTool] = []
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def stream(self, query: str, *, cb: Optional[StreamCallback] = None) -> Iterator[dict]:
        """
        统一流式接口 - 子类必须实现
        
        Args:
            query: 查询文本
            cb: 回调函数
            
        Yields:
            流式结果
        """
        raise NotImplementedError
    
    def _get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """
        根据名称获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例或None
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
