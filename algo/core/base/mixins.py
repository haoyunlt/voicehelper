"""
V2架构Mixin组件
提供重试、可观测、缓存等横切能力
"""

from typing import Callable, Any, Optional, Dict, Tuple
import logging
import time
import hashlib
from functools import wraps

from .protocols import StreamCallback


class RetryableMixin:
    """重试能力 Mixin"""
    
    max_retries: int = 1
    retry_delay: float = 0.1
    
    def _retry(self, fn: Callable[[], Any]) -> Any:
        """
        重试封装
        
        Args:
            fn: 要重试的函数
            
        Returns:
            函数执行结果
            
        Raises:
            最后一次执行的异常
        """
        for i in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if i == self.max_retries:
                    raise
                time.sleep(self.retry_delay * (2 ** i))


class ObservableMixin:
    """可观测能力 Mixin"""
    
    def emit(self, cb: Optional[StreamCallback], event: str, payload: dict):
        """
        发送事件
        
        Args:
            cb: 回调函数
            event: 事件类型
            payload: 事件数据
        """
        if cb:
            cb(event, payload)
    
    def _log_performance(self, operation: str):
        """
        性能日志装饰器
        
        Args:
            operation: 操作名称
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logging.info(f"{operation} completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logging.error(f"{operation} failed after {duration:.3f}s: {e}")
                    raise
            return wrapper
        return decorator


class CacheableMixin:
    """缓存能力 Mixin"""
    
    _cache: Dict[str, Tuple[float, Any]] = {}
    cache_ttl: int = 300  # 5分钟
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键
        """
        key_str = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cached_call(self, fn: Callable, *args, **kwargs):
        """
        缓存调用
        
        Args:
            fn: 要缓存的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        cache_key = self._get_cache_key(*args, **kwargs)
        now = time.time()
        
        if cache_key in self._cache:
            cached_time, result = self._cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return result
        
        result = fn(*args, **kwargs)
        self._cache[cache_key] = (now, result)
        return result
