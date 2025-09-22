"""
VoiceHelper v1.24.0 - 错误处理系统
实现完善的异常处理和降级机制，提升系统稳定性
"""

import asyncio
import time
import logging
import json
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import functools
import inspect
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误类别"""
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL_SERVICE = "internal_service"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

class FallbackStrategy(Enum):
    """降级策略"""
    CACHE_ONLY = "cache_only"
    SIMPLIFIED_RESPONSE = "simplified_response"
    DEFAULT_VALUE = "default_value"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY = "retry"
    SKIP = "skip"

@dataclass
class ErrorEvent:
    """错误事件"""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: str = ""
    function_name: str = ""

@dataclass
class FallbackConfig:
    """降级配置"""
    strategy: FallbackStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    cache_ttl: float = 300.0
    default_value: Any = None
    fallback_function: Optional[Callable] = None

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    async def call(self, func: Callable, *args, **kwargs):
        """调用函数并处理熔断"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

class RetryHandler:
    """重试处理器"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """执行函数并重试"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.error_history = deque(maxlen=10000)
        
    def register_error_handler(self, 
                             category: ErrorCategory, 
                             handler: Callable):
        """注册错误处理器"""
        self.error_handlers[category].append(handler)
        logger.info(f"Registered error handler for category: {category.value}")
    
    def register_fallback_config(self, 
                               service_name: str, 
                               config: FallbackConfig):
        """注册降级配置"""
        self.fallback_configs[service_name] = config
        
        # 创建熔断器
        if config.strategy == FallbackStrategy.CIRCUIT_BREAKER:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                recovery_timeout=config.circuit_breaker_timeout
            )
        
        # 创建重试处理器
        if config.strategy == FallbackStrategy.RETRY:
            self.retry_handlers[service_name] = RetryHandler(
                max_retries=config.max_retries,
                base_delay=config.retry_delay
            )
        
        logger.info(f"Registered fallback config for service: {service_name}")
    
    async def handle_error(self, 
                          error: ErrorEvent,
                          service_name: str = None) -> Any:
        """处理错误"""
        # 记录错误
        self.error_history.append(error)
        
        # 调用特定类别的错误处理器
        if error.category in self.error_handlers:
            for handler in self.error_handlers[error.category]:
                try:
                    await handler(error)
                except Exception as e:
                    logger.error(f"Error handler failed: {e}")
        
        # 应用降级策略
        if service_name and service_name in self.fallback_configs:
            return await self._apply_fallback(error, service_name)
        
        return None
    
    async def _apply_fallback(self, error: ErrorEvent, service_name: str) -> Any:
        """应用降级策略"""
        config = self.fallback_configs[service_name]
        
        try:
            if config.strategy == FallbackStrategy.CACHE_ONLY:
                return await self._cache_only_fallback(service_name)
            
            elif config.strategy == FallbackStrategy.SIMPLIFIED_RESPONSE:
                return await self._simplified_response_fallback(service_name)
            
            elif config.strategy == FallbackStrategy.DEFAULT_VALUE:
                return config.default_value
            
            elif config.strategy == FallbackStrategy.CIRCUIT_BREAKER:
                return await self._circuit_breaker_fallback(service_name)
            
            elif config.strategy == FallbackStrategy.RETRY:
                return await self._retry_fallback(service_name)
            
            elif config.strategy == FallbackStrategy.SKIP:
                return None
            
            else:
                logger.warning(f"Unknown fallback strategy: {config.strategy}")
                return None
                
        except Exception as e:
            logger.error(f"Fallback strategy failed: {e}")
            return config.default_value
    
    async def _cache_only_fallback(self, service_name: str) -> Any:
        """缓存降级"""
        # 实现缓存降级逻辑
        logger.info(f"Applying cache-only fallback for {service_name}")
        return None
    
    async def _simplified_response_fallback(self, service_name: str) -> Any:
        """简化响应降级"""
        # 实现简化响应降级逻辑
        logger.info(f"Applying simplified response fallback for {service_name}")
        return {"status": "degraded", "service": service_name}
    
    async def _circuit_breaker_fallback(self, service_name: str) -> Any:
        """熔断器降级"""
        logger.info(f"Circuit breaker activated for {service_name}")
        return {"status": "circuit_breaker_open", "service": service_name}
    
    async def _retry_fallback(self, service_name: str) -> Any:
        """重试降级"""
        logger.info(f"Retry fallback applied for {service_name}")
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = list(self.error_history)[-1000:]  # 最近1000个错误
        
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in recent_errors:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1
        
        return {
            "total_errors": len(recent_errors),
            "category_distribution": dict(category_counts),
            "severity_distribution": dict(severity_counts),
            "error_rate": len(recent_errors) / 1000.0,  # 错误率
            "timestamp": time.time()
        }

def error_handler(service_name: str = None, 
                 fallback_config: FallbackConfig = None,
                 log_errors: bool = True):
    """错误处理装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler_instance = get_error_handler()
            
            try:
                # 如果有熔断器，使用熔断器调用
                if service_name and service_name in error_handler_instance.circuit_breakers:
                    circuit_breaker = error_handler_instance.circuit_breakers[service_name]
                    return await circuit_breaker.call(func, *args, **kwargs)
                
                # 如果有重试配置，使用重试调用
                elif service_name and service_name in error_handler_instance.retry_handlers:
                    retry_handler = error_handler_instance.retry_handlers[service_name]
                    return await retry_handler.execute_with_retry(func, *args, **kwargs)
                
                # 普通调用
                else:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
            except Exception as e:
                # 创建错误事件
                error_event = ErrorEvent(
                    error_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.INTERNAL_SERVICE,
                    message=str(e),
                    exception=e,
                    stack_trace=traceback.format_exc(),
                    service_name=service_name or "unknown",
                    function_name=func.__name__,
                    context={
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
                
                # 记录错误
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                
                # 处理错误
                fallback_result = await error_handler_instance.handle_error(
                    error_event, service_name
                )
                
                if fallback_result is not None:
                    return fallback_result
                
                # 重新抛出异常
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同步函数的异步包装
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

@asynccontextmanager
async def error_context(service_name: str, 
                       operation_name: str,
                       user_id: str = None,
                       session_id: str = None):
    """错误上下文管理器"""
    request_id = str(uuid.uuid4())
    error_handler_instance = get_error_handler()
    
    try:
        yield {
            "request_id": request_id,
            "service_name": service_name,
            "operation_name": operation_name,
            "user_id": user_id,
            "session_id": session_id
        }
    except Exception as e:
        # 创建错误事件
        error_event = ErrorEvent(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INTERNAL_SERVICE,
            message=str(e),
            exception=e,
            stack_trace=traceback.format_exc(),
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            service_name=service_name,
            function_name=operation_name
        )
        
        # 处理错误
        await error_handler_instance.handle_error(error_event, service_name)
        raise e

class ErrorHandlingSystem:
    """错误处理系统"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.performance_metrics = {
            "total_errors": 0,
            "handled_errors": 0,
            "fallback_activations": 0,
            "circuit_breaker_trips": 0
        }
        
        # 注册默认错误处理器
        self._register_default_handlers()
        
        logger.info("Error handling system initialized")
    
    def _register_default_handlers(self):
        """注册默认错误处理器"""
        # 网络错误处理器
        self.error_handler.register_error_handler(
            ErrorCategory.NETWORK,
            self._handle_network_error
        )
        
        # 数据库错误处理器
        self.error_handler.register_error_handler(
            ErrorCategory.DATABASE,
            self._handle_database_error
        )
        
        # 认证错误处理器
        self.error_handler.register_error_handler(
            ErrorCategory.AUTHENTICATION,
            self._handle_auth_error
        )
    
    async def _handle_network_error(self, error: ErrorEvent):
        """处理网络错误"""
        logger.warning(f"Network error: {error.message}")
        # 实现网络错误处理逻辑
    
    async def _handle_database_error(self, error: ErrorEvent):
        """处理数据库错误"""
        logger.error(f"Database error: {error.message}")
        # 实现数据库错误处理逻辑
    
    async def _handle_auth_error(self, error: ErrorEvent):
        """处理认证错误"""
        logger.warning(f"Authentication error: {error.message}")
        # 实现认证错误处理逻辑
    
    def register_service_fallback(self, 
                                service_name: str,
                                strategy: FallbackStrategy,
                                **kwargs):
        """注册服务降级配置"""
        config = FallbackConfig(strategy=strategy, **kwargs)
        self.error_handler.register_fallback_config(service_name, config)
        logger.info(f"Registered fallback for service: {service_name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        return self.error_handler.get_error_statistics()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.performance_metrics,
            "error_statistics": self.get_error_statistics()
        }

# 全局实例
_error_handling_system = None

def get_error_handler() -> ErrorHandler:
    """获取错误处理器实例"""
    global _error_handling_system
    if _error_handling_system is None:
        _error_handling_system = ErrorHandlingSystem()
    return _error_handling_system.error_handler

def get_error_handling_system() -> ErrorHandlingSystem:
    """获取错误处理系统实例"""
    global _error_handling_system
    if _error_handling_system is None:
        _error_handling_system = ErrorHandlingSystem()
    return _error_handling_system

# 使用示例
if __name__ == "__main__":
    async def test_error_handling():
        """测试错误处理"""
        error_system = get_error_handling_system()
        
        # 注册服务降级配置
        error_system.register_service_fallback(
            "emotion_service",
            FallbackStrategy.DEFAULT_VALUE,
            default_value={"emotion": "neutral", "confidence": 0.5}
        )
        
        # 测试错误处理装饰器
        @error_handler("test_service")
        async def test_function():
            raise ValueError("Test error")
        
        try:
            await test_function()
        except ValueError as e:
            print(f"Caught error: {e}")
        
        # 获取错误统计
        stats = error_system.get_error_statistics()
        print(f"Error statistics: {stats}")
    
    # 运行测试
    asyncio.run(test_error_handling())
