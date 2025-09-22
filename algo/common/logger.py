"""
VoiceHelper 日志系统 (Python版本)
提供结构化日志记录，包含网络信息、错误码等
"""

import json
import logging
import os
import socket
import time
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

from fastapi import Request
import uvicorn

from .errors import ErrorCode, VoiceHelperError


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogType(Enum):
    """日志类型"""
    STARTUP = "startup"
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    DEBUG = "debug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"


class NetworkInfo:
    """网络信息"""
    
    def __init__(self, request: Optional[Request] = None):
        if request:
            self.local_ip = self._get_local_ip()
            self.local_port = self._get_local_port(request)
            self.remote_ip = self._get_client_ip(request)
            self.remote_port = self._get_remote_port(request)
            self.url = str(request.url)
            self.method = request.method
            self.user_agent = request.headers.get("user-agent", "")
            self.request_id = request.headers.get("x-request-id", "")
        else:
            self.local_ip = self._get_local_ip()
            self.local_port = ""
            self.remote_ip = ""
            self.remote_port = ""
            self.url = ""
            self.method = ""
            self.user_agent = ""
            self.request_id = ""
    
    def _get_local_ip(self) -> str:
        """获取本地IP地址"""
        try:
            # 连接到外部地址来获取本地IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def _get_local_port(self, request: Request) -> str:
        """获取本地端口"""
        try:
            url_parts = urlparse(str(request.url))
            return str(url_parts.port) if url_parts.port else "8000"
        except Exception:
            return "8000"
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端真实IP"""
        # 检查X-Forwarded-For头
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        
        # 检查X-Real-IP头
        xri = request.headers.get("x-real-ip")
        if xri:
            return xri
        
        # 检查X-Client-IP头
        xci = request.headers.get("x-client-ip")
        if xci:
            return xci
        
        # 使用客户端地址
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return ""
    
    def _get_remote_port(self, request: Request) -> str:
        """获取远程端口"""
        try:
            if hasattr(request, "client") and request.client:
                return str(request.client.port)
        except Exception:
            pass
        return ""
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "local_ip": self.local_ip,
            "local_port": self.local_port,
            "remote_ip": self.remote_ip,
            "remote_port": self.remote_port,
            "url": self.url,
            "method": self.method,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
        }


class LogEntry:
    """日志条目"""
    
    def __init__(
        self,
        level: LogLevel,
        log_type: LogType,
        message: str,
        service: str = "voicehelper-algo",
        module: str = "",
        error_code: Optional[ErrorCode] = None,
        network: Optional[NetworkInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        status_code: Optional[int] = None,
        stack_trace: Optional[str] = None,
    ):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.level = level.value
        self.type = log_type.value
        self.service = service
        self.module = module
        self.message = message
        self.error_code = int(error_code) if error_code else None
        self.network = network.to_dict() if network else None
        self.context = context or {}
        self.duration_ms = duration_ms
        self.request_size = request_size
        self.response_size = response_size
        self.status_code = status_code
        self.stack_trace = stack_trace
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = {
            "timestamp": self.timestamp,
            "level": self.level,
            "type": self.type,
            "service": self.service,
            "message": self.message,
        }
        
        if self.module:
            data["module"] = self.module
        if self.error_code:
            data["error_code"] = self.error_code
        if self.network:
            data["network"] = self.network
        if self.context:
            data["context"] = self.context
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        if self.request_size is not None:
            data["request_size"] = self.request_size
        if self.response_size is not None:
            data["response_size"] = self.response_size
        if self.status_code is not None:
            data["status_code"] = self.status_code
        if self.stack_trace:
            data["stack_trace"] = self.stack_trace
        
        return data
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))


class VoiceHelperLogger:
    """VoiceHelper日志器"""
    
    def __init__(self, service: str = "voicehelper-algo", module: str = ""):
        self.service = service
        self.module = module
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志器"""
        self.logger = logging.getLogger(self.service)
        
        # 设置日志级别
        log_level = os.getenv("LOG_LEVEL", "info").upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 创建控制台处理器
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # 防止重复日志
        self.logger.propagate = False
    
    def _log(
        self,
        level: LogLevel,
        log_type: LogType,
        message: str,
        error_code: Optional[ErrorCode] = None,
        network: Optional[NetworkInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        status_code: Optional[int] = None,
        include_stack: bool = False,
    ):
        """记录日志"""
        stack_trace = None
        if include_stack or level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            stack_trace = traceback.format_exc() if traceback.format_exc().strip() != "NoneType: None" else None
        
        entry = LogEntry(
            level=level,
            log_type=log_type,
            message=message,
            service=self.service,
            module=self.module,
            error_code=error_code,
            network=network,
            context=context,
            duration_ms=duration_ms,
            request_size=request_size,
            response_size=response_size,
            status_code=status_code,
            stack_trace=stack_trace,
        )
        
        # 使用标准logging记录
        log_method = getattr(self.logger, level.value.lower())
        log_method(entry.to_json())
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self._log(LogLevel.DEBUG, LogType.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self._log(LogLevel.INFO, LogType.SYSTEM, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self._log(LogLevel.WARNING, LogType.SYSTEM, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self._log(LogLevel.ERROR, LogType.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self._log(LogLevel.CRITICAL, LogType.ERROR, message, **kwargs)
    
    def startup(self, message: str, **kwargs):
        """记录启动日志"""
        self._log(LogLevel.INFO, LogType.STARTUP, message, **kwargs)
    
    def request(self, request: Request, **kwargs):
        """记录请求日志"""
        network = NetworkInfo(request)
        context = kwargs.get("context", {})
        
        # 添加请求大小信息
        request_size = None
        if hasattr(request, "headers"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    request_size = int(content_length)
                except ValueError:
                    pass
        
        self._log(
            LogLevel.INFO,
            LogType.REQUEST,
            f"{request.method} {request.url.path}",
            network=network,
            context=context,
            request_size=request_size,
        )
    
    def response(
        self,
        request: Request,
        status_code: int,
        duration_ms: float,
        response_size: Optional[int] = None,
        **kwargs
    ):
        """记录响应日志"""
        network = NetworkInfo(request)
        
        # 根据状态码确定日志级别
        if status_code >= 500:
            level = LogLevel.ERROR
        elif status_code >= 400:
            level = LogLevel.WARNING
        else:
            level = LogLevel.INFO
        
        self._log(
            level,
            LogType.RESPONSE,
            f"{request.method} {request.url.path} - {status_code}",
            network=network,
            status_code=status_code,
            duration_ms=duration_ms,
            response_size=response_size,
            context=kwargs.get("context"),
        )
    
    def error_with_code(self, error_code: ErrorCode, message: str, **kwargs):
        """记录带错误码的错误日志"""
        self._log(LogLevel.ERROR, LogType.ERROR, message, error_code=error_code, **kwargs)
    
    def performance(self, operation: str, duration_ms: float, **kwargs):
        """记录性能日志"""
        self._log(
            LogLevel.INFO,
            LogType.PERFORMANCE,
            f"Performance: {operation}",
            duration_ms=duration_ms,
            **kwargs
        )
    
    def security(self, event: str, **kwargs):
        """记录安全日志"""
        self._log(LogLevel.WARNING, LogType.SECURITY, f"Security Event: {event}", **kwargs)
    
    def business(self, event: str, **kwargs):
        """记录业务日志"""
        self._log(LogLevel.INFO, LogType.BUSINESS, f"Business Event: {event}", **kwargs)
    
    def exception(self, message: str, exc: Exception, **kwargs):
        """记录异常日志"""
        context = kwargs.get("context", {})
        context["exception_type"] = type(exc).__name__
        context["exception_message"] = str(exc)
        
        error_code = None
        if isinstance(exc, VoiceHelperError):
            error_code = exc.code
        
        self._log(
            LogLevel.ERROR,
            LogType.ERROR,
            message,
            error_code=error_code,
            context=context,
            include_stack=True,
            **kwargs
        )
    
    def with_module(self, module: str) -> 'VoiceHelperLogger':
        """创建带模块名的日志器"""
        new_logger = VoiceHelperLogger(self.service, module)
        return new_logger
    
    def with_context(self, **context) -> 'VoiceHelperLogger':
        """创建带上下文的日志器"""
        # 这里可以实现上下文传递，简化版本直接返回自身
        return self


# 全局日志器实例
_default_logger: Optional[VoiceHelperLogger] = None


def init_logger(service: str = "voicehelper-algo"):
    """初始化全局日志器"""
    global _default_logger
    _default_logger = VoiceHelperLogger(service)


def get_logger(module: str = "") -> VoiceHelperLogger:
    """获取日志器"""
    if _default_logger is None:
        init_logger()
    
    if module:
        return _default_logger.with_module(module)
    return _default_logger


# 便利函数
def debug(message: str, **kwargs):
    """记录调试日志"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """记录信息日志"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """记录警告日志"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """记录错误日志"""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """记录严重错误日志"""
    get_logger().critical(message, **kwargs)


def startup(message: str, **kwargs):
    """记录启动日志"""
    get_logger().startup(message, **kwargs)


def error_with_code(error_code: ErrorCode, message: str, **kwargs):
    """记录带错误码的错误日志"""
    get_logger().error_with_code(error_code, message, **kwargs)


def performance(operation: str, duration_ms: float, **kwargs):
    """记录性能日志"""
    get_logger().performance(operation, duration_ms, **kwargs)


def security(event: str, **kwargs):
    """记录安全日志"""
    get_logger().security(event, **kwargs)


def business(event: str, **kwargs):
    """记录业务日志"""
    get_logger().business(event, **kwargs)


def exception(message: str, exc: Exception, **kwargs):
    """记录异常日志"""
    get_logger().exception(message, exc, **kwargs)


# FastAPI中间件
class LoggingMiddleware:
    """FastAPI日志中间件"""
    
    def __init__(self, logger: Optional[VoiceHelperLogger] = None):
        self.logger = logger or get_logger("middleware")
    
    async def __call__(self, request: Request, call_next):
        """处理请求和响应日志"""
        start_time = time.time()
        
        # 记录请求日志
        self.logger.request(request)
        
        # 处理请求
        try:
            response = await call_next(request)
            
            # 计算处理时间
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录响应日志
            response_size = None
            if hasattr(response, "headers"):
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        response_size = int(content_length)
                    except ValueError:
                        pass
            
            self.logger.response(
                request,
                response.status_code,
                duration_ms,
                response_size=response_size
            )
            
            return response
            
        except Exception as exc:
            # 记录异常日志
            duration_ms = (time.time() - start_time) * 1000
            self.logger.exception("Request processing failed", exc, context={
                "method": request.method,
                "url": str(request.url),
                "duration_ms": duration_ms,
            })
            raise


# Uvicorn日志配置
def get_uvicorn_log_config() -> Dict[str, Any]:
    """获取Uvicorn日志配置"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": os.getenv("LOG_LEVEL", "INFO").upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": os.getenv("LOG_LEVEL", "INFO").upper(),
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": os.getenv("LOG_LEVEL", "INFO").upper(),
                "propagate": False,
            },
        },
    }
