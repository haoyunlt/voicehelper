"""
VoiceHelper Python SDK 日志系统
提供结构化日志记录，包含SDK使用信息和性能指标
"""

import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

from .errors import ErrorCode, VoiceHelperSDKError, get_error_info


class LogLevel:
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogType:
    """日志类型"""
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    NETWORK = "network"
    FILE_OPERATION = "file_operation"
    SERIALIZATION = "serialization"
    AUTHENTICATION = "authentication"
    ERROR = "error"
    DEBUG = "debug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"


class SDKInfo:
    """SDK信息"""
    
    def __init__(self):
        self.name = "voicehelper-python-sdk"
        self.version = self._get_sdk_version()
        self.python_version = sys.version
        self.platform = platform.platform()
        self.architecture = platform.architecture()[0]
        self.machine = platform.machine()
        self.processor = platform.processor()
        self.system = platform.system()
        self.release = platform.release()
        self.user_agent = f"{self.name}/{self.version} Python/{platform.python_version()} {self.system}/{self.release}"
    
    def _get_sdk_version(self) -> str:
        """获取SDK版本"""
        try:
            # 尝试从包信息获取版本
            import pkg_resources
            return pkg_resources.get_distribution("voicehelper-sdk").version
        except:
            # 如果获取失败，返回默认版本
            return "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "python_version": self.python_version,
            "platform": self.platform,
            "architecture": self.architecture,
            "machine": self.machine,
            "processor": self.processor,
            "system": self.system,
            "release": self.release,
            "user_agent": self.user_agent
        }


class NetworkInfo:
    """网络信息"""
    
    def __init__(self, url: Optional[str] = None, method: Optional[str] = None):
        self.url = url
        self.method = method
        self.host = None
        self.port = None
        self.scheme = None
        
        if url:
            parsed = urlparse(url)
            self.host = parsed.hostname
            self.port = parsed.port
            self.scheme = parsed.scheme
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "url": self.url,
            "method": self.method,
            "host": self.host,
            "port": self.port,
            "scheme": self.scheme
        }


class PerformanceInfo:
    """性能信息"""
    
    def __init__(self):
        self.memory_usage = self._get_memory_usage()
        self.cpu_count = os.cpu_count()
        self.process_id = os.getpid()
    
    def _get_memory_usage(self) -> Optional[Dict[str, Any]]:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,  # 物理内存
                "vms": memory_info.vms,  # 虚拟内存
                "percent": process.memory_percent()  # 内存使用百分比
            }
        except ImportError:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "memory_usage": self.memory_usage,
            "cpu_count": self.cpu_count,
            "process_id": self.process_id
        }


class LogEntry:
    """日志条目"""
    
    def __init__(
        self,
        level: str,
        log_type: str,
        service: str,
        module: str,
        message: str,
        error_code: Optional[ErrorCode] = None,
        sdk_info: Optional[SDKInfo] = None,
        network_info: Optional[NetworkInfo] = None,
        performance_info: Optional[PerformanceInfo] = None,
        context: Optional[Dict[str, Any]] = None,
        stack: Optional[str] = None,
        duration_ms: Optional[float] = None,
        file_path: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.level = level
        self.type = log_type
        self.service = service
        self.module = module
        self.message = message
        self.error_code = error_code.value if error_code else None
        self.sdk_info = sdk_info.to_dict() if sdk_info else None
        self.network_info = network_info.to_dict() if network_info else None
        self.performance_info = performance_info.to_dict() if performance_info else None
        self.context = context or {}
        self.stack = stack
        self.duration_ms = duration_ms
        self.file_path = file_path
        self.api_endpoint = api_endpoint
        self.request_id = request_id
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = {
            "timestamp": self.timestamp,
            "level": self.level,
            "type": self.type,
            "service": self.service,
            "module": self.module,
            "message": self.message,
            "context": self.context
        }
        
        # 添加可选字段
        if self.error_code is not None:
            data["error_code"] = self.error_code
        if self.sdk_info is not None:
            data["sdk_info"] = self.sdk_info
        if self.network_info is not None:
            data["network_info"] = self.network_info
        if self.performance_info is not None:
            data["performance_info"] = self.performance_info
        if self.stack is not None:
            data["stack"] = self.stack
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        if self.file_path is not None:
            data["file_path"] = self.file_path
        if self.api_endpoint is not None:
            data["api_endpoint"] = self.api_endpoint
        if self.request_id is not None:
            data["request_id"] = self.request_id
        
        return data
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))


class VoiceHelperSDKLogger:
    """VoiceHelper SDK日志器"""
    
    def __init__(self, service: str = "voicehelper-python-sdk", module: str = ""):
        self.service = service
        self.module = module
        self.base_context: Dict[str, Any] = {}
        self.sdk_info = SDKInfo()
        
        # 设置Python标准日志器
        self.logger = logging.getLogger(f"{service}.{module}" if module else service)
        self.logger.setLevel(self._get_log_level())
        
        # 如果没有处理器，添加一个
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        
        # 设置日志文件
        self.log_file_path = self._setup_log_file()
        
        # 初始化时记录启动日志
        self.startup("SDK日志系统初始化", {
            "service": self.service,
            "module": self.module,
            "log_file_path": str(self.log_file_path) if self.log_file_path else None
        })
    
    def _get_log_level(self) -> int:
        """获取日志级别"""
        level_str = os.getenv("VOICEHELPER_LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_str, logging.INFO)
    
    def _setup_log_file(self) -> Optional[Path]:
        """设置日志文件"""
        log_dir = os.getenv("VOICEHELPER_LOG_DIR")
        if not log_dir:
            return None
        
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            log_file = log_path / f"voicehelper-sdk-{datetime.now().strftime('%Y-%m-%d')}.log"
            
            # 添加文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)
            
            return log_file
        except Exception as e:
            # 如果设置日志文件失败，只记录到控制台
            self.logger.warning(f"Failed to setup log file: {e}")
            return None
    
    def _build_log_entry(
        self,
        level: str,
        log_type: str,
        message: str,
        error_code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        file_path: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        request_id: Optional[str] = None,
        include_stack: bool = False,
        include_sdk_info: bool = False,
        include_performance: bool = False,
        network_info: Optional[NetworkInfo] = None
    ) -> LogEntry:
        """构建日志条目"""
        
        # 合并上下文
        merged_context = {**self.base_context}
        if context:
            merged_context.update(context)
        
        # 获取堆栈信息
        stack = None
        if include_stack or level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            stack = traceback.format_stack()
            stack = ''.join(stack[:-1])  # 排除当前调用
        
        # 获取SDK信息
        sdk_info = self.sdk_info if include_sdk_info or log_type == LogType.STARTUP else None
        
        # 获取性能信息
        performance_info = PerformanceInfo() if include_performance or log_type == LogType.PERFORMANCE else None
        
        return LogEntry(
            level=level,
            log_type=log_type,
            service=self.service,
            module=self.module,
            message=message,
            error_code=error_code,
            sdk_info=sdk_info,
            network_info=network_info,
            performance_info=performance_info,
            context=merged_context,
            stack=stack,
            duration_ms=duration_ms,
            file_path=file_path,
            api_endpoint=api_endpoint,
            request_id=request_id
        )
    
    def _log(self, entry: LogEntry):
        """记录日志"""
        json_message = entry.to_json()
        
        # 根据级别选择日志方法
        if entry.level == LogLevel.DEBUG:
            self.logger.debug(json_message)
        elif entry.level == LogLevel.INFO:
            self.logger.info(json_message)
        elif entry.level == LogLevel.WARNING:
            self.logger.warning(json_message)
        elif entry.level == LogLevel.ERROR:
            self.logger.error(json_message)
        elif entry.level == LogLevel.CRITICAL:
            self.logger.critical(json_message)
        else:
            self.logger.info(json_message)
    
    # 基础日志方法
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录调试日志"""
        entry = self._build_log_entry(LogLevel.DEBUG, LogType.DEBUG, message, context=context)
        self._log(entry)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录信息日志"""
        entry = self._build_log_entry(LogLevel.INFO, LogType.SYSTEM, message, context=context)
        self._log(entry)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录警告日志"""
        entry = self._build_log_entry(LogLevel.WARNING, LogType.SYSTEM, message, context=context)
        self._log(entry)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录错误日志"""
        entry = self._build_log_entry(
            LogLevel.ERROR, LogType.ERROR, message, 
            context=context, include_stack=True, include_sdk_info=True
        )
        self._log(entry)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录严重错误日志"""
        entry = self._build_log_entry(
            LogLevel.CRITICAL, LogType.ERROR, message, 
            context=context, include_stack=True, include_sdk_info=True
        )
        self._log(entry)
    
    # 特定类型日志方法
    def startup(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录启动日志"""
        entry = self._build_log_entry(
            LogLevel.INFO, LogType.STARTUP, message, 
            context=context, include_sdk_info=True, include_performance=True
        )
        self._log(entry)
    
    def shutdown(self, message: str, context: Optional[Dict[str, Any]] = None):
        """记录关闭日志"""
        entry = self._build_log_entry(
            LogLevel.INFO, LogType.SHUTDOWN, message, 
            context=context, include_performance=True
        )
        self._log(entry)
    
    def api_request(self, method: str, url: str, request_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """记录API请求日志"""
        message = f"API Request: {method} {url}"
        network_info = NetworkInfo(url, method)
        entry = self._build_log_entry(
            LogLevel.DEBUG, LogType.API_REQUEST, message,
            context=context, api_endpoint=url, request_id=request_id, network_info=network_info
        )
        self._log(entry)
    
    def api_response(
        self, 
        method: str, 
        url: str, 
        status_code: int, 
        duration_ms: float, 
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录API响应日志"""
        message = f"API Response: {method} {url} - {status_code}"
        level = LogLevel.ERROR if status_code >= 500 else LogLevel.WARNING if status_code >= 400 else LogLevel.DEBUG
        network_info = NetworkInfo(url, method)
        
        merged_context = {"status_code": status_code}
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(
            level, LogType.API_RESPONSE, message,
            context=merged_context, duration_ms=duration_ms, 
            api_endpoint=url, request_id=request_id, network_info=network_info
        )
        self._log(entry)
    
    def file_operation(self, operation: str, file_path: str, success: bool, context: Optional[Dict[str, Any]] = None):
        """记录文件操作日志"""
        message = f"File {operation}: {file_path} - {'success' if success else 'failed'}"
        level = LogLevel.DEBUG if success else LogLevel.WARNING
        
        merged_context = {"operation": operation, "success": success}
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(
            level, LogType.FILE_OPERATION, message,
            context=merged_context, file_path=file_path
        )
        self._log(entry)
    
    def authentication(self, action: str, success: bool, context: Optional[Dict[str, Any]] = None):
        """记录认证日志"""
        message = f"Authentication {action}: {'success' if success else 'failed'}"
        level = LogLevel.INFO if success else LogLevel.WARNING
        
        merged_context = {"action": action, "success": success}
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(level, LogType.AUTHENTICATION, message, context=merged_context)
        self._log(entry)
    
    def error_with_code(self, error_code: ErrorCode, message: str, context: Optional[Dict[str, Any]] = None):
        """记录带错误码的错误日志"""
        error_info = get_error_info(error_code)
        
        merged_context = {
            "error_info": {
                "code": error_info.code.value,
                "message": error_info.message,
                "description": error_info.description,
                "category": error_info.category,
                "service": error_info.service
            }
        }
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(
            LogLevel.ERROR, LogType.ERROR, message,
            error_code=error_code, context=merged_context, 
            include_stack=True, include_sdk_info=True
        )
        self._log(entry)
    
    def performance(self, operation: str, duration_ms: float, context: Optional[Dict[str, Any]] = None):
        """记录性能日志"""
        message = f"Performance: {operation}"
        
        merged_context = {"operation": operation}
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(
            LogLevel.INFO, LogType.PERFORMANCE, message,
            context=merged_context, duration_ms=duration_ms, include_performance=True
        )
        self._log(entry)
    
    def security(self, event: str, context: Optional[Dict[str, Any]] = None):
        """记录安全日志"""
        message = f"Security Event: {event}"
        
        merged_context = {"event": event}
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(
            LogLevel.WARNING, LogType.SECURITY, message,
            context=merged_context, include_sdk_info=True
        )
        self._log(entry)
    
    def business(self, event: str, context: Optional[Dict[str, Any]] = None):
        """记录业务日志"""
        message = f"Business Event: {event}"
        
        merged_context = {"event": event}
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(LogLevel.INFO, LogType.BUSINESS, message, context=merged_context)
        self._log(entry)
    
    def exception(self, message: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """记录异常日志"""
        error_code = None
        if isinstance(error, VoiceHelperSDKError):
            error_code = error.code
        
        merged_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        if context:
            merged_context.update(context)
        
        entry = self._build_log_entry(
            LogLevel.ERROR, LogType.ERROR, message,
            error_code=error_code, context=merged_context, 
            include_stack=True, include_sdk_info=True
        )
        self._log(entry)
    
    # 上下文管理
    def set_context(self, context: Dict[str, Any]):
        """设置基础上下文"""
        self.base_context.update(context)
    
    def with_module(self, module: str) -> 'VoiceHelperSDKLogger':
        """创建带有指定模块的新日志器"""
        new_logger = VoiceHelperSDKLogger(self.service, module)
        new_logger.base_context = self.base_context.copy()
        return new_logger
    
    def with_context(self, context: Dict[str, Any]) -> 'VoiceHelperSDKLogger':
        """创建带有额外上下文的新日志器"""
        new_logger = VoiceHelperSDKLogger(self.service, self.module)
        new_logger.base_context = {**self.base_context, **context}
        return new_logger
    
    # 日志文件管理
    def get_log_file_path(self) -> Optional[Path]:
        """获取日志文件路径"""
        return self.log_file_path
    
    def clean_old_logs(self, days_to_keep: int = 7):
        """清理过期日志"""
        if not self.log_file_path:
            return
        
        log_dir = self.log_file_path.parent
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        try:
            for log_file in log_dir.glob("voicehelper-sdk-*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.info("删除过期日志文件", {"log_file": str(log_file), "age": days_to_keep})
        except Exception as e:
            self.exception("清理过期日志失败", e)


# 全局日志器实例
_default_logger: Optional[VoiceHelperSDKLogger] = None


def init_logger(service: str = "voicehelper-python-sdk") -> VoiceHelperSDKLogger:
    """初始化默认日志器"""
    global _default_logger
    _default_logger = VoiceHelperSDKLogger(service)
    return _default_logger


def get_logger(module: str = "") -> VoiceHelperSDKLogger:
    """获取日志器"""
    if _default_logger is None:
        init_logger()
    
    if module:
        return _default_logger.with_module(module)
    
    return _default_logger


# 便利函数
def debug(message: str, context: Optional[Dict[str, Any]] = None):
    """记录调试日志"""
    get_logger().debug(message, context)


def info(message: str, context: Optional[Dict[str, Any]] = None):
    """记录信息日志"""
    get_logger().info(message, context)


def warning(message: str, context: Optional[Dict[str, Any]] = None):
    """记录警告日志"""
    get_logger().warning(message, context)


def error(message: str, context: Optional[Dict[str, Any]] = None):
    """记录错误日志"""
    get_logger().error(message, context)


def critical(message: str, context: Optional[Dict[str, Any]] = None):
    """记录严重错误日志"""
    get_logger().critical(message, context)


def startup(message: str, context: Optional[Dict[str, Any]] = None):
    """记录启动日志"""
    get_logger().startup(message, context)


def shutdown(message: str, context: Optional[Dict[str, Any]] = None):
    """记录关闭日志"""
    get_logger().shutdown(message, context)


def error_with_code(error_code: ErrorCode, message: str, context: Optional[Dict[str, Any]] = None):
    """记录带错误码的错误日志"""
    get_logger().error_with_code(error_code, message, context)


def performance(operation: str, duration_ms: float, context: Optional[Dict[str, Any]] = None):
    """记录性能日志"""
    get_logger().performance(operation, duration_ms, context)


def security(event: str, context: Optional[Dict[str, Any]] = None):
    """记录安全日志"""
    get_logger().security(event, context)


def business(event: str, context: Optional[Dict[str, Any]] = None):
    """记录业务日志"""
    get_logger().business(event, context)


def exception(message: str, error: Exception, context: Optional[Dict[str, Any]] = None):
    """记录异常日志"""
    get_logger().exception(message, error, context)
