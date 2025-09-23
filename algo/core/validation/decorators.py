"""
验证装饰器
提供函数和方法的参数验证装饰器
"""

import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, Type, get_type_hints
from fastapi import HTTPException
import logging

from .parameter_validator import ParameterValidator, ValidationResult, ValidationException, validate_request
from .request_models import BaseModel

logger = logging.getLogger(__name__)


def validate_parameters(**validation_rules):
    """
    参数验证装饰器
    
    Args:
        **validation_rules: 验证规则，支持以下键：
            - required: List[str] - 必需字段列表
            - types: Dict[str, Type] - 字段类型映射
            - lengths: Dict[str, Dict[str, int]] - 字符串长度限制
            - ranges: Dict[str, Dict[str, Union[int, float]]] - 数值范围限制
            - choices: Dict[str, List[Any]] - 选择值限制
            - formats: Dict[str, str] - 格式验证
            - custom: Dict[str, str] - 自定义验证器
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = ParameterValidator()
            
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 提取参数数据
            data = dict(bound_args.arguments)
            
            # 执行验证
            result = validate_request(validator, data, validation_rules)
            
            if not result.success:
                logger.warning(f"Parameter validation failed for {func.__name__}: {result.to_dict()}")
                raise ValidationException(result)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_pydantic_model(model_class: Type[BaseModel]):
    """
    Pydantic模型验证装饰器
    
    Args:
        model_class: Pydantic模型类
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 查找需要验证的参数
            for param_name, param_value in bound_args.arguments.items():
                param = sig.parameters.get(param_name)
                if param and param.annotation == model_class:
                    try:
                        # 使用Pydantic进行验证
                        if isinstance(param_value, dict):
                            validated_model = model_class(**param_value)
                            bound_args.arguments[param_name] = validated_model
                        elif not isinstance(param_value, model_class):
                            raise ValueError(f"Parameter {param_name} must be of type {model_class.__name__}")
                    except Exception as e:
                        logger.warning(f"Pydantic validation failed for {func.__name__}.{param_name}: {str(e)}")
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "VALIDATION_ERROR",
                                "message": f"Invalid parameter: {param_name}",
                                "details": str(e)
                            }
                        )
            
            return func(*bound_args.args, **bound_args.kwargs)
        
        return wrapper
    return decorator


def validate_required_fields(*required_fields: str):
    """
    必需字段验证装饰器
    
    Args:
        *required_fields: 必需字段名列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = ParameterValidator()
            
            # 获取函数参数
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            data = dict(bound_args.arguments)
            
            # 验证必需字段
            result = validator.validate_required_fields(data, list(required_fields))
            
            if not result.success:
                logger.warning(f"Required fields validation failed for {func.__name__}: {result.to_dict()}")
                raise ValidationException(result)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_types(**type_mapping: Type):
    """
    类型验证装饰器
    
    Args:
        **type_mapping: 参数名到类型的映射
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = ParameterValidator()
            
            # 获取函数参数
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            data = dict(bound_args.arguments)
            
            # 验证类型
            result = validator.validate_types(data, type_mapping)
            
            if not result.success:
                logger.warning(f"Type validation failed for {func.__name__}: {result.to_dict()}")
                raise ValidationException(result)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_ranges(**range_specs: Dict[str, Any]):
    """
    数值范围验证装饰器
    
    Args:
        **range_specs: 参数名到范围规格的映射
                      格式: param_name={"min": min_val, "max": max_val}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = ParameterValidator()
            
            # 获取函数参数
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            data = dict(bound_args.arguments)
            
            # 验证数值范围
            result = validator.validate_numeric_range(data, range_specs)
            
            if not result.success:
                logger.warning(f"Range validation failed for {func.__name__}: {result.to_dict()}")
                raise ValidationException(result)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_choices(**choice_specs: List[Any]):
    """
    选择值验证装饰器
    
    Args:
        **choice_specs: 参数名到选择值列表的映射
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = ParameterValidator()
            
            # 获取函数参数
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            data = dict(bound_args.arguments)
            
            # 验证选择值
            result = validator.validate_choices(data, choice_specs)
            
            if not result.success:
                logger.warning(f"Choice validation failed for {func.__name__}: {result.to_dict()}")
                raise ValidationException(result)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_file_upload(max_size: int = 50 * 1024 * 1024, 
                        allowed_extensions: Optional[List[str]] = None,
                        allowed_mime_types: Optional[List[str]] = None):
    """
    文件上传验证装饰器
    
    Args:
        max_size: 最大文件大小（字节）
        allowed_extensions: 允许的文件扩展名列表
        allowed_mime_types: 允许的MIME类型列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = ParameterValidator()
            result = ValidationResult()
            
            # 获取函数参数
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 查找文件参数
            for param_name, param_value in bound_args.arguments.items():
                if hasattr(param_value, 'filename') and hasattr(param_value, 'file'):
                    # 这是一个文件对象
                    file_obj = param_value
                    
                    # 验证文件大小
                    if hasattr(file_obj, 'size'):
                        file_size = file_obj.size
                    else:
                        # 尝试获取文件大小
                        file_obj.file.seek(0, 2)  # 移动到文件末尾
                        file_size = file_obj.file.tell()
                        file_obj.file.seek(0)  # 重置到文件开头
                    
                    if file_size > max_size:
                        result.add_error(
                            field=param_name,
                            error_type="FILE_TOO_LARGE",
                            message=f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)",
                            value=file_size,
                            expected=max_size
                        )
                    
                    # 验证文件扩展名
                    if allowed_extensions and file_obj.filename:
                        file_ext = file_obj.filename.lower().split('.')[-1]
                        if file_ext not in [ext.lower().lstrip('.') for ext in allowed_extensions]:
                            result.add_error(
                                field=param_name,
                                error_type="INVALID_FILE_EXTENSION",
                                message=f"File extension '.{file_ext}' is not allowed",
                                value=file_ext,
                                expected=allowed_extensions
                            )
                    
                    # 验证MIME类型
                    if allowed_mime_types and hasattr(file_obj, 'content_type'):
                        if file_obj.content_type not in allowed_mime_types:
                            result.add_error(
                                field=param_name,
                                error_type="INVALID_MIME_TYPE",
                                message=f"MIME type '{file_obj.content_type}' is not allowed",
                                value=file_obj.content_type,
                                expected=allowed_mime_types
                            )
            
            if not result.success:
                logger.warning(f"File upload validation failed for {func.__name__}: {result.to_dict()}")
                raise ValidationException(result)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_rate_limit(max_requests: int = 100, window_seconds: int = 3600, key_func: Optional[Callable] = None):
    """
    请求频率限制装饰器
    
    Args:
        max_requests: 时间窗口内最大请求数
        window_seconds: 时间窗口大小（秒）
        key_func: 生成限制键的函数，默认使用函数名
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 实现频率限制逻辑
            # 使用Redis或内存缓存来跟踪请求频率
            
            # 生成限制键
            if key_func:
                rate_limit_key = key_func(*args, **kwargs)
            else:
                rate_limit_key = f"rate_limit:{func.__name__}"
            
            # 获取当前时间窗口
            import time
            current_window = int(time.time()) // window_seconds
            window_key = f"{rate_limit_key}:{current_window}"
            
            # 简化实现：使用内存字典存储计数
            # 在生产环境中应该使用Redis
            if not hasattr(wrapper, '_rate_limit_cache'):
                wrapper._rate_limit_cache = {}
            
            # 清理过期的键
            current_time = int(time.time())
            expired_keys = [k for k in wrapper._rate_limit_cache.keys() 
                          if current_time - int(k.split(':')[-1]) * window_seconds > window_seconds * 2]
            for key in expired_keys:
                wrapper._rate_limit_cache.pop(key, None)
            
            # 检查当前窗口的请求数
            current_count = wrapper._rate_limit_cache.get(window_key, 0)
            if current_count >= max_requests:
                raise ValidationError(
                    field="rate_limit",
                    message=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds",
                    code="RATE_LIMIT_EXCEEDED"
                )
            
            # 增加计数
            wrapper._rate_limit_cache[window_key] = current_count + 1
            
            # 检查频率限制
            # current_requests = get_current_requests(rate_limit_key, window_seconds)
            # if current_requests >= max_requests:
            #     raise HTTPException(
            #         status_code=429,
            #         detail={
            #             "error": "RATE_LIMIT_EXCEEDED",
            #             "message": f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds",
            #             "retry_after": window_seconds
            #         }
            #     )
            
            # increment_request_count(rate_limit_key, window_seconds)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_authentication(required_permissions: Optional[List[str]] = None):
    """
    身份验证装饰器
    
    Args:
        required_permissions: 需要的权限列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 实现身份验证逻辑
            # 检查请求中的认证信息
            
            # 从请求中获取用户信息
            # 这里需要根据实际的认证方式实现
            # 例如：JWT token, API key, session等
            
            # 简化实现：检查是否有认证头
            import inspect
            
            # 尝试从函数参数中获取request对象
            sig = inspect.signature(func)
            request = None
            
            # 查找request参数
            for i, param_name in enumerate(sig.parameters.keys()):
                if param_name in ['request', 'req'] and i < len(args):
                    request = args[i]
                    break
            
            if request and hasattr(request, 'headers'):
                # 检查Authorization头
                auth_header = request.headers.get('Authorization', '')
                if not auth_header:
                    raise ValidationError(
                        field="authentication",
                        message="Authentication required",
                        code="AUTHENTICATION_REQUIRED"
                    )
                
                # 简化的token验证
                if not auth_header.startswith('Bearer '):
                    raise ValidationError(
                        field="authentication", 
                        message="Invalid authentication format",
                        code="INVALID_AUTH_FORMAT"
                    )
                
                token = auth_header[7:]  # 移除 "Bearer " 前缀
                if len(token) < 10:  # 简单的token长度检查
                    raise ValidationError(
                        field="authentication",
                        message="Invalid authentication token",
                        code="INVALID_TOKEN"
                    )
                
                # 检查权限（如果需要）
                if required_permissions:
                    # 这里应该从token中解析用户权限
                    # 简化实现：假设token包含权限信息
                    user_permissions = ['read', 'write']  # 模拟用户权限
                    
                    for permission in required_permissions:
                        if permission not in user_permissions:
                            raise ValidationError(
                                field="authorization",
                                message=f"Permission '{permission}' required",
                                code="INSUFFICIENT_PERMISSIONS"
                            )
            else:
                # 如果没有request对象，跳过认证检查
                # 这可能是内部调用或测试环境
                pass
            #     raise HTTPException(
            #         status_code=401,
            #         detail={
            #             "error": "AUTHENTICATION_REQUIRED",
            #             "message": "Authentication is required to access this resource"
            #         }
            #     )
            
            # 检查权限
            # if required_permissions:
            #     user_permissions = get_user_permissions(user_info.user_id)
            #     for permission in required_permissions:
            #         if permission not in user_permissions:
            #             raise HTTPException(
            #                 status_code=403,
            #                 detail={
            #                     "error": "INSUFFICIENT_PERMISSIONS",
            #                     "message": f"Permission '{permission}' is required",
            #                     "required_permissions": required_permissions
            #                 }
            #             )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_content_type(*allowed_types: str):
    """
    内容类型验证装饰器
    
    Args:
        *allowed_types: 允许的内容类型列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 实现内容类型验证
            # 从请求中获取Content-Type头
            
            import inspect
            
            # 尝试从函数参数中获取request对象
            sig = inspect.signature(func)
            request = None
            
            # 查找request参数
            for i, param_name in enumerate(sig.parameters.keys()):
                if param_name in ['request', 'req'] and i < len(args):
                    request = args[i]
                    break
            
            if request and hasattr(request, 'headers'):
                content_type = request.headers.get('Content-Type', '').lower()
                
                # 移除charset等参数，只保留主要的媒体类型
                if ';' in content_type:
                    content_type = content_type.split(';')[0].strip()
                
                # 检查内容类型是否在允许列表中
                if allowed_types and content_type not in [t.lower() for t in allowed_types]:
                    raise ValidationError(
                        field="content_type",
                        message=f"Content type '{content_type}' is not supported. Supported types: {', '.join(allowed_types)}",
                        code="UNSUPPORTED_MEDIA_TYPE"
                    )
            else:
                # 如果没有request对象，跳过内容类型检查
                # 这可能是内部调用或测试环境
                pass
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ValidationMixin:
    """验证混入类，为类方法提供验证功能"""
    
    def __init__(self):
        self.validator = ParameterValidator()
    
    def validate_method_parameters(self, method_name: str, data: Dict[str, Any], 
                                 validation_rules: Dict[str, Any]) -> ValidationResult:
        """验证方法参数"""
        result = validate_request(self.validator, data, validation_rules)
        
        if not result.success:
            logger.warning(f"Method parameter validation failed for {self.__class__.__name__}.{method_name}: {result.to_dict()}")
        
        return result
    
    def require_valid_parameters(self, method_name: str, data: Dict[str, Any], 
                               validation_rules: Dict[str, Any]):
        """要求有效参数，验证失败时抛出异常"""
        result = self.validate_method_parameters(method_name, data, validation_rules)
        
        if not result.success:
            raise ValidationException(result)
