"""
参数验证模块
提供统一的参数验证功能，确保所有接口都有严格的参数检查
"""

import re
import json
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, ValidationError, validator
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class ValidationErrorType(Enum):
    """验证错误类型"""
    MISSING_REQUIRED = "missing_required"
    INVALID_TYPE = "invalid_type"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    INVALID_VALUE = "invalid_value"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    CUSTOM_ERROR = "custom_error"


@dataclass
class ValidationError:
    """验证错误详情"""
    field: str
    error_type: ValidationErrorType
    message: str
    value: Any = None
    expected: Any = None


class ValidationResult:
    """验证结果"""
    
    def __init__(self, success: bool = True, errors: List[ValidationError] = None):
        self.success = success
        self.errors = errors or []
        self.message = "Validation passed" if success else "Validation failed"
    
    def add_error(self, field: str, error_type: ValidationErrorType, message: str, 
                  value: Any = None, expected: Any = None):
        """添加验证错误"""
        error = ValidationError(
            field=field,
            error_type=error_type,
            message=message,
            value=value,
            expected=expected
        )
        self.errors.append(error)
        self.success = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "message": self.message,
            "errors": [
                {
                    "field": error.field,
                    "error_type": error.error_type.value,
                    "message": error.message,
                    "value": error.value,
                    "expected": error.expected
                }
                for error in self.errors
            ]
        }


class ParameterValidator:
    """参数验证器"""
    
    # 支持的音频格式
    SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "webm", "m4a", "flac", "ogg"}
    
    # 支持的语言代码
    SUPPORTED_LANGUAGES = {
        "zh-CN", "zh-TW", "en-US", "en-GB", "ja-JP", "ko-KR",
        "fr-FR", "de-DE", "es-ES", "it-IT", "pt-BR", "ru-RU"
    }
    
    # 支持的情感类型
    SUPPORTED_EMOTIONS = {
        "neutral", "happy", "sad", "angry", "surprised", "fear",
        "disgust", "calm", "excited", "frustrated", "confident"
    }
    
    # 支持的搜索类型
    SUPPORTED_SEARCH_TYPES = {"semantic", "keyword", "hybrid", "fuzzy"}
    
    # 支持的融合类型
    SUPPORTED_FUSION_TYPES = {"early", "late", "attention", "gated", "hierarchical"}
    
    def __init__(self):
        self.custom_validators = {}
    
    def register_custom_validator(self, name: str, validator_func: Callable):
        """注册自定义验证器"""
        self.custom_validators[name] = validator_func
    
    def validate_required_fields(self, data: Dict[str, Any], 
                                required_fields: List[str]) -> ValidationResult:
        """验证必需字段"""
        result = ValidationResult()
        
        for field in required_fields:
            if field not in data:
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.MISSING_REQUIRED,
                    message=f"Field '{field}' is required"
                )
            elif self._is_empty(data[field]):
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.MISSING_REQUIRED,
                    message=f"Field '{field}' cannot be empty"
                )
        
        return result
    
    def validate_types(self, data: Dict[str, Any], 
                      type_specs: Dict[str, Type]) -> ValidationResult:
        """验证字段类型"""
        result = ValidationResult()
        
        for field, expected_type in type_specs.items():
            if field in data:
                value = data[field]
                if not isinstance(value, expected_type):
                    result.add_error(
                        field=field,
                        error_type=ValidationErrorType.INVALID_TYPE,
                        message=f"Field '{field}' must be of type {expected_type.__name__}",
                        value=type(value).__name__,
                        expected=expected_type.__name__
                    )
        
        return result
    
    def validate_string_length(self, data: Dict[str, Any], 
                              length_specs: Dict[str, Dict[str, int]]) -> ValidationResult:
        """验证字符串长度"""
        result = ValidationResult()
        
        for field, spec in length_specs.items():
            if field in data and isinstance(data[field], str):
                value = data[field]
                min_len = spec.get("min", 0)
                max_len = spec.get("max", float('inf'))
                
                if len(value) < min_len:
                    result.add_error(
                        field=field,
                        error_type=ValidationErrorType.TOO_SHORT,
                        message=f"Field '{field}' must be at least {min_len} characters",
                        value=len(value),
                        expected=f"min={min_len}"
                    )
                elif len(value) > max_len:
                    result.add_error(
                        field=field,
                        error_type=ValidationErrorType.TOO_LONG,
                        message=f"Field '{field}' must be at most {max_len} characters",
                        value=len(value),
                        expected=f"max={max_len}"
                    )
        
        return result
    
    def validate_numeric_range(self, data: Dict[str, Any], 
                              range_specs: Dict[str, Dict[str, Union[int, float]]]) -> ValidationResult:
        """验证数值范围"""
        result = ValidationResult()
        
        for field, spec in range_specs.items():
            if field in data and isinstance(data[field], (int, float)):
                value = data[field]
                min_val = spec.get("min", float('-inf'))
                max_val = spec.get("max", float('inf'))
                
                if value < min_val:
                    result.add_error(
                        field=field,
                        error_type=ValidationErrorType.OUT_OF_RANGE,
                        message=f"Field '{field}' must be at least {min_val}",
                        value=value,
                        expected=f"min={min_val}"
                    )
                elif value > max_val:
                    result.add_error(
                        field=field,
                        error_type=ValidationErrorType.OUT_OF_RANGE,
                        message=f"Field '{field}' must be at most {max_val}",
                        value=value,
                        expected=f"max={max_val}"
                    )
        
        return result
    
    def validate_choices(self, data: Dict[str, Any], 
                        choice_specs: Dict[str, List[Any]]) -> ValidationResult:
        """验证选择值"""
        result = ValidationResult()
        
        for field, choices in choice_specs.items():
            if field in data:
                value = data[field]
                if value not in choices:
                    result.add_error(
                        field=field,
                        error_type=ValidationErrorType.INVALID_VALUE,
                        message=f"Field '{field}' must be one of: {', '.join(map(str, choices))}",
                        value=value,
                        expected=choices
                    )
        
        return result
    
    def validate_audio_format(self, data: Dict[str, Any], field: str) -> ValidationResult:
        """验证音频格式"""
        result = ValidationResult()
        
        if field in data:
            format_value = data[field].lower()
            if format_value not in self.SUPPORTED_AUDIO_FORMATS:
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    message=f"Unsupported audio format: {format_value}",
                    value=format_value,
                    expected=list(self.SUPPORTED_AUDIO_FORMATS)
                )
        
        return result
    
    def validate_language_code(self, data: Dict[str, Any], field: str) -> ValidationResult:
        """验证语言代码"""
        result = ValidationResult()
        
        if field in data:
            lang_code = data[field]
            if lang_code not in self.SUPPORTED_LANGUAGES:
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.INVALID_VALUE,
                    message=f"Unsupported language code: {lang_code}",
                    value=lang_code,
                    expected=list(self.SUPPORTED_LANGUAGES)
                )
        
        return result
    
    def validate_emotion(self, data: Dict[str, Any], field: str) -> ValidationResult:
        """验证情感类型"""
        result = ValidationResult()
        
        if field in data:
            emotion = data[field]
            if emotion not in self.SUPPORTED_EMOTIONS:
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.INVALID_VALUE,
                    message=f"Unsupported emotion: {emotion}",
                    value=emotion,
                    expected=list(self.SUPPORTED_EMOTIONS)
                )
        
        return result
    
    def validate_email(self, data: Dict[str, Any], field: str) -> ValidationResult:
        """验证邮箱格式"""
        result = ValidationResult()
        
        if field in data:
            email = data[field]
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    message=f"Invalid email format: {email}",
                    value=email
                )
        
        return result
    
    def validate_url(self, data: Dict[str, Any], field: str) -> ValidationResult:
        """验证URL格式"""
        result = ValidationResult()
        
        if field in data:
            url = data[field]
            url_pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
            if not re.match(url_pattern, url):
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    message=f"Invalid URL format: {url}",
                    value=url
                )
        
        return result
    
    def validate_file_size(self, file_size: int, max_size: int, field: str) -> ValidationResult:
        """验证文件大小"""
        result = ValidationResult()
        
        if file_size > max_size:
            result.add_error(
                field=field,
                error_type=ValidationErrorType.OUT_OF_RANGE,
                message=f"File size exceeds maximum allowed size",
                value=f"{file_size} bytes",
                expected=f"max={max_size} bytes"
            )
        
        return result
    
    def validate_custom(self, data: Dict[str, Any], field: str, 
                       validator_name: str) -> ValidationResult:
        """使用自定义验证器"""
        result = ValidationResult()
        
        if validator_name not in self.custom_validators:
            result.add_error(
                field=field,
                error_type=ValidationErrorType.CUSTOM_ERROR,
                message=f"Custom validator '{validator_name}' not found"
            )
            return result
        
        if field in data:
            try:
                is_valid, error_message = self.custom_validators[validator_name](data[field])
                if not is_valid:
                    result.add_error(
                        field=field,
                        error_type=ValidationErrorType.CUSTOM_ERROR,
                        message=error_message or f"Custom validation failed for field '{field}'",
                        value=data[field]
                    )
            except Exception as e:
                result.add_error(
                    field=field,
                    error_type=ValidationErrorType.CUSTOM_ERROR,
                    message=f"Custom validator error: {str(e)}",
                    value=data[field]
                )
        
        return result
    
    def combine_results(self, *results: ValidationResult) -> ValidationResult:
        """合并多个验证结果"""
        combined = ValidationResult()
        
        for result in results:
            if not result.success:
                combined.success = False
                combined.errors.extend(result.errors)
        
        if not combined.success:
            combined.message = f"Validation failed with {len(combined.errors)} errors"
        
        return combined
    
    def _is_empty(self, value: Any) -> bool:
        """检查值是否为空"""
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, dict, tuple)):
            return len(value) == 0
        return False


class ValidationException(HTTPException):
    """验证异常"""
    
    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        super().__init__(
            status_code=400,
            detail={
                "error": "VALIDATION_ERROR",
                "message": validation_result.message,
                "details": validation_result.to_dict()
            }
        )


def validate_request(validator: ParameterValidator, 
                    data: Dict[str, Any],
                    validation_rules: Dict[str, Any]) -> ValidationResult:
    """通用请求验证函数"""
    results = []
    
    # 验证必需字段
    if "required" in validation_rules:
        results.append(validator.validate_required_fields(data, validation_rules["required"]))
    
    # 验证类型
    if "types" in validation_rules:
        results.append(validator.validate_types(data, validation_rules["types"]))
    
    # 验证字符串长度
    if "lengths" in validation_rules:
        results.append(validator.validate_string_length(data, validation_rules["lengths"]))
    
    # 验证数值范围
    if "ranges" in validation_rules:
        results.append(validator.validate_numeric_range(data, validation_rules["ranges"]))
    
    # 验证选择值
    if "choices" in validation_rules:
        results.append(validator.validate_choices(data, validation_rules["choices"]))
    
    # 验证特殊格式
    if "formats" in validation_rules:
        for field, format_type in validation_rules["formats"].items():
            if format_type == "email":
                results.append(validator.validate_email(data, field))
            elif format_type == "url":
                results.append(validator.validate_url(data, field))
            elif format_type == "audio_format":
                results.append(validator.validate_audio_format(data, field))
            elif format_type == "language_code":
                results.append(validator.validate_language_code(data, field))
            elif format_type == "emotion":
                results.append(validator.validate_emotion(data, field))
    
    # 自定义验证
    if "custom" in validation_rules:
        for field, validator_name in validation_rules["custom"].items():
            results.append(validator.validate_custom(data, field, validator_name))
    
    return validator.combine_results(*results)


# 全局验证器实例
global_validator = ParameterValidator()


def get_validator() -> ParameterValidator:
    """获取全局验证器实例"""
    return global_validator
