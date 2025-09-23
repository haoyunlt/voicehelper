"""
参数验证使用示例
展示如何在各种场景下使用参数验证功能
"""

import asyncio
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

# 导入验证模块
from algo.core.validation.parameter_validator import ParameterValidator, ValidationException
from algo.core.validation.request_models import (
    TranscribeRequest, SynthesizeRequest, QueryRequest, 
    ChatRequest, MultimodalRequest, AgentRequest
)
from algo.core.validation.decorators import (
    validate_parameters, validate_pydantic_model, validate_required_fields,
    validate_file_upload, validate_rate_limit, validate_authentication
)

app = FastAPI(title="VoiceHelper API with Validation")


# 示例1: 使用装饰器进行参数验证
@app.post("/api/v1/voice/transcribe")
@validate_file_upload(
    max_size=50 * 1024 * 1024,  # 50MB
    allowed_extensions=['.wav', '.mp3', '.webm', '.m4a'],
    allowed_mime_types=['audio/wav', 'audio/mpeg', 'audio/webm', 'audio/x-m4a']
)
@validate_rate_limit(max_requests=200, window_seconds=3600)
@validate_authentication(required_permissions=["voice:transcribe"])
async def transcribe_audio_with_validation(
    audio_file: UploadFile = File(...),
    language: str = Form(...),
    audio_format: str = Form(...),
    sample_rate: int = Form(16000),
    enable_emotion: bool = Form(False)
):
    """
    语音转文字接口 - 使用装饰器验证
    """
    try:
        # 手动验证参数
        validator = ParameterValidator()
        
        # 验证语言代码
        result = validator.validate_language_code({"language": language}, "language")
        if not result.success:
            raise ValidationException(result)
        
        # 验证音频格式
        result = validator.validate_audio_format({"audio_format": audio_format}, "audio_format")
        if not result.success:
            raise ValidationException(result)
        
        # 验证采样率
        result = validator.validate_numeric_range(
            {"sample_rate": sample_rate}, 
            {"sample_rate": {"min": 8000, "max": 48000}}
        )
        if not result.success:
            raise ValidationException(result)
        
        # 处理音频文件
        audio_data = await audio_file.read()
        
        # 这里调用实际的语音识别服务
        # result = await voice_service.transcribe(audio_data, language, ...)
        
        return {
            "success": True,
            "result": {
                "text": "这是识别出的文本内容",
                "confidence": 0.95,
                "language": language,
                "duration": 5.2
            }
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 示例2: 使用Pydantic模型验证
@app.post("/api/v1/voice/synthesize")
@validate_pydantic_model(SynthesizeRequest)
@validate_rate_limit(max_requests=500, window_seconds=3600)
async def synthesize_text_with_validation(request: SynthesizeRequest):
    """
    文字转语音接口 - 使用Pydantic模型验证
    """
    try:
        # Pydantic已经完成了基本验证
        # 这里可以添加额外的业务逻辑验证
        
        # 检查文本长度（业务规则）
        word_count = len(request.text.split())
        if word_count > 500:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "TEXT_TOO_LONG",
                    "message": f"Text is too long: {word_count} words (max 500)",
                    "word_count": word_count,
                    "max_words": 500
                }
            )
        
        # 检查敏感内容
        sensitive_words = ["敏感词1", "敏感词2"]  # 实际应该从配置加载
        for word in sensitive_words:
            if word in request.text:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "SENSITIVE_CONTENT",
                        "message": "Text contains sensitive content"
                    }
                )
        
        # 这里调用实际的语音合成服务
        # audio_data = await tts_service.synthesize(request.text, ...)
        
        return {
            "success": True,
            "result": {
                "audio_url": "https://example.com/audio/output.wav",
                "duration": 8.5,
                "text_length": len(request.text),
                "voice_info": {
                    "voice_id": request.voice_id,
                    "language": request.language.value,
                    "emotion": request.emotion.value if request.emotion else "neutral"
                }
            }
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 示例3: 手动参数验证
@app.post("/api/v1/search")
async def search_documents_with_manual_validation(request: Dict[str, Any]):
    """
    文档搜索接口 - 手动参数验证
    """
    try:
        validator = ParameterValidator()
        
        # 定义验证规则
        validation_rules = {
            "required": ["query"],
            "types": {
                "query": str,
                "limit": int,
                "offset": int,
                "search_type": str
            },
            "lengths": {
                "query": {"min": 1, "max": 1000}
            },
            "ranges": {
                "limit": {"min": 1, "max": 100},
                "offset": {"min": 0}
            },
            "choices": {
                "search_type": ["semantic", "keyword", "hybrid", "fuzzy"]
            }
        }
        
        # 执行验证
        from algo.core.validation.parameter_validator import validate_request
        result = validate_request(validator, request, validation_rules)
        
        if not result.success:
            raise ValidationException(result)
        
        # 设置默认值
        query = request["query"]
        limit = request.get("limit", 10)
        offset = request.get("offset", 0)
        search_type = request.get("search_type", "semantic")
        filters = request.get("filters", {})
        
        # 验证过滤条件
        if filters:
            allowed_filter_keys = ["category", "language", "source", "created_after"]
            invalid_keys = set(filters.keys()) - set(allowed_filter_keys)
            if invalid_keys:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "INVALID_FILTERS",
                        "message": "Invalid filter keys",
                        "invalid_keys": list(invalid_keys),
                        "allowed_keys": allowed_filter_keys
                    }
                )
        
        # 这里调用实际的搜索服务
        # results = await search_service.search(query, limit, offset, ...)
        
        return {
            "success": True,
            "results": [
                {
                    "id": "doc_001",
                    "title": "示例文档1",
                    "content": "这是文档内容...",
                    "score": 0.95
                },
                {
                    "id": "doc_002", 
                    "title": "示例文档2",
                    "content": "这是另一个文档内容...",
                    "score": 0.87
                }
            ],
            "total": 2,
            "query": query,
            "limit": limit,
            "offset": offset
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 示例4: 批量操作验证
@app.post("/api/v1/batch")
async def batch_operations_with_validation(request: Dict[str, Any]):
    """
    批量操作接口 - 复杂验证示例
    """
    try:
        validator = ParameterValidator()
        
        # 验证基本结构
        required_result = validator.validate_required_fields(request, ["operations"])
        if not required_result.success:
            raise ValidationException(required_result)
        
        operations = request["operations"]
        if not isinstance(operations, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_TYPE",
                    "message": "Operations must be a list"
                }
            )
        
        if len(operations) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "EMPTY_OPERATIONS",
                    "message": "Operations list cannot be empty"
                }
            )
        
        if len(operations) > 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "TOO_MANY_OPERATIONS",
                    "message": f"Too many operations: {len(operations)} (max 100)"
                }
            )
        
        # 验证每个操作
        valid_operation_types = ["transcribe", "synthesize", "search", "ingest"]
        operation_ids = set()
        
        for i, operation in enumerate(operations):
            # 验证操作结构
            op_result = validator.validate_required_fields(operation, ["id", "type", "parameters"])
            if not op_result.success:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "INVALID_OPERATION",
                        "message": f"Invalid operation at index {i}",
                        "details": op_result.to_dict()
                    }
                )
            
            # 验证操作ID唯一性
            op_id = operation["id"]
            if op_id in operation_ids:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "DUPLICATE_OPERATION_ID",
                        "message": f"Duplicate operation ID: {op_id}"
                    }
                )
            operation_ids.add(op_id)
            
            # 验证操作类型
            op_type = operation["type"]
            if op_type not in valid_operation_types:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "INVALID_OPERATION_TYPE",
                        "message": f"Invalid operation type: {op_type}",
                        "valid_types": valid_operation_types
                    }
                )
            
            # 根据操作类型验证参数
            parameters = operation["parameters"]
            if op_type == "transcribe":
                param_result = validator.validate_required_fields(parameters, ["audio_data", "language"])
                if not param_result.success:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "INVALID_TRANSCRIBE_PARAMETERS",
                            "message": f"Invalid transcribe parameters for operation {op_id}",
                            "details": param_result.to_dict()
                        }
                    )
            elif op_type == "synthesize":
                param_result = validator.validate_required_fields(parameters, ["text", "voice_id"])
                if not param_result.success:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "INVALID_SYNTHESIZE_PARAMETERS",
                            "message": f"Invalid synthesize parameters for operation {op_id}",
                            "details": param_result.to_dict()
                        }
                    )
        
        # 处理批量操作
        results = []
        for operation in operations:
            # 这里调用相应的服务处理每个操作
            # result = await process_operation(operation)
            results.append({
                "id": operation["id"],
                "status": "success",
                "result": {"message": f"Operation {operation['id']} completed"}
            })
        
        return {
            "success": True,
            "results": results,
            "total_operations": len(operations),
            "successful_operations": len(results)
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 示例5: 自定义验证器
def validate_phone_number(phone: str) -> tuple[bool, str]:
    """自定义手机号验证器"""
    import re
    pattern = r'^1[3-9]\d{9}$'
    if re.match(pattern, phone):
        return True, ""
    return False, "Invalid phone number format"


def validate_id_card(id_card: str) -> tuple[bool, str]:
    """自定义身份证号验证器"""
    if len(id_card) != 18:
        return False, "ID card must be 18 digits"
    
    # 简单的校验位验证
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
    
    try:
        sum_val = sum(int(id_card[i]) * weights[i] for i in range(17))
        check_code = check_codes[sum_val % 11]
        if id_card[17].upper() == check_code:
            return True, ""
        return False, "Invalid ID card check code"
    except (ValueError, IndexError):
        return False, "Invalid ID card format"


@app.post("/api/v1/user/register")
async def register_user_with_custom_validation(request: Dict[str, Any]):
    """
    用户注册接口 - 自定义验证器示例
    """
    try:
        validator = ParameterValidator()
        
        # 注册自定义验证器
        validator.register_custom_validator("phone_number", validate_phone_number)
        validator.register_custom_validator("id_card", validate_id_card)
        
        # 定义验证规则
        validation_rules = {
            "required": ["username", "phone", "id_card"],
            "types": {
                "username": str,
                "phone": str,
                "id_card": str,
                "email": str
            },
            "lengths": {
                "username": {"min": 3, "max": 20},
                "phone": {"min": 11, "max": 11},
                "id_card": {"min": 18, "max": 18}
            },
            "formats": {
                "email": "email"
            },
            "custom": {
                "phone": "phone_number",
                "id_card": "id_card"
            }
        }
        
        # 执行验证
        from algo.core.validation.parameter_validator import validate_request
        result = validate_request(validator, request, validation_rules)
        
        if not result.success:
            raise ValidationException(result)
        
        # 额外的业务逻辑验证
        username = request["username"]
        
        # 检查用户名是否已存在（模拟）
        existing_users = ["admin", "test", "user123"]  # 实际应该查询数据库
        if username in existing_users:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "USERNAME_EXISTS",
                    "message": f"Username '{username}' already exists"
                }
            )
        
        # 这里执行用户注册逻辑
        # user_id = await user_service.create_user(request)
        
        return {
            "success": True,
            "result": {
                "user_id": "user_12345",
                "username": username,
                "status": "registered"
            }
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=e.detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 示例6: 错误处理和日志记录
@app.exception_handler(ValidationException)
async def validation_exception_handler(request, exc: ValidationException):
    """全局验证异常处理器"""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.warning(f"Validation failed for {request.url}: {exc.validation_result.to_dict()}")
    
    return {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": exc.validation_result.to_dict(),
            "timestamp": int(asyncio.get_event_loop().time())
        }
    }


# 示例7: 健康检查和配置验证
@app.get("/api/v1/health")
async def health_check():
    """健康检查接口"""
    try:
        # 验证系统配置
        validator = ParameterValidator()
        
        # 检查验证器状态
        validator_status = {
            "custom_validators": len(validator.custom_validators),
            "supported_languages": len(validator.SUPPORTED_LANGUAGES),
            "supported_audio_formats": len(validator.SUPPORTED_AUDIO_FORMATS)
        }
        
        return {
            "success": True,
            "status": "healthy",
            "timestamp": int(asyncio.get_event_loop().time()),
            "validation_system": validator_status,
            "services": {
                "parameter_validation": "active",
                "request_processing": "active"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": int(asyncio.get_event_loop().time())
        }


if __name__ == "__main__":
    import uvicorn
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
