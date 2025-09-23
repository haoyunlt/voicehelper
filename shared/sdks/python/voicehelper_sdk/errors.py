"""
VoiceHelper Python SDK 错误码系统
与其他平台保持一致的错误码体系
"""

from typing import Dict, Any, Optional
from enum import IntEnum


class ErrorCode(IntEnum):
    """VoiceHelper统一错误码"""
    
    # 成功码
    SUCCESS = 0
    
    # ========== Gateway服务错误码 (1xxxxx) ==========
    GATEWAY_INTERNAL_ERROR = 102001
    GATEWAY_SERVICE_UNAVAILABLE = 102002
    GATEWAY_TIMEOUT = 102003
    GATEWAY_INVALID_REQUEST = 111001
    GATEWAY_MISSING_PARAMETER = 111002
    GATEWAY_INVALID_PARAMETER = 111003
    GATEWAY_REQUEST_TOO_LARGE = 111004
    GATEWAY_RATE_LIMIT_EXCEEDED = 111005
    GATEWAY_NETWORK_ERROR = 133001
    GATEWAY_CONNECTION_FAILED = 133002
    GATEWAY_DNS_RESOLVE_FAILED = 133003
    
    # ========== 认证服务错误码 (2xxxxx) ==========
    AUTH_INTERNAL_ERROR = 202001
    AUTH_SERVICE_UNAVAILABLE = 202002
    AUTH_INVALID_CREDENTIALS = 211001
    AUTH_TOKEN_EXPIRED = 211002
    AUTH_TOKEN_INVALID = 211003
    AUTH_PERMISSION_DENIED = 211004
    AUTH_USER_NOT_FOUND = 211005
    AUTH_USER_DISABLED = 211006
    AUTH_SECURITY_VIOLATION = 281001
    AUTH_BRUTE_FORCE_DETECTED = 281002
    AUTH_SUSPICIOUS_ACTIVITY = 281003
    
    # ========== 聊天服务错误码 (3xxxxx) ==========
    CHAT_INTERNAL_ERROR = 302001
    CHAT_SERVICE_UNAVAILABLE = 302002
    CHAT_INVALID_MESSAGE = 311001
    CHAT_MESSAGE_TOO_LONG = 311002
    CHAT_SESSION_NOT_FOUND = 311003
    CHAT_SESSION_EXPIRED = 311004
    CHAT_CONTEXT_LIMIT_EXCEEDED = 311005
    CHAT_RESPONSE_TIMEOUT = 371001
    CHAT_QUEUE_FULL = 371002
    CHAT_CONCURRENCY_LIMIT = 371003
    
    # ========== 语音服务错误码 (4xxxxx) ==========
    VOICE_INTERNAL_ERROR = 402001
    VOICE_SERVICE_UNAVAILABLE = 402002
    VOICE_INVALID_FORMAT = 411001
    VOICE_FILE_TOO_LARGE = 411002
    VOICE_PROCESSING_FAILED = 411003
    VOICE_ASR_FAILED = 411004
    VOICE_TTS_FAILED = 411005
    VOICE_EMOTION_ANALYSIS_FAILED = 411006
    VOICE_FILE_NOT_FOUND = 461001
    VOICE_FILE_CORRUPTED = 461002
    VOICE_STORAGE_FAILED = 461003
    
    # ========== RAG服务错误码 (5xxxxx) ==========
    RAG_INTERNAL_ERROR = 502001
    RAG_SERVICE_UNAVAILABLE = 502002
    RAG_INVALID_QUERY = 511001
    RAG_DOCUMENT_NOT_FOUND = 511002
    RAG_INDEXING_FAILED = 511003
    RAG_RETRIEVAL_FAILED = 511004
    RAG_EMBEDDING_FAILED = 511005
    RAG_RERANKING_FAILED = 511006
    RAG_VECTOR_DB_ERROR = 533001
    RAG_VECTOR_DB_CONNECTION_FAILED = 533002
    RAG_COLLECTION_NOT_FOUND = 533003
    
    # ========== 存储服务错误码 (6xxxxx) ==========
    STORAGE_INTERNAL_ERROR = 602001
    STORAGE_SERVICE_UNAVAILABLE = 602002
    STORAGE_FILE_NOT_FOUND = 661001
    STORAGE_FILE_ACCESS_DENIED = 661002
    STORAGE_FILE_CORRUPTED = 661003
    STORAGE_INSUFFICIENT_SPACE = 661004
    STORAGE_UPLOAD_FAILED = 661005
    STORAGE_DOWNLOAD_FAILED = 661006
    
    # ========== SDK特有错误码 (7xxxxx) ==========
    # SDK通用错误 (70xxxx)
    SDK_INTERNAL_ERROR = 702001
    SDK_INITIALIZATION_FAILED = 702002
    SDK_CONFIGURATION_ERROR = 702003
    SDK_VERSION_MISMATCH = 702004
    
    # SDK网络错误 (73xxxx)
    SDK_NETWORK_ERROR = 731001
    SDK_CONNECTION_TIMEOUT = 731002
    SDK_CONNECTION_REFUSED = 731003
    SDK_SSL_ERROR = 731004
    SDK_PROXY_ERROR = 731005
    
    # SDK认证错误 (72xxxx)
    SDK_AUTH_ERROR = 721001
    SDK_API_KEY_INVALID = 721002
    SDK_API_KEY_EXPIRED = 721003
    SDK_SIGNATURE_ERROR = 721004
    
    # SDK参数错误 (71xxxx)
    SDK_INVALID_PARAMETER = 711001
    SDK_MISSING_PARAMETER = 711002
    SDK_PARAMETER_TYPE_ERROR = 711003
    SDK_PARAMETER_VALUE_ERROR = 711004
    
    # SDK文件错误 (76xxxx)
    SDK_FILE_NOT_FOUND = 761001
    SDK_FILE_READ_ERROR = 761002
    SDK_FILE_WRITE_ERROR = 761003
    SDK_FILE_FORMAT_ERROR = 761004
    
    # SDK序列化错误 (75xxxx)
    SDK_SERIALIZATION_ERROR = 751001
    SDK_DESERIALIZATION_ERROR = 751002
    SDK_JSON_ERROR = 751003
    SDK_ENCODING_ERROR = 751004
    
    # ========== 通用错误码 (9xxxxx) ==========
    SYSTEM_INTERNAL_ERROR = 902001
    SYSTEM_MAINTENANCE_MODE = 902002
    SYSTEM_OVERLOADED = 902003
    CONFIG_NOT_FOUND = 961001
    CONFIG_INVALID = 961002
    CONFIG_LOAD_FAILED = 961003
    NETWORK_TIMEOUT = 933001
    NETWORK_CONNECTION_REFUSED = 933002
    NETWORK_HOST_UNREACHABLE = 933003


class ErrorInfo:
    """错误信息类"""
    
    def __init__(self, code: ErrorCode, message: str, description: str, category: str, service: str):
        self.code = code
        self.message = message
        self.description = description
        self.category = category
        self.service = service
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "description": self.description,
            "category": self.category,
            "service": self.service
        }


# 错误信息映射
_ERROR_INFO_MAP: Dict[ErrorCode, ErrorInfo] = {
    ErrorCode.SUCCESS: ErrorInfo(
        ErrorCode.SUCCESS, "Success", "操作成功", "Success", "Common"
    ),
    
    # SDK错误
    ErrorCode.SDK_INTERNAL_ERROR: ErrorInfo(
        ErrorCode.SDK_INTERNAL_ERROR, "SDK Internal Error", "SDK内部错误", "SDK", "SDK"
    ),
    ErrorCode.SDK_INITIALIZATION_FAILED: ErrorInfo(
        ErrorCode.SDK_INITIALIZATION_FAILED, "SDK Initialization Failed", "SDK初始化失败", "SDK", "SDK"
    ),
    ErrorCode.SDK_CONFIGURATION_ERROR: ErrorInfo(
        ErrorCode.SDK_CONFIGURATION_ERROR, "SDK Configuration Error", "SDK配置错误", "SDK", "SDK"
    ),
    ErrorCode.SDK_VERSION_MISMATCH: ErrorInfo(
        ErrorCode.SDK_VERSION_MISMATCH, "SDK Version Mismatch", "SDK版本不匹配", "SDK", "SDK"
    ),
    ErrorCode.SDK_NETWORK_ERROR: ErrorInfo(
        ErrorCode.SDK_NETWORK_ERROR, "SDK Network Error", "SDK网络错误", "SDK", "SDK"
    ),
    ErrorCode.SDK_CONNECTION_TIMEOUT: ErrorInfo(
        ErrorCode.SDK_CONNECTION_TIMEOUT, "SDK Connection Timeout", "SDK连接超时", "SDK", "SDK"
    ),
    ErrorCode.SDK_AUTH_ERROR: ErrorInfo(
        ErrorCode.SDK_AUTH_ERROR, "SDK Authentication Error", "SDK认证错误", "SDK", "SDK"
    ),
    ErrorCode.SDK_API_KEY_INVALID: ErrorInfo(
        ErrorCode.SDK_API_KEY_INVALID, "SDK API Key Invalid", "SDK API密钥无效", "SDK", "SDK"
    ),
    ErrorCode.SDK_INVALID_PARAMETER: ErrorInfo(
        ErrorCode.SDK_INVALID_PARAMETER, "SDK Invalid Parameter", "SDK参数无效", "SDK", "SDK"
    ),
    ErrorCode.SDK_MISSING_PARAMETER: ErrorInfo(
        ErrorCode.SDK_MISSING_PARAMETER, "SDK Missing Parameter", "SDK缺少参数", "SDK", "SDK"
    ),
    ErrorCode.SDK_FILE_NOT_FOUND: ErrorInfo(
        ErrorCode.SDK_FILE_NOT_FOUND, "SDK File Not Found", "SDK文件未找到", "SDK", "SDK"
    ),
    ErrorCode.SDK_SERIALIZATION_ERROR: ErrorInfo(
        ErrorCode.SDK_SERIALIZATION_ERROR, "SDK Serialization Error", "SDK序列化错误", "SDK", "SDK"
    ),
    
    # 通用错误
    ErrorCode.SYSTEM_INTERNAL_ERROR: ErrorInfo(
        ErrorCode.SYSTEM_INTERNAL_ERROR, "System Internal Error", "系统内部错误", "System", "Common"
    ),
    ErrorCode.NETWORK_TIMEOUT: ErrorInfo(
        ErrorCode.NETWORK_TIMEOUT, "Network Timeout", "网络超时", "Network", "Common"
    ),
    
    # 语音服务错误
    ErrorCode.VOICE_ASR_FAILED: ErrorInfo(
        ErrorCode.VOICE_ASR_FAILED, "ASR Failed", "语音识别失败", "Voice", "Voice"
    ),
    ErrorCode.VOICE_TTS_FAILED: ErrorInfo(
        ErrorCode.VOICE_TTS_FAILED, "TTS Failed", "语音合成失败", "Voice", "Voice"
    ),
    
    # RAG服务错误
    ErrorCode.RAG_RETRIEVAL_FAILED: ErrorInfo(
        ErrorCode.RAG_RETRIEVAL_FAILED, "RAG Retrieval Failed", "RAG检索失败", "RAG", "RAG"
    ),
    ErrorCode.RAG_SERVICE_UNAVAILABLE: ErrorInfo(
        ErrorCode.RAG_SERVICE_UNAVAILABLE, "RAG Service Unavailable", "RAG服务不可用", "RAG", "RAG"
    ),
    
    # 认证错误
    ErrorCode.AUTH_TOKEN_EXPIRED: ErrorInfo(
        ErrorCode.AUTH_TOKEN_EXPIRED, "Token Expired", "Token过期", "Auth", "Auth"
    ),
    ErrorCode.AUTH_PERMISSION_DENIED: ErrorInfo(
        ErrorCode.AUTH_PERMISSION_DENIED, "Permission Denied", "权限不足", "Auth", "Auth"
    ),
}


def get_error_info(code: ErrorCode) -> ErrorInfo:
    """获取错误信息"""
    return _ERROR_INFO_MAP.get(code, ErrorInfo(
        code, "Unknown Error", f"未知错误码: {code.value}", "Unknown", "Unknown"
    ))


class VoiceHelperSDKError(Exception):
    """VoiceHelper SDK自定义异常类"""
    
    def __init__(self, code: ErrorCode, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.error_info = get_error_info(code)
        self.message = message or self.error_info.message
        self.details = details or {}
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            **self.error_info.to_dict(),
            "custom_message": self.message if self.message != self.error_info.message else None,
            "details": self.details
        }
    
    def __str__(self) -> str:
        return f"VoiceHelperSDKError({self.code.value}): {self.message}"
    
    def __repr__(self) -> str:
        return f"VoiceHelperSDKError(code={self.code}, message='{self.message}', details={self.details})"


# 便利函数
def create_auth_error(message: str = "认证失败", details: Optional[Dict[str, Any]] = None) -> VoiceHelperSDKError:
    """创建认证错误"""
    return VoiceHelperSDKError(ErrorCode.SDK_AUTH_ERROR, message, details)


def create_network_error(message: str = "网络错误", details: Optional[Dict[str, Any]] = None) -> VoiceHelperSDKError:
    """创建网络错误"""
    return VoiceHelperSDKError(ErrorCode.SDK_NETWORK_ERROR, message, details)


def create_parameter_error(message: str = "参数错误", details: Optional[Dict[str, Any]] = None) -> VoiceHelperSDKError:
    """创建参数错误"""
    return VoiceHelperSDKError(ErrorCode.SDK_INVALID_PARAMETER, message, details)


def create_file_error(message: str = "文件错误", details: Optional[Dict[str, Any]] = None) -> VoiceHelperSDKError:
    """创建文件错误"""
    return VoiceHelperSDKError(ErrorCode.SDK_FILE_NOT_FOUND, message, details)


def create_serialization_error(message: str = "序列化错误", details: Optional[Dict[str, Any]] = None) -> VoiceHelperSDKError:
    """创建序列化错误"""
    return VoiceHelperSDKError(ErrorCode.SDK_SERIALIZATION_ERROR, message, details)
