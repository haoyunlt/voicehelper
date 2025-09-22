"""
VoiceHelper 错误码定义 (Python版本)
与Go版本保持一致的错误码体系
"""

from enum import IntEnum
from typing import Dict, Any
import json


class ErrorCode(IntEnum):
    """错误码枚举"""
    # 成功码
    SUCCESS = 0
    
    # ========== Gateway服务错误码 (1xxxxx) ==========
    # Gateway通用错误 (10xxxx)
    GATEWAY_INTERNAL_ERROR = 102001
    GATEWAY_SERVICE_UNAVAILABLE = 102002
    GATEWAY_TIMEOUT = 102003
    
    # Gateway API错误 (11xxxx)
    GATEWAY_INVALID_REQUEST = 111001
    GATEWAY_MISSING_PARAMETER = 111002
    GATEWAY_INVALID_PARAMETER = 111003
    GATEWAY_REQUEST_TOO_LARGE = 111004
    GATEWAY_RATE_LIMIT_EXCEEDED = 111005
    
    # Gateway网络错误 (13xxxx)
    GATEWAY_NETWORK_ERROR = 133001
    GATEWAY_CONNECTION_FAILED = 133002
    GATEWAY_DNS_RESOLVE_FAILED = 133003

    # ========== 认证服务错误码 (2xxxxx) ==========
    # 认证通用错误 (20xxxx)
    AUTH_INTERNAL_ERROR = 202001
    AUTH_SERVICE_UNAVAILABLE = 202002
    
    # 认证API错误 (21xxxx)
    AUTH_INVALID_CREDENTIALS = 211001
    AUTH_TOKEN_EXPIRED = 211002
    AUTH_TOKEN_INVALID = 211003
    AUTH_PERMISSION_DENIED = 211004
    AUTH_USER_NOT_FOUND = 211005
    AUTH_USER_DISABLED = 211006
    
    # 认证安全错误 (28xxxx)
    AUTH_SECURITY_VIOLATION = 281001
    AUTH_BRUTE_FORCE_DETECTED = 281002
    AUTH_SUSPICIOUS_ACTIVITY = 281003

    # ========== 聊天服务错误码 (3xxxxx) ==========
    # 聊天通用错误 (30xxxx)
    CHAT_INTERNAL_ERROR = 302001
    CHAT_SERVICE_UNAVAILABLE = 302002
    
    # 聊天API错误 (31xxxx)
    CHAT_INVALID_MESSAGE = 311001
    CHAT_MESSAGE_TOO_LONG = 311002
    CHAT_SESSION_NOT_FOUND = 311003
    CHAT_SESSION_EXPIRED = 311004
    CHAT_CONTEXT_LIMIT_EXCEEDED = 311005
    
    # 聊天性能错误 (37xxxx)
    CHAT_RESPONSE_TIMEOUT = 371001
    CHAT_QUEUE_FULL = 371002
    CHAT_CONCURRENCY_LIMIT = 371003

    # ========== 语音服务错误码 (4xxxxx) ==========
    # 语音通用错误 (40xxxx)
    VOICE_INTERNAL_ERROR = 402001
    VOICE_SERVICE_UNAVAILABLE = 402002
    
    # 语音API错误 (41xxxx)
    VOICE_INVALID_FORMAT = 411001
    VOICE_FILE_TOO_LARGE = 411002
    VOICE_PROCESSING_FAILED = 411003
    VOICE_ASR_FAILED = 411004
    VOICE_TTS_FAILED = 411005
    VOICE_EMOTION_ANALYSIS_FAILED = 411006
    
    # 语音文件错误 (46xxxx)
    VOICE_FILE_NOT_FOUND = 461001
    VOICE_FILE_CORRUPTED = 461002
    VOICE_STORAGE_FAILED = 461003

    # ========== RAG服务错误码 (5xxxxx) ==========
    # RAG通用错误 (50xxxx)
    RAG_INTERNAL_ERROR = 502001
    RAG_SERVICE_UNAVAILABLE = 502002
    
    # RAG API错误 (51xxxx)
    RAG_INVALID_QUERY = 511001
    RAG_DOCUMENT_NOT_FOUND = 511002
    RAG_INDEXING_FAILED = 511003
    RAG_RETRIEVAL_FAILED = 511004
    RAG_EMBEDDING_FAILED = 511005
    RAG_RERANKING_FAILED = 511006
    
    # RAG数据库错误 (53xxxx)
    RAG_VECTOR_DB_ERROR = 533001
    RAG_VECTOR_DB_CONNECTION_FAILED = 533002
    RAG_COLLECTION_NOT_FOUND = 533003

    # ========== 存储服务错误码 (6xxxxx) ==========
    # 存储通用错误 (60xxxx)
    STORAGE_INTERNAL_ERROR = 602001
    STORAGE_SERVICE_UNAVAILABLE = 602002
    
    # 存储文件错误 (66xxxx)
    STORAGE_FILE_NOT_FOUND = 661001
    STORAGE_FILE_ACCESS_DENIED = 661002
    STORAGE_FILE_CORRUPTED = 661003
    STORAGE_INSUFFICIENT_SPACE = 661004
    STORAGE_UPLOAD_FAILED = 661005
    STORAGE_DOWNLOAD_FAILED = 661006

    # ========== 集成服务错误码 (7xxxxx) ==========
    # 集成通用错误 (70xxxx)
    INTEGRATION_INTERNAL_ERROR = 702001
    INTEGRATION_SERVICE_UNAVAILABLE = 702002
    
    # 集成API错误 (71xxxx)
    INTEGRATION_INVALID_CONFIG = 711001
    INTEGRATION_CONNECTION_FAILED = 711002
    INTEGRATION_AUTH_FAILED = 711003
    INTEGRATION_API_LIMIT_EXCEEDED = 711004
    INTEGRATION_DATA_SYNC_FAILED = 711005

    # ========== 监控服务错误码 (8xxxxx) ==========
    # 监控通用错误 (80xxxx)
    MONITOR_INTERNAL_ERROR = 802001
    MONITOR_SERVICE_UNAVAILABLE = 802002
    
    # 监控API错误 (81xxxx)
    MONITOR_METRIC_NOT_FOUND = 811001
    MONITOR_INVALID_TIME_RANGE = 811002
    MONITOR_QUERY_FAILED = 811003
    MONITOR_ALERT_CONFIG_INVALID = 811004

    # ========== 通用错误码 (9xxxxx) ==========
    # 通用系统错误 (90xxxx)
    SYSTEM_INTERNAL_ERROR = 902001
    SYSTEM_MAINTENANCE_MODE = 902002
    SYSTEM_OVERLOADED = 902003
    
    # 通用配置错误 (96xxxx)
    CONFIG_NOT_FOUND = 961001
    CONFIG_INVALID = 961002
    CONFIG_LOAD_FAILED = 961003
    
    # 通用网络错误 (93xxxx)
    NETWORK_TIMEOUT = 933001
    NETWORK_CONNECTION_REFUSED = 933002
    NETWORK_HOST_UNREACHABLE = 933003


class ErrorInfo:
    """错误信息类"""
    
    def __init__(self, code: ErrorCode, message: str, description: str, 
                 http_status: int, category: str, service: str):
        self.code = code
        self.message = message
        self.description = description
        self.http_status = http_status
        self.category = category
        self.service = service
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "code": int(self.code),
            "message": self.message,
            "description": self.description,
            "http_status": self.http_status,
            "category": self.category,
            "service": self.service
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def __str__(self) -> str:
        return f"[{self.code}] {self.message}: {self.description}"


# 错误码信息映射
ERROR_INFO_MAP: Dict[ErrorCode, ErrorInfo] = {
    ErrorCode.SUCCESS: ErrorInfo(ErrorCode.SUCCESS, "Success", "操作成功", 200, "Success", "Common"),
    
    # Gateway错误
    ErrorCode.GATEWAY_INTERNAL_ERROR: ErrorInfo(ErrorCode.GATEWAY_INTERNAL_ERROR, "Gateway Internal Error", "网关内部错误", 500, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.GATEWAY_SERVICE_UNAVAILABLE, "Gateway Service Unavailable", "网关服务不可用", 503, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_TIMEOUT: ErrorInfo(ErrorCode.GATEWAY_TIMEOUT, "Gateway Timeout", "网关超时", 504, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_INVALID_REQUEST: ErrorInfo(ErrorCode.GATEWAY_INVALID_REQUEST, "Invalid Request", "无效请求", 400, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_MISSING_PARAMETER: ErrorInfo(ErrorCode.GATEWAY_MISSING_PARAMETER, "Missing Parameter", "缺少参数", 400, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_INVALID_PARAMETER: ErrorInfo(ErrorCode.GATEWAY_INVALID_PARAMETER, "Invalid Parameter", "参数无效", 400, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_REQUEST_TOO_LARGE: ErrorInfo(ErrorCode.GATEWAY_REQUEST_TOO_LARGE, "Request Too Large", "请求体过大", 413, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_RATE_LIMIT_EXCEEDED: ErrorInfo(ErrorCode.GATEWAY_RATE_LIMIT_EXCEEDED, "Rate Limit Exceeded", "请求频率超限", 429, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_NETWORK_ERROR: ErrorInfo(ErrorCode.GATEWAY_NETWORK_ERROR, "Network Error", "网络错误", 502, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_CONNECTION_FAILED: ErrorInfo(ErrorCode.GATEWAY_CONNECTION_FAILED, "Connection Failed", "连接失败", 502, "Gateway", "Gateway"),
    ErrorCode.GATEWAY_DNS_RESOLVE_FAILED: ErrorInfo(ErrorCode.GATEWAY_DNS_RESOLVE_FAILED, "DNS Resolve Failed", "DNS解析失败", 502, "Gateway", "Gateway"),
    
    # 认证错误
    ErrorCode.AUTH_INTERNAL_ERROR: ErrorInfo(ErrorCode.AUTH_INTERNAL_ERROR, "Auth Internal Error", "认证服务内部错误", 500, "Auth", "Auth"),
    ErrorCode.AUTH_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.AUTH_SERVICE_UNAVAILABLE, "Auth Service Unavailable", "认证服务不可用", 503, "Auth", "Auth"),
    ErrorCode.AUTH_INVALID_CREDENTIALS: ErrorInfo(ErrorCode.AUTH_INVALID_CREDENTIALS, "Invalid Credentials", "无效凭证", 401, "Auth", "Auth"),
    ErrorCode.AUTH_TOKEN_EXPIRED: ErrorInfo(ErrorCode.AUTH_TOKEN_EXPIRED, "Token Expired", "Token过期", 401, "Auth", "Auth"),
    ErrorCode.AUTH_TOKEN_INVALID: ErrorInfo(ErrorCode.AUTH_TOKEN_INVALID, "Token Invalid", "Token无效", 401, "Auth", "Auth"),
    ErrorCode.AUTH_PERMISSION_DENIED: ErrorInfo(ErrorCode.AUTH_PERMISSION_DENIED, "Permission Denied", "权限不足", 403, "Auth", "Auth"),
    ErrorCode.AUTH_USER_NOT_FOUND: ErrorInfo(ErrorCode.AUTH_USER_NOT_FOUND, "User Not Found", "用户不存在", 404, "Auth", "Auth"),
    ErrorCode.AUTH_USER_DISABLED: ErrorInfo(ErrorCode.AUTH_USER_DISABLED, "User Disabled", "用户已禁用", 403, "Auth", "Auth"),
    ErrorCode.AUTH_SECURITY_VIOLATION: ErrorInfo(ErrorCode.AUTH_SECURITY_VIOLATION, "Security Violation", "安全违规", 403, "Auth", "Auth"),
    ErrorCode.AUTH_BRUTE_FORCE_DETECTED: ErrorInfo(ErrorCode.AUTH_BRUTE_FORCE_DETECTED, "Brute Force Detected", "检测到暴力破解", 429, "Auth", "Auth"),
    ErrorCode.AUTH_SUSPICIOUS_ACTIVITY: ErrorInfo(ErrorCode.AUTH_SUSPICIOUS_ACTIVITY, "Suspicious Activity", "可疑活动", 403, "Auth", "Auth"),
    
    # 聊天错误
    ErrorCode.CHAT_INTERNAL_ERROR: ErrorInfo(ErrorCode.CHAT_INTERNAL_ERROR, "Chat Internal Error", "聊天服务内部错误", 500, "Chat", "Chat"),
    ErrorCode.CHAT_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.CHAT_SERVICE_UNAVAILABLE, "Chat Service Unavailable", "聊天服务不可用", 503, "Chat", "Chat"),
    ErrorCode.CHAT_INVALID_MESSAGE: ErrorInfo(ErrorCode.CHAT_INVALID_MESSAGE, "Invalid Message", "无效消息", 400, "Chat", "Chat"),
    ErrorCode.CHAT_MESSAGE_TOO_LONG: ErrorInfo(ErrorCode.CHAT_MESSAGE_TOO_LONG, "Message Too Long", "消息过长", 400, "Chat", "Chat"),
    ErrorCode.CHAT_SESSION_NOT_FOUND: ErrorInfo(ErrorCode.CHAT_SESSION_NOT_FOUND, "Session Not Found", "会话不存在", 404, "Chat", "Chat"),
    ErrorCode.CHAT_SESSION_EXPIRED: ErrorInfo(ErrorCode.CHAT_SESSION_EXPIRED, "Session Expired", "会话过期", 410, "Chat", "Chat"),
    ErrorCode.CHAT_CONTEXT_LIMIT_EXCEEDED: ErrorInfo(ErrorCode.CHAT_CONTEXT_LIMIT_EXCEEDED, "Context Limit Exceeded", "上下文长度超限", 400, "Chat", "Chat"),
    ErrorCode.CHAT_RESPONSE_TIMEOUT: ErrorInfo(ErrorCode.CHAT_RESPONSE_TIMEOUT, "Response Timeout", "响应超时", 408, "Chat", "Chat"),
    ErrorCode.CHAT_QUEUE_FULL: ErrorInfo(ErrorCode.CHAT_QUEUE_FULL, "Queue Full", "队列已满", 503, "Chat", "Chat"),
    ErrorCode.CHAT_CONCURRENCY_LIMIT: ErrorInfo(ErrorCode.CHAT_CONCURRENCY_LIMIT, "Concurrency Limit", "并发限制", 429, "Chat", "Chat"),
    
    # 语音错误
    ErrorCode.VOICE_INTERNAL_ERROR: ErrorInfo(ErrorCode.VOICE_INTERNAL_ERROR, "Voice Internal Error", "语音服务内部错误", 500, "Voice", "Voice"),
    ErrorCode.VOICE_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.VOICE_SERVICE_UNAVAILABLE, "Voice Service Unavailable", "语音服务不可用", 503, "Voice", "Voice"),
    ErrorCode.VOICE_INVALID_FORMAT: ErrorInfo(ErrorCode.VOICE_INVALID_FORMAT, "Invalid Audio Format", "音频格式无效", 400, "Voice", "Voice"),
    ErrorCode.VOICE_FILE_TOO_LARGE: ErrorInfo(ErrorCode.VOICE_FILE_TOO_LARGE, "Audio File Too Large", "音频文件过大", 413, "Voice", "Voice"),
    ErrorCode.VOICE_PROCESSING_FAILED: ErrorInfo(ErrorCode.VOICE_PROCESSING_FAILED, "Audio Processing Failed", "音频处理失败", 500, "Voice", "Voice"),
    ErrorCode.VOICE_ASR_FAILED: ErrorInfo(ErrorCode.VOICE_ASR_FAILED, "ASR Failed", "语音识别失败", 500, "Voice", "Voice"),
    ErrorCode.VOICE_TTS_FAILED: ErrorInfo(ErrorCode.VOICE_TTS_FAILED, "TTS Failed", "语音合成失败", 500, "Voice", "Voice"),
    ErrorCode.VOICE_EMOTION_ANALYSIS_FAILED: ErrorInfo(ErrorCode.VOICE_EMOTION_ANALYSIS_FAILED, "Emotion Analysis Failed", "情感分析失败", 500, "Voice", "Voice"),
    ErrorCode.VOICE_FILE_NOT_FOUND: ErrorInfo(ErrorCode.VOICE_FILE_NOT_FOUND, "Audio File Not Found", "音频文件不存在", 404, "Voice", "Voice"),
    ErrorCode.VOICE_FILE_CORRUPTED: ErrorInfo(ErrorCode.VOICE_FILE_CORRUPTED, "Audio File Corrupted", "音频文件损坏", 400, "Voice", "Voice"),
    ErrorCode.VOICE_STORAGE_FAILED: ErrorInfo(ErrorCode.VOICE_STORAGE_FAILED, "Audio Storage Failed", "音频存储失败", 500, "Voice", "Voice"),
    
    # RAG错误
    ErrorCode.RAG_INTERNAL_ERROR: ErrorInfo(ErrorCode.RAG_INTERNAL_ERROR, "RAG Internal Error", "RAG服务内部错误", 500, "RAG", "RAG"),
    ErrorCode.RAG_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.RAG_SERVICE_UNAVAILABLE, "RAG Service Unavailable", "RAG服务不可用", 503, "RAG", "RAG"),
    ErrorCode.RAG_INVALID_QUERY: ErrorInfo(ErrorCode.RAG_INVALID_QUERY, "Invalid Query", "无效查询", 400, "RAG", "RAG"),
    ErrorCode.RAG_DOCUMENT_NOT_FOUND: ErrorInfo(ErrorCode.RAG_DOCUMENT_NOT_FOUND, "Document Not Found", "文档不存在", 404, "RAG", "RAG"),
    ErrorCode.RAG_INDEXING_FAILED: ErrorInfo(ErrorCode.RAG_INDEXING_FAILED, "Indexing Failed", "索引失败", 500, "RAG", "RAG"),
    ErrorCode.RAG_RETRIEVAL_FAILED: ErrorInfo(ErrorCode.RAG_RETRIEVAL_FAILED, "Retrieval Failed", "检索失败", 500, "RAG", "RAG"),
    ErrorCode.RAG_EMBEDDING_FAILED: ErrorInfo(ErrorCode.RAG_EMBEDDING_FAILED, "Embedding Failed", "向量化失败", 500, "RAG", "RAG"),
    ErrorCode.RAG_RERANKING_FAILED: ErrorInfo(ErrorCode.RAG_RERANKING_FAILED, "Reranking Failed", "重排序失败", 500, "RAG", "RAG"),
    ErrorCode.RAG_VECTOR_DB_ERROR: ErrorInfo(ErrorCode.RAG_VECTOR_DB_ERROR, "Vector DB Error", "向量数据库错误", 500, "RAG", "RAG"),
    ErrorCode.RAG_VECTOR_DB_CONNECTION_FAILED: ErrorInfo(ErrorCode.RAG_VECTOR_DB_CONNECTION_FAILED, "Vector DB Connection Failed", "向量数据库连接失败", 503, "RAG", "RAG"),
    ErrorCode.RAG_COLLECTION_NOT_FOUND: ErrorInfo(ErrorCode.RAG_COLLECTION_NOT_FOUND, "Collection Not Found", "集合不存在", 404, "RAG", "RAG"),
    
    # 存储错误
    ErrorCode.STORAGE_INTERNAL_ERROR: ErrorInfo(ErrorCode.STORAGE_INTERNAL_ERROR, "Storage Internal Error", "存储服务内部错误", 500, "Storage", "Storage"),
    ErrorCode.STORAGE_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.STORAGE_SERVICE_UNAVAILABLE, "Storage Service Unavailable", "存储服务不可用", 503, "Storage", "Storage"),
    ErrorCode.STORAGE_FILE_NOT_FOUND: ErrorInfo(ErrorCode.STORAGE_FILE_NOT_FOUND, "File Not Found", "文件不存在", 404, "Storage", "Storage"),
    ErrorCode.STORAGE_FILE_ACCESS_DENIED: ErrorInfo(ErrorCode.STORAGE_FILE_ACCESS_DENIED, "File Access Denied", "文件访问被拒绝", 403, "Storage", "Storage"),
    ErrorCode.STORAGE_FILE_CORRUPTED: ErrorInfo(ErrorCode.STORAGE_FILE_CORRUPTED, "File Corrupted", "文件损坏", 400, "Storage", "Storage"),
    ErrorCode.STORAGE_INSUFFICIENT_SPACE: ErrorInfo(ErrorCode.STORAGE_INSUFFICIENT_SPACE, "Insufficient Space", "存储空间不足", 507, "Storage", "Storage"),
    ErrorCode.STORAGE_UPLOAD_FAILED: ErrorInfo(ErrorCode.STORAGE_UPLOAD_FAILED, "Upload Failed", "文件上传失败", 500, "Storage", "Storage"),
    ErrorCode.STORAGE_DOWNLOAD_FAILED: ErrorInfo(ErrorCode.STORAGE_DOWNLOAD_FAILED, "Download Failed", "文件下载失败", 500, "Storage", "Storage"),
    
    # 集成错误
    ErrorCode.INTEGRATION_INTERNAL_ERROR: ErrorInfo(ErrorCode.INTEGRATION_INTERNAL_ERROR, "Integration Internal Error", "集成服务内部错误", 500, "Integration", "Integration"),
    ErrorCode.INTEGRATION_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.INTEGRATION_SERVICE_UNAVAILABLE, "Integration Service Unavailable", "集成服务不可用", 503, "Integration", "Integration"),
    ErrorCode.INTEGRATION_INVALID_CONFIG: ErrorInfo(ErrorCode.INTEGRATION_INVALID_CONFIG, "Invalid Config", "无效配置", 400, "Integration", "Integration"),
    ErrorCode.INTEGRATION_CONNECTION_FAILED: ErrorInfo(ErrorCode.INTEGRATION_CONNECTION_FAILED, "Connection Failed", "连接失败", 502, "Integration", "Integration"),
    ErrorCode.INTEGRATION_AUTH_FAILED: ErrorInfo(ErrorCode.INTEGRATION_AUTH_FAILED, "Auth Failed", "认证失败", 401, "Integration", "Integration"),
    ErrorCode.INTEGRATION_API_LIMIT_EXCEEDED: ErrorInfo(ErrorCode.INTEGRATION_API_LIMIT_EXCEEDED, "API Limit Exceeded", "API调用限制超出", 429, "Integration", "Integration"),
    ErrorCode.INTEGRATION_DATA_SYNC_FAILED: ErrorInfo(ErrorCode.INTEGRATION_DATA_SYNC_FAILED, "Data Sync Failed", "数据同步失败", 500, "Integration", "Integration"),
    
    # 监控错误
    ErrorCode.MONITOR_INTERNAL_ERROR: ErrorInfo(ErrorCode.MONITOR_INTERNAL_ERROR, "Monitor Internal Error", "监控服务内部错误", 500, "Monitor", "Monitor"),
    ErrorCode.MONITOR_SERVICE_UNAVAILABLE: ErrorInfo(ErrorCode.MONITOR_SERVICE_UNAVAILABLE, "Monitor Service Unavailable", "监控服务不可用", 503, "Monitor", "Monitor"),
    ErrorCode.MONITOR_METRIC_NOT_FOUND: ErrorInfo(ErrorCode.MONITOR_METRIC_NOT_FOUND, "Metric Not Found", "指标不存在", 404, "Monitor", "Monitor"),
    ErrorCode.MONITOR_INVALID_TIME_RANGE: ErrorInfo(ErrorCode.MONITOR_INVALID_TIME_RANGE, "Invalid Time Range", "无效时间范围", 400, "Monitor", "Monitor"),
    ErrorCode.MONITOR_QUERY_FAILED: ErrorInfo(ErrorCode.MONITOR_QUERY_FAILED, "Query Failed", "查询失败", 500, "Monitor", "Monitor"),
    ErrorCode.MONITOR_ALERT_CONFIG_INVALID: ErrorInfo(ErrorCode.MONITOR_ALERT_CONFIG_INVALID, "Alert Config Invalid", "告警配置无效", 400, "Monitor", "Monitor"),
    
    # 通用错误
    ErrorCode.SYSTEM_INTERNAL_ERROR: ErrorInfo(ErrorCode.SYSTEM_INTERNAL_ERROR, "System Internal Error", "系统内部错误", 500, "System", "Common"),
    ErrorCode.SYSTEM_MAINTENANCE_MODE: ErrorInfo(ErrorCode.SYSTEM_MAINTENANCE_MODE, "System Maintenance Mode", "系统维护模式", 503, "System", "Common"),
    ErrorCode.SYSTEM_OVERLOADED: ErrorInfo(ErrorCode.SYSTEM_OVERLOADED, "System Overloaded", "系统过载", 503, "System", "Common"),
    ErrorCode.CONFIG_NOT_FOUND: ErrorInfo(ErrorCode.CONFIG_NOT_FOUND, "Config Not Found", "配置不存在", 404, "Config", "Common"),
    ErrorCode.CONFIG_INVALID: ErrorInfo(ErrorCode.CONFIG_INVALID, "Config Invalid", "配置无效", 400, "Config", "Common"),
    ErrorCode.CONFIG_LOAD_FAILED: ErrorInfo(ErrorCode.CONFIG_LOAD_FAILED, "Config Load Failed", "配置加载失败", 500, "Config", "Common"),
    ErrorCode.NETWORK_TIMEOUT: ErrorInfo(ErrorCode.NETWORK_TIMEOUT, "Network Timeout", "网络超时", 408, "Network", "Common"),
    ErrorCode.NETWORK_CONNECTION_REFUSED: ErrorInfo(ErrorCode.NETWORK_CONNECTION_REFUSED, "Connection Refused", "连接被拒绝", 502, "Network", "Common"),
    ErrorCode.NETWORK_HOST_UNREACHABLE: ErrorInfo(ErrorCode.NETWORK_HOST_UNREACHABLE, "Host Unreachable", "主机不可达", 502, "Network", "Common"),
}


def get_error_info(code: ErrorCode) -> ErrorInfo:
    """获取错误信息"""
    return ERROR_INFO_MAP.get(code, ErrorInfo(
        code, "Unknown Error", f"未知错误码: {code}", 500, "Unknown", "Unknown"
    ))


class VoiceHelperError(Exception):
    """VoiceHelper自定义异常"""
    
    def __init__(self, code: ErrorCode, message: str = None, details: Dict[str, Any] = None):
        self.code = code
        self.error_info = get_error_info(code)
        self.message = message or self.error_info.message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = self.error_info.to_dict()
        if self.message != self.error_info.message:
            result["custom_message"] = self.message
        if self.details:
            result["details"] = self.details
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @property
    def http_status(self) -> int:
        """获取HTTP状态码"""
        return self.error_info.http_status
