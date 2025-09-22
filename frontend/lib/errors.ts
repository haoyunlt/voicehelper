/**
 * VoiceHelper 前端错误码系统
 * 与后端保持一致的错误码体系
 */

export enum ErrorCode {
  // 成功码
  SUCCESS = 0,

  // ========== Gateway服务错误码 (1xxxxx) ==========
  // Gateway通用错误 (10xxxx)
  GATEWAY_INTERNAL_ERROR = 102001,
  GATEWAY_SERVICE_UNAVAILABLE = 102002,
  GATEWAY_TIMEOUT = 102003,
  
  // Gateway API错误 (11xxxx)
  GATEWAY_INVALID_REQUEST = 111001,
  GATEWAY_MISSING_PARAMETER = 111002,
  GATEWAY_INVALID_PARAMETER = 111003,
  GATEWAY_REQUEST_TOO_LARGE = 111004,
  GATEWAY_RATE_LIMIT_EXCEEDED = 111005,
  
  // Gateway网络错误 (13xxxx)
  GATEWAY_NETWORK_ERROR = 133001,
  GATEWAY_CONNECTION_FAILED = 133002,
  GATEWAY_DNS_RESOLVE_FAILED = 133003,

  // ========== 认证服务错误码 (2xxxxx) ==========
  // 认证通用错误 (20xxxx)
  AUTH_INTERNAL_ERROR = 202001,
  AUTH_SERVICE_UNAVAILABLE = 202002,
  
  // 认证API错误 (21xxxx)
  AUTH_INVALID_CREDENTIALS = 211001,
  AUTH_TOKEN_EXPIRED = 211002,
  AUTH_TOKEN_INVALID = 211003,
  AUTH_PERMISSION_DENIED = 211004,
  AUTH_USER_NOT_FOUND = 211005,
  AUTH_USER_DISABLED = 211006,
  
  // 认证安全错误 (28xxxx)
  AUTH_SECURITY_VIOLATION = 281001,
  AUTH_BRUTE_FORCE_DETECTED = 281002,
  AUTH_SUSPICIOUS_ACTIVITY = 281003,

  // ========== 聊天服务错误码 (3xxxxx) ==========
  // 聊天通用错误 (30xxxx)
  CHAT_INTERNAL_ERROR = 302001,
  CHAT_SERVICE_UNAVAILABLE = 302002,
  
  // 聊天API错误 (31xxxx)
  CHAT_INVALID_MESSAGE = 311001,
  CHAT_MESSAGE_TOO_LONG = 311002,
  CHAT_SESSION_NOT_FOUND = 311003,
  CHAT_SESSION_EXPIRED = 311004,
  CHAT_CONTEXT_LIMIT_EXCEEDED = 311005,
  
  // 聊天性能错误 (37xxxx)
  CHAT_RESPONSE_TIMEOUT = 371001,
  CHAT_QUEUE_FULL = 371002,
  CHAT_CONCURRENCY_LIMIT = 371003,

  // ========== 语音服务错误码 (4xxxxx) ==========
  // 语音通用错误 (40xxxx)
  VOICE_INTERNAL_ERROR = 402001,
  VOICE_SERVICE_UNAVAILABLE = 402002,
  
  // 语音API错误 (41xxxx)
  VOICE_INVALID_FORMAT = 411001,
  VOICE_FILE_TOO_LARGE = 411002,
  VOICE_PROCESSING_FAILED = 411003,
  VOICE_ASR_FAILED = 411004,
  VOICE_TTS_FAILED = 411005,
  VOICE_EMOTION_ANALYSIS_FAILED = 411006,
  
  // 语音文件错误 (46xxxx)
  VOICE_FILE_NOT_FOUND = 461001,
  VOICE_FILE_CORRUPTED = 461002,
  VOICE_STORAGE_FAILED = 461003,

  // ========== RAG服务错误码 (5xxxxx) ==========
  // RAG通用错误 (50xxxx)
  RAG_INTERNAL_ERROR = 502001,
  RAG_SERVICE_UNAVAILABLE = 502002,
  
  // RAG API错误 (51xxxx)
  RAG_INVALID_QUERY = 511001,
  RAG_DOCUMENT_NOT_FOUND = 511002,
  RAG_INDEXING_FAILED = 511003,
  RAG_RETRIEVAL_FAILED = 511004,
  RAG_EMBEDDING_FAILED = 511005,
  RAG_RERANKING_FAILED = 511006,
  
  // RAG数据库错误 (53xxxx)
  RAG_VECTOR_DB_ERROR = 533001,
  RAG_VECTOR_DB_CONNECTION_FAILED = 533002,
  RAG_COLLECTION_NOT_FOUND = 533003,

  // ========== 存储服务错误码 (6xxxxx) ==========
  // 存储通用错误 (60xxxx)
  STORAGE_INTERNAL_ERROR = 602001,
  STORAGE_SERVICE_UNAVAILABLE = 602002,
  
  // 存储文件错误 (66xxxx)
  STORAGE_FILE_NOT_FOUND = 661001,
  STORAGE_FILE_ACCESS_DENIED = 661002,
  STORAGE_FILE_CORRUPTED = 661003,
  STORAGE_INSUFFICIENT_SPACE = 661004,
  STORAGE_UPLOAD_FAILED = 661005,
  STORAGE_DOWNLOAD_FAILED = 661006,

  // ========== 前端特有错误码 (8xxxxx) ==========
  // 前端通用错误 (80xxxx)
  FRONTEND_INTERNAL_ERROR = 802001,
  FRONTEND_RENDER_ERROR = 802002,
  FRONTEND_HYDRATION_ERROR = 802003,
  
  // 前端API错误 (81xxxx)
  FRONTEND_API_ERROR = 811001,
  FRONTEND_NETWORK_ERROR = 811002,
  FRONTEND_TIMEOUT_ERROR = 811003,
  FRONTEND_CORS_ERROR = 811004,
  
  // 前端UI错误 (82xxxx)
  FRONTEND_COMPONENT_ERROR = 821001,
  FRONTEND_ROUTING_ERROR = 821002,
  FRONTEND_STATE_ERROR = 821003,
  
  // 前端存储错误 (86xxxx)
  FRONTEND_LOCALSTORAGE_ERROR = 861001,
  FRONTEND_SESSIONSTORAGE_ERROR = 861002,
  FRONTEND_INDEXEDDB_ERROR = 861003,

  // ========== 通用错误码 (9xxxxx) ==========
  // 通用系统错误 (90xxxx)
  SYSTEM_INTERNAL_ERROR = 902001,
  SYSTEM_MAINTENANCE_MODE = 902002,
  SYSTEM_OVERLOADED = 902003,
  
  // 通用配置错误 (96xxxx)
  CONFIG_NOT_FOUND = 961001,
  CONFIG_INVALID = 961002,
  CONFIG_LOAD_FAILED = 961003,
  
  // 通用网络错误 (93xxxx)
  NETWORK_TIMEOUT = 933001,
  NETWORK_CONNECTION_REFUSED = 933002,
  NETWORK_HOST_UNREACHABLE = 933003,
}

export interface ErrorInfo {
  code: ErrorCode;
  message: string;
  description: string;
  httpStatus: number;
  category: string;
  service: string;
}

const errorInfoMap: Record<ErrorCode, ErrorInfo> = {
  [ErrorCode.SUCCESS]: {
    code: ErrorCode.SUCCESS,
    message: "Success",
    description: "操作成功",
    httpStatus: 200,
    category: "Success",
    service: "Common"
  },
  
  // Gateway错误
  [ErrorCode.GATEWAY_INTERNAL_ERROR]: {
    code: ErrorCode.GATEWAY_INTERNAL_ERROR,
    message: "Gateway Internal Error",
    description: "网关内部错误",
    httpStatus: 500,
    category: "Gateway",
    service: "Gateway"
  },
  [ErrorCode.GATEWAY_NETWORK_ERROR]: {
    code: ErrorCode.GATEWAY_NETWORK_ERROR,
    message: "Network Error",
    description: "网络错误",
    httpStatus: 502,
    category: "Gateway",
    service: "Gateway"
  },
  
  // 认证错误
  [ErrorCode.AUTH_TOKEN_EXPIRED]: {
    code: ErrorCode.AUTH_TOKEN_EXPIRED,
    message: "Token Expired",
    description: "Token过期",
    httpStatus: 401,
    category: "Auth",
    service: "Auth"
  },
  [ErrorCode.AUTH_PERMISSION_DENIED]: {
    code: ErrorCode.AUTH_PERMISSION_DENIED,
    message: "Permission Denied",
    description: "权限不足",
    httpStatus: 403,
    category: "Auth",
    service: "Auth"
  },
  
  // 聊天错误
  [ErrorCode.CHAT_SESSION_NOT_FOUND]: {
    code: ErrorCode.CHAT_SESSION_NOT_FOUND,
    message: "Session Not Found",
    description: "会话不存在",
    httpStatus: 404,
    category: "Chat",
    service: "Chat"
  },
  [ErrorCode.CHAT_RESPONSE_TIMEOUT]: {
    code: ErrorCode.CHAT_RESPONSE_TIMEOUT,
    message: "Response Timeout",
    description: "响应超时",
    httpStatus: 408,
    category: "Chat",
    service: "Chat"
  },
  
  // 语音错误
  [ErrorCode.VOICE_ASR_FAILED]: {
    code: ErrorCode.VOICE_ASR_FAILED,
    message: "ASR Failed",
    description: "语音识别失败",
    httpStatus: 500,
    category: "Voice",
    service: "Voice"
  },
  [ErrorCode.VOICE_TTS_FAILED]: {
    code: ErrorCode.VOICE_TTS_FAILED,
    message: "TTS Failed",
    description: "语音合成失败",
    httpStatus: 500,
    category: "Voice",
    service: "Voice"
  },
  
  // RAG错误
  [ErrorCode.RAG_RETRIEVAL_FAILED]: {
    code: ErrorCode.RAG_RETRIEVAL_FAILED,
    message: "Retrieval Failed",
    description: "检索失败",
    httpStatus: 500,
    category: "RAG",
    service: "RAG"
  },
  [ErrorCode.RAG_SERVICE_UNAVAILABLE]: {
    code: ErrorCode.RAG_SERVICE_UNAVAILABLE,
    message: "RAG Service Unavailable",
    description: "RAG服务不可用",
    httpStatus: 503,
    category: "RAG",
    service: "RAG"
  },
  
  // 前端错误
  [ErrorCode.FRONTEND_INTERNAL_ERROR]: {
    code: ErrorCode.FRONTEND_INTERNAL_ERROR,
    message: "Frontend Internal Error",
    description: "前端内部错误",
    httpStatus: 500,
    category: "Frontend",
    service: "Frontend"
  },
  [ErrorCode.FRONTEND_API_ERROR]: {
    code: ErrorCode.FRONTEND_API_ERROR,
    message: "Frontend API Error",
    description: "前端API错误",
    httpStatus: 500,
    category: "Frontend",
    service: "Frontend"
  },
  [ErrorCode.FRONTEND_NETWORK_ERROR]: {
    code: ErrorCode.FRONTEND_NETWORK_ERROR,
    message: "Frontend Network Error",
    description: "前端网络错误",
    httpStatus: 502,
    category: "Frontend",
    service: "Frontend"
  },
  [ErrorCode.FRONTEND_COMPONENT_ERROR]: {
    code: ErrorCode.FRONTEND_COMPONENT_ERROR,
    message: "Component Error",
    description: "组件错误",
    httpStatus: 500,
    category: "Frontend",
    service: "Frontend"
  },
  
  // 系统错误
  [ErrorCode.SYSTEM_INTERNAL_ERROR]: {
    code: ErrorCode.SYSTEM_INTERNAL_ERROR,
    message: "System Internal Error",
    description: "系统内部错误",
    httpStatus: 500,
    category: "System",
    service: "Common"
  },
  [ErrorCode.NETWORK_TIMEOUT]: {
    code: ErrorCode.NETWORK_TIMEOUT,
    message: "Network Timeout",
    description: "网络超时",
    httpStatus: 408,
    category: "Network",
    service: "Common"
  },
};

export function getErrorInfo(code: ErrorCode): ErrorInfo {
  return errorInfoMap[code] || {
    code,
    message: "Unknown Error",
    description: `未知错误码: ${code}`,
    httpStatus: 500,
    category: "Unknown",
    service: "Unknown"
  };
}

export class VoiceHelperError extends Error {
  public readonly code: ErrorCode;
  public readonly errorInfo: ErrorInfo;
  public readonly details?: Record<string, any>;

  constructor(code: ErrorCode, message?: string, details?: Record<string, any>) {
    const errorInfo = getErrorInfo(code);
    super(message || errorInfo.message);
    
    this.code = code;
    this.errorInfo = errorInfo;
    this.details = details;
    this.name = 'VoiceHelperError';
  }

  toJSON() {
    return {
      ...this.errorInfo,
      customMessage: this.message !== this.errorInfo.message ? this.message : undefined,
      details: this.details,
    };
  }

  get httpStatus(): number {
    return this.errorInfo.httpStatus;
  }
}
