/**
 * VoiceHelper 桌面应用错误码系统
 * 与其他平台保持一致的错误码体系
 */

export enum ErrorCode {
  // 成功码
  SUCCESS = 0,

  // ========== Gateway服务错误码 (1xxxxx) ==========
  GATEWAY_INTERNAL_ERROR = 102001,
  GATEWAY_SERVICE_UNAVAILABLE = 102002,
  GATEWAY_TIMEOUT = 102003,
  GATEWAY_INVALID_REQUEST = 111001,
  GATEWAY_MISSING_PARAMETER = 111002,
  GATEWAY_INVALID_PARAMETER = 111003,
  GATEWAY_REQUEST_TOO_LARGE = 111004,
  GATEWAY_RATE_LIMIT_EXCEEDED = 111005,
  GATEWAY_NETWORK_ERROR = 133001,
  GATEWAY_CONNECTION_FAILED = 133002,
  GATEWAY_DNS_RESOLVE_FAILED = 133003,

  // ========== 认证服务错误码 (2xxxxx) ==========
  AUTH_INTERNAL_ERROR = 202001,
  AUTH_SERVICE_UNAVAILABLE = 202002,
  AUTH_INVALID_CREDENTIALS = 211001,
  AUTH_TOKEN_EXPIRED = 211002,
  AUTH_TOKEN_INVALID = 211003,
  AUTH_PERMISSION_DENIED = 211004,
  AUTH_USER_NOT_FOUND = 211005,
  AUTH_USER_DISABLED = 211006,
  AUTH_SECURITY_VIOLATION = 281001,
  AUTH_BRUTE_FORCE_DETECTED = 281002,
  AUTH_SUSPICIOUS_ACTIVITY = 281003,

  // ========== 聊天服务错误码 (3xxxxx) ==========
  CHAT_INTERNAL_ERROR = 302001,
  CHAT_SERVICE_UNAVAILABLE = 302002,
  CHAT_INVALID_MESSAGE = 311001,
  CHAT_MESSAGE_TOO_LONG = 311002,
  CHAT_SESSION_NOT_FOUND = 311003,
  CHAT_SESSION_EXPIRED = 311004,
  CHAT_CONTEXT_LIMIT_EXCEEDED = 311005,
  CHAT_RESPONSE_TIMEOUT = 371001,
  CHAT_QUEUE_FULL = 371002,
  CHAT_CONCURRENCY_LIMIT = 371003,

  // ========== 语音服务错误码 (4xxxxx) ==========
  VOICE_INTERNAL_ERROR = 402001,
  VOICE_SERVICE_UNAVAILABLE = 402002,
  VOICE_INVALID_FORMAT = 411001,
  VOICE_FILE_TOO_LARGE = 411002,
  VOICE_PROCESSING_FAILED = 411003,
  VOICE_ASR_FAILED = 411004,
  VOICE_TTS_FAILED = 411005,
  VOICE_EMOTION_ANALYSIS_FAILED = 411006,
  VOICE_FILE_NOT_FOUND = 461001,
  VOICE_FILE_CORRUPTED = 461002,
  VOICE_STORAGE_FAILED = 461003,

  // ========== RAG服务错误码 (5xxxxx) ==========
  RAG_INTERNAL_ERROR = 502001,
  RAG_SERVICE_UNAVAILABLE = 502002,
  RAG_INVALID_QUERY = 511001,
  RAG_DOCUMENT_NOT_FOUND = 511002,
  RAG_INDEXING_FAILED = 511003,
  RAG_RETRIEVAL_FAILED = 511004,
  RAG_EMBEDDING_FAILED = 511005,
  RAG_RERANKING_FAILED = 511006,
  RAG_VECTOR_DB_ERROR = 533001,
  RAG_VECTOR_DB_CONNECTION_FAILED = 533002,
  RAG_COLLECTION_NOT_FOUND = 533003,

  // ========== 存储服务错误码 (6xxxxx) ==========
  STORAGE_INTERNAL_ERROR = 602001,
  STORAGE_SERVICE_UNAVAILABLE = 602002,
  STORAGE_FILE_NOT_FOUND = 661001,
  STORAGE_FILE_ACCESS_DENIED = 661002,
  STORAGE_FILE_CORRUPTED = 661003,
  STORAGE_INSUFFICIENT_SPACE = 661004,
  STORAGE_UPLOAD_FAILED = 661005,
  STORAGE_DOWNLOAD_FAILED = 661006,

  // ========== 桌面应用特有错误码 (7xxxxx) ==========
  // 桌面应用通用错误 (70xxxx)
  DESKTOP_INTERNAL_ERROR = 702001,
  DESKTOP_INITIALIZATION_FAILED = 702002,
  DESKTOP_UPDATE_FAILED = 702003,
  
  // 桌面应用窗口错误 (71xxxx)
  DESKTOP_WINDOW_CREATE_FAILED = 711001,
  DESKTOP_WINDOW_LOAD_FAILED = 711002,
  DESKTOP_WINDOW_CRASH = 711003,
  
  // 桌面应用IPC错误 (72xxxx)
  DESKTOP_IPC_ERROR = 721001,
  DESKTOP_IPC_TIMEOUT = 721002,
  DESKTOP_IPC_INVALID_MESSAGE = 721003,
  
  // 桌面应用文件系统错误 (76xxxx)
  DESKTOP_FILE_READ_ERROR = 761001,
  DESKTOP_FILE_WRITE_ERROR = 761002,
  DESKTOP_FILE_PERMISSION_ERROR = 761003,
  DESKTOP_DIRECTORY_CREATE_ERROR = 761004,
  
  // 桌面应用系统错误 (77xxxx)
  DESKTOP_SYSTEM_ERROR = 771001,
  DESKTOP_PERMISSION_ERROR = 771002,
  DESKTOP_RESOURCE_ERROR = 771003,
  
  // 桌面应用网络错误 (73xxxx)
  DESKTOP_NETWORK_ERROR = 731001,
  DESKTOP_PROXY_ERROR = 731002,
  DESKTOP_SSL_ERROR = 731003,

  // ========== 通用错误码 (9xxxxx) ==========
  SYSTEM_INTERNAL_ERROR = 902001,
  SYSTEM_MAINTENANCE_MODE = 902002,
  SYSTEM_OVERLOADED = 902003,
  CONFIG_NOT_FOUND = 961001,
  CONFIG_INVALID = 961002,
  CONFIG_LOAD_FAILED = 961003,
  NETWORK_TIMEOUT = 933001,
  NETWORK_CONNECTION_REFUSED = 933002,
  NETWORK_HOST_UNREACHABLE = 933003,
}

export interface ErrorInfo {
  code: ErrorCode;
  message: string;
  description: string;
  category: string;
  service: string;
}

const errorInfoMap: Record<ErrorCode, ErrorInfo> = {
  [ErrorCode.SUCCESS]: {
    code: ErrorCode.SUCCESS,
    message: "Success",
    description: "操作成功",
    category: "Success",
    service: "Common"
  },
  
  // 桌面应用错误
  [ErrorCode.DESKTOP_INTERNAL_ERROR]: {
    code: ErrorCode.DESKTOP_INTERNAL_ERROR,
    message: "Desktop Internal Error",
    description: "桌面应用内部错误",
    category: "Desktop",
    service: "Desktop"
  },
  [ErrorCode.DESKTOP_INITIALIZATION_FAILED]: {
    code: ErrorCode.DESKTOP_INITIALIZATION_FAILED,
    message: "Desktop Initialization Failed",
    description: "桌面应用初始化失败",
    category: "Desktop",
    service: "Desktop"
  },
  [ErrorCode.DESKTOP_WINDOW_CREATE_FAILED]: {
    code: ErrorCode.DESKTOP_WINDOW_CREATE_FAILED,
    message: "Window Create Failed",
    description: "窗口创建失败",
    category: "Desktop",
    service: "Desktop"
  },
  [ErrorCode.DESKTOP_IPC_ERROR]: {
    code: ErrorCode.DESKTOP_IPC_ERROR,
    message: "IPC Error",
    description: "进程间通信错误",
    category: "Desktop",
    service: "Desktop"
  },
  [ErrorCode.DESKTOP_FILE_READ_ERROR]: {
    code: ErrorCode.DESKTOP_FILE_READ_ERROR,
    message: "File Read Error",
    description: "文件读取错误",
    category: "Desktop",
    service: "Desktop"
  },
  [ErrorCode.DESKTOP_NETWORK_ERROR]: {
    code: ErrorCode.DESKTOP_NETWORK_ERROR,
    message: "Desktop Network Error",
    description: "桌面应用网络错误",
    category: "Desktop",
    service: "Desktop"
  },
  
  // 通用错误
  [ErrorCode.SYSTEM_INTERNAL_ERROR]: {
    code: ErrorCode.SYSTEM_INTERNAL_ERROR,
    message: "System Internal Error",
    description: "系统内部错误",
    category: "System",
    service: "Common"
  },
  [ErrorCode.NETWORK_TIMEOUT]: {
    code: ErrorCode.NETWORK_TIMEOUT,
    message: "Network Timeout",
    description: "网络超时",
    category: "Network",
    service: "Common"
  },
};

export function getErrorInfo(code: ErrorCode): ErrorInfo {
  return errorInfoMap[code] || {
    code,
    message: "Unknown Error",
    description: `未知错误码: ${code}`,
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
}
