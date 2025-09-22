/**
 * VoiceHelper 微信小程序错误码系统
 * 与其他平台保持一致的错误码体系
 */

// 错误码枚举
const ErrorCode = {
  // 成功码
  SUCCESS: 0,

  // ========== Gateway服务错误码 (1xxxxx) ==========
  GATEWAY_INTERNAL_ERROR: 102001,
  GATEWAY_SERVICE_UNAVAILABLE: 102002,
  GATEWAY_TIMEOUT: 102003,
  GATEWAY_INVALID_REQUEST: 111001,
  GATEWAY_MISSING_PARAMETER: 111002,
  GATEWAY_INVALID_PARAMETER: 111003,
  GATEWAY_REQUEST_TOO_LARGE: 111004,
  GATEWAY_RATE_LIMIT_EXCEEDED: 111005,
  GATEWAY_NETWORK_ERROR: 133001,
  GATEWAY_CONNECTION_FAILED: 133002,
  GATEWAY_DNS_RESOLVE_FAILED: 133003,

  // ========== 认证服务错误码 (2xxxxx) ==========
  AUTH_INTERNAL_ERROR: 202001,
  AUTH_SERVICE_UNAVAILABLE: 202002,
  AUTH_INVALID_CREDENTIALS: 211001,
  AUTH_TOKEN_EXPIRED: 211002,
  AUTH_TOKEN_INVALID: 211003,
  AUTH_PERMISSION_DENIED: 211004,
  AUTH_USER_NOT_FOUND: 211005,
  AUTH_USER_DISABLED: 211006,
  AUTH_SECURITY_VIOLATION: 281001,
  AUTH_BRUTE_FORCE_DETECTED: 281002,
  AUTH_SUSPICIOUS_ACTIVITY: 281003,

  // ========== 聊天服务错误码 (3xxxxx) ==========
  CHAT_INTERNAL_ERROR: 302001,
  CHAT_SERVICE_UNAVAILABLE: 302002,
  CHAT_INVALID_MESSAGE: 311001,
  CHAT_MESSAGE_TOO_LONG: 311002,
  CHAT_SESSION_NOT_FOUND: 311003,
  CHAT_SESSION_EXPIRED: 311004,
  CHAT_CONTEXT_LIMIT_EXCEEDED: 311005,
  CHAT_RESPONSE_TIMEOUT: 371001,
  CHAT_QUEUE_FULL: 371002,
  CHAT_CONCURRENCY_LIMIT: 371003,

  // ========== 语音服务错误码 (4xxxxx) ==========
  VOICE_INTERNAL_ERROR: 402001,
  VOICE_SERVICE_UNAVAILABLE: 402002,
  VOICE_INVALID_FORMAT: 411001,
  VOICE_FILE_TOO_LARGE: 411002,
  VOICE_PROCESSING_FAILED: 411003,
  VOICE_ASR_FAILED: 411004,
  VOICE_TTS_FAILED: 411005,
  VOICE_EMOTION_ANALYSIS_FAILED: 411006,
  VOICE_FILE_NOT_FOUND: 461001,
  VOICE_FILE_CORRUPTED: 461002,
  VOICE_STORAGE_FAILED: 461003,

  // ========== RAG服务错误码 (5xxxxx) ==========
  RAG_INTERNAL_ERROR: 502001,
  RAG_SERVICE_UNAVAILABLE: 502002,
  RAG_INVALID_QUERY: 511001,
  RAG_DOCUMENT_NOT_FOUND: 511002,
  RAG_INDEXING_FAILED: 511003,
  RAG_RETRIEVAL_FAILED: 511004,
  RAG_EMBEDDING_FAILED: 511005,
  RAG_RERANKING_FAILED: 511006,
  RAG_VECTOR_DB_ERROR: 533001,
  RAG_VECTOR_DB_CONNECTION_FAILED: 533002,
  RAG_COLLECTION_NOT_FOUND: 533003,

  // ========== 存储服务错误码 (6xxxxx) ==========
  STORAGE_INTERNAL_ERROR: 602001,
  STORAGE_SERVICE_UNAVAILABLE: 602002,
  STORAGE_FILE_NOT_FOUND: 661001,
  STORAGE_FILE_ACCESS_DENIED: 661002,
  STORAGE_FILE_CORRUPTED: 661003,
  STORAGE_INSUFFICIENT_SPACE: 661004,
  STORAGE_UPLOAD_FAILED: 661005,
  STORAGE_DOWNLOAD_FAILED: 661006,

  // ========== 微信小程序特有错误码 (8xxxxx) ==========
  // 小程序通用错误 (80xxxx)
  MINIPROGRAM_INTERNAL_ERROR: 802001,
  MINIPROGRAM_INITIALIZATION_FAILED: 802002,
  MINIPROGRAM_UPDATE_FAILED: 802003,

  // 小程序API错误 (81xxxx)
  MINIPROGRAM_API_ERROR: 811001,
  MINIPROGRAM_NETWORK_ERROR: 811002,
  MINIPROGRAM_STORAGE_ERROR: 811003,
  MINIPROGRAM_LOCATION_ERROR: 811004,
  MINIPROGRAM_CAMERA_ERROR: 811005,
  MINIPROGRAM_RECORDER_ERROR: 811006,

  // 小程序权限错误 (82xxxx)
  MINIPROGRAM_PERMISSION_DENIED: 821001,
  MINIPROGRAM_MICROPHONE_PERMISSION_DENIED: 821002,
  MINIPROGRAM_CAMERA_PERMISSION_DENIED: 821003,
  MINIPROGRAM_LOCATION_PERMISSION_DENIED: 821004,
  MINIPROGRAM_ALBUM_PERMISSION_DENIED: 821005,

  // 小程序存储错误 (86xxxx)
  MINIPROGRAM_LOCALSTORAGE_ERROR: 861001,
  MINIPROGRAM_FILESTORAGE_ERROR: 861002,
  MINIPROGRAM_CACHE_ERROR: 861003,

  // 小程序支付错误 (84xxxx)
  MINIPROGRAM_PAYMENT_ERROR: 841001,
  MINIPROGRAM_PAYMENT_CANCELLED: 841002,
  MINIPROGRAM_PAYMENT_FAILED: 841003,

  // 小程序分享错误 (85xxxx)
  MINIPROGRAM_SHARE_ERROR: 851001,
  MINIPROGRAM_SHARE_CANCELLED: 851002,

  // ========== 通用错误码 (9xxxxx) ==========
  SYSTEM_INTERNAL_ERROR: 902001,
  SYSTEM_MAINTENANCE_MODE: 902002,
  SYSTEM_OVERLOADED: 902003,
  CONFIG_NOT_FOUND: 961001,
  CONFIG_INVALID: 961002,
  CONFIG_LOAD_FAILED: 961003,
  NETWORK_TIMEOUT: 933001,
  NETWORK_CONNECTION_REFUSED: 933002,
  NETWORK_HOST_UNREACHABLE: 933003
};

// 错误信息映射
const errorInfoMap = {
  [ErrorCode.SUCCESS]: {
    code: ErrorCode.SUCCESS,
    message: "Success",
    description: "操作成功",
    category: "Success",
    service: "Common"
  },

  // 小程序错误
  [ErrorCode.MINIPROGRAM_INTERNAL_ERROR]: {
    code: ErrorCode.MINIPROGRAM_INTERNAL_ERROR,
    message: "MiniProgram Internal Error",
    description: "小程序内部错误",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_INITIALIZATION_FAILED]: {
    code: ErrorCode.MINIPROGRAM_INITIALIZATION_FAILED,
    message: "MiniProgram Initialization Failed",
    description: "小程序初始化失败",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_API_ERROR]: {
    code: ErrorCode.MINIPROGRAM_API_ERROR,
    message: "MiniProgram API Error",
    description: "小程序API错误",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_NETWORK_ERROR]: {
    code: ErrorCode.MINIPROGRAM_NETWORK_ERROR,
    message: "MiniProgram Network Error",
    description: "小程序网络错误",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_PERMISSION_DENIED]: {
    code: ErrorCode.MINIPROGRAM_PERMISSION_DENIED,
    message: "Permission Denied",
    description: "权限被拒绝",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_MICROPHONE_PERMISSION_DENIED]: {
    code: ErrorCode.MINIPROGRAM_MICROPHONE_PERMISSION_DENIED,
    message: "Microphone Permission Denied",
    description: "麦克风权限被拒绝",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_LOCALSTORAGE_ERROR]: {
    code: ErrorCode.MINIPROGRAM_LOCALSTORAGE_ERROR,
    message: "LocalStorage Error",
    description: "本地存储错误",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_PAYMENT_ERROR]: {
    code: ErrorCode.MINIPROGRAM_PAYMENT_ERROR,
    message: "Payment Error",
    description: "支付错误",
    category: "MiniProgram",
    service: "MiniProgram"
  },
  [ErrorCode.MINIPROGRAM_SHARE_ERROR]: {
    code: ErrorCode.MINIPROGRAM_SHARE_ERROR,
    message: "Share Error",
    description: "分享错误",
    category: "MiniProgram",
    service: "MiniProgram"
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
  }
};

/**
 * 获取错误信息
 * @param {number} code 错误码
 * @returns {Object} 错误信息对象
 */
function getErrorInfo(code) {
  return errorInfoMap[code] || {
    code: code,
    message: "Unknown Error",
    description: `未知错误码: ${code}`,
    category: "Unknown",
    service: "Unknown"
  };
}

/**
 * VoiceHelper错误类
 */
class VoiceHelperError extends Error {
  constructor(code, message, details) {
    const errorInfo = getErrorInfo(code);
    super(message || errorInfo.message);
    
    this.code = code;
    this.errorInfo = errorInfo;
    this.details = details || {};
    this.name = 'VoiceHelperError';
  }

  toJSON() {
    return {
      ...this.errorInfo,
      customMessage: this.message !== this.errorInfo.message ? this.message : undefined,
      details: this.details
    };
  }
}

module.exports = {
  ErrorCode,
  getErrorInfo,
  VoiceHelperError
};
