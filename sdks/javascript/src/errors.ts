/**
 * VoiceHelper JavaScript SDK 错误码系统
 */

export enum ErrorCode {
  SUCCESS = 0,
  
  // SDK错误
  SDK_INTERNAL_ERROR = 702001,
  SDK_INITIALIZATION_FAILED = 702002,
  SDK_NETWORK_ERROR = 731001,
  SDK_AUTH_ERROR = 721001,
  SDK_INVALID_PARAMETER = 711001,
  SDK_BROWSER_NOT_SUPPORTED = 741001,
  SDK_WEBSOCKET_ERROR = 751001,
  SDK_LOCALSTORAGE_ERROR = 761001,
  
  // 通用错误
  SYSTEM_INTERNAL_ERROR = 902001,
  NETWORK_TIMEOUT = 933001,
  
  // 语音服务错误
  VOICE_ASR_FAILED = 411004,
  VOICE_TTS_FAILED = 411005,
  
  // RAG服务错误
  RAG_RETRIEVAL_FAILED = 511004,
  
  // 认证错误
  AUTH_TOKEN_EXPIRED = 211002,
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
  [ErrorCode.SDK_INTERNAL_ERROR]: {
    code: ErrorCode.SDK_INTERNAL_ERROR,
    message: "SDK Internal Error",
    description: "SDK内部错误",
    category: "SDK",
    service: "SDK"
  },
  [ErrorCode.SDK_NETWORK_ERROR]: {
    code: ErrorCode.SDK_NETWORK_ERROR,
    message: "SDK Network Error",
    description: "SDK网络错误",
    category: "SDK",
    service: "SDK"
  },
  [ErrorCode.SDK_AUTH_ERROR]: {
    code: ErrorCode.SDK_AUTH_ERROR,
    message: "SDK Authentication Error",
    description: "SDK认证错误",
    category: "SDK",
    service: "SDK"
  },
  [ErrorCode.SDK_INVALID_PARAMETER]: {
    code: ErrorCode.SDK_INVALID_PARAMETER,
    message: "SDK Invalid Parameter",
    description: "SDK参数无效",
    category: "SDK",
    service: "SDK"
  },
  [ErrorCode.SDK_BROWSER_NOT_SUPPORTED]: {
    code: ErrorCode.SDK_BROWSER_NOT_SUPPORTED,
    message: "Browser Not Supported",
    description: "浏览器不支持",
    category: "SDK",
    service: "SDK"
  },
  [ErrorCode.SDK_WEBSOCKET_ERROR]: {
    code: ErrorCode.SDK_WEBSOCKET_ERROR,
    message: "WebSocket Error",
    description: "WebSocket错误",
    category: "SDK",
    service: "SDK"
  },
  [ErrorCode.SDK_LOCALSTORAGE_ERROR]: {
    code: ErrorCode.SDK_LOCALSTORAGE_ERROR,
    message: "LocalStorage Error",
    description: "本地存储错误",
    category: "SDK",
    service: "SDK"
  },
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
  [ErrorCode.VOICE_ASR_FAILED]: {
    code: ErrorCode.VOICE_ASR_FAILED,
    message: "ASR Failed",
    description: "语音识别失败",
    category: "Voice",
    service: "Voice"
  },
  [ErrorCode.VOICE_TTS_FAILED]: {
    code: ErrorCode.VOICE_TTS_FAILED,
    message: "TTS Failed",
    description: "语音合成失败",
    category: "Voice",
    service: "Voice"
  },
  [ErrorCode.RAG_RETRIEVAL_FAILED]: {
    code: ErrorCode.RAG_RETRIEVAL_FAILED,
    message: "RAG Retrieval Failed",
    description: "RAG检索失败",
    category: "RAG",
    service: "RAG"
  },
  [ErrorCode.AUTH_TOKEN_EXPIRED]: {
    code: ErrorCode.AUTH_TOKEN_EXPIRED,
    message: "Token Expired",
    description: "Token过期",
    category: "Auth",
    service: "Auth"
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

export class VoiceHelperSDKError extends Error {
  public readonly code: ErrorCode;
  public readonly errorInfo: ErrorInfo;
  public readonly details?: Record<string, any>;

  constructor(code: ErrorCode, message?: string, details?: Record<string, any>) {
    const errorInfo = getErrorInfo(code);
    super(message || errorInfo.message);
    
    this.code = code;
    this.errorInfo = errorInfo;
    this.details = details;
    this.name = 'VoiceHelperSDKError';
  }

  toJSON() {
    return {
      ...this.errorInfo,
      customMessage: this.message !== this.errorInfo.message ? this.message : undefined,
      details: this.details,
    };
  }
}