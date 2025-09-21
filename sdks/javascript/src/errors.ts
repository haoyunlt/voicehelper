/**
 * SDK错误类定义
 */

/**
 * 基础Chatbot错误类
 */
export class ChatbotError extends Error {
  public readonly name = 'ChatbotError';

  constructor(message: string, public readonly cause?: Error) {
    super(message);
    
    // 确保错误堆栈正确显示
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

/**
 * API错误类
 */
export class APIError extends ChatbotError {
  public readonly name = 'APIError';

  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
    public readonly details?: Record<string, any>
  ) {
    super(message);
  }

  /**
   * 判断是否为客户端错误 (4xx)
   */
  get isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }

  /**
   * 判断是否为服务器错误 (5xx)
   */
  get isServerError(): boolean {
    return this.status >= 500;
  }

  /**
   * 判断是否为认证错误
   */
  get isAuthError(): boolean {
    return this.status === 401 || this.status === 403;
  }

  /**
   * 判断是否为限流错误
   */
  get isRateLimitError(): boolean {
    return this.status === 429;
  }
}

/**
 * 网络错误类
 */
export class NetworkError extends ChatbotError {
  public readonly name = 'NetworkError';

  constructor(message: string, cause?: Error) {
    super(message, cause);
  }
}

/**
 * 验证错误类
 */
export class ValidationError extends ChatbotError {
  public readonly name = 'ValidationError';

  constructor(
    message: string,
    public readonly field?: string,
    public readonly value?: any
  ) {
    super(message);
  }
}

/**
 * 配置错误类
 */
export class ConfigError extends ChatbotError {
  public readonly name = 'ConfigError';

  constructor(message: string) {
    super(message);
  }
}

/**
 * WebSocket错误类
 */
export class WebSocketError extends ChatbotError {
  public readonly name = 'WebSocketError';

  constructor(
    message: string,
    public readonly code?: number,
    public readonly reason?: string
  ) {
    super(message);
  }
}

/**
 * 流式处理错误类
 */
export class StreamError extends ChatbotError {
  public readonly name = 'StreamError';

  constructor(message: string, cause?: Error) {
    super(message, cause);
  }
}

/**
 * 文件处理错误类
 */
export class FileError extends ChatbotError {
  public readonly name = 'FileError';

  constructor(
    message: string,
    public readonly fileName?: string,
    public readonly fileSize?: number
  ) {
    super(message);
  }
}

/**
 * 错误工厂函数
 */
export class ErrorFactory {
  /**
   * 根据HTTP状态码创建相应的错误
   */
  static fromHTTPStatus(
    status: number,
    message: string,
    code?: string,
    details?: Record<string, any>
  ): APIError {
    return new APIError(message, status, code, details);
  }

  /**
   * 从axios错误创建ChatbotError
   */
  static fromAxiosError(error: any): ChatbotError {
    if (error.response) {
      // 服务器返回了错误响应
      return new APIError(
        error.response.data?.error || error.message,
        error.response.status,
        error.response.data?.code,
        error.response.data?.details
      );
    } else if (error.request) {
      // 请求发出但没有收到响应
      return new NetworkError('No response received from server', error);
    } else {
      // 请求配置错误
      return new ChatbotError('Request configuration error: ' + error.message, error);
    }
  }

  /**
   * 从WebSocket错误创建WebSocketError
   */
  static fromWebSocketError(error: any, code?: number, reason?: string): WebSocketError {
    return new WebSocketError(error.message || 'WebSocket error', code, reason);
  }

  /**
   * 验证文件大小
   */
  static validateFileSize(file: File, maxSize: number): void {
    if (file.size > maxSize) {
      throw new FileError(
        `File size ${file.size} exceeds maximum allowed size ${maxSize}`,
        file.name,
        file.size
      );
    }
  }

  /**
   * 验证文件类型
   */
  static validateFileType(file: File, allowedTypes: string[]): void {
    if (!allowedTypes.includes(file.type)) {
      throw new FileError(
        `File type ${file.type} is not allowed. Allowed types: ${allowedTypes.join(', ')}`,
        file.name
      );
    }
  }

  /**
   * 验证必需参数
   */
  static validateRequired(value: any, fieldName: string): void {
    if (value === undefined || value === null || value === '') {
      throw new ValidationError(`${fieldName} is required`, fieldName, value);
    }
  }

  /**
   * 验证字符串长度
   */
  static validateStringLength(
    value: string,
    fieldName: string,
    minLength?: number,
    maxLength?: number
  ): void {
    if (minLength !== undefined && value.length < minLength) {
      throw new ValidationError(
        `${fieldName} must be at least ${minLength} characters long`,
        fieldName,
        value
      );
    }
    if (maxLength !== undefined && value.length > maxLength) {
      throw new ValidationError(
        `${fieldName} must be no more than ${maxLength} characters long`,
        fieldName,
        value
      );
    }
  }

  /**
   * 验证数值范围
   */
  static validateNumberRange(
    value: number,
    fieldName: string,
    min?: number,
    max?: number
  ): void {
    if (min !== undefined && value < min) {
      throw new ValidationError(
        `${fieldName} must be at least ${min}`,
        fieldName,
        value
      );
    }
    if (max !== undefined && value > max) {
      throw new ValidationError(
        `${fieldName} must be no more than ${max}`,
        fieldName,
        value
      );
    }
  }
}

/**
 * 错误处理工具函数
 */
export class ErrorUtils {
  /**
   * 判断错误是否可重试
   */
  static isRetryable(error: Error): boolean {
    if (error instanceof APIError) {
      // 5xx服务器错误和429限流错误可重试
      return error.isServerError || error.isRateLimitError;
    }
    if (error instanceof NetworkError) {
      // 网络错误可重试
      return true;
    }
    return false;
  }

  /**
   * 获取错误的用户友好消息
   */
  static getUserFriendlyMessage(error: Error): string {
    if (error instanceof APIError) {
      switch (error.status) {
        case 400:
          return '请求参数有误，请检查输入内容';
        case 401:
          return '身份验证失败，请重新登录';
        case 403:
          return '权限不足，无法执行此操作';
        case 404:
          return '请求的资源不存在';
        case 429:
          return '请求过于频繁，请稍后再试';
        case 500:
          return '服务器内部错误，请稍后再试';
        case 502:
        case 503:
        case 504:
          return '服务暂时不可用，请稍后再试';
        default:
          return error.message;
      }
    }
    if (error instanceof NetworkError) {
      return '网络连接失败，请检查网络设置';
    }
    if (error instanceof ValidationError) {
      return error.message;
    }
    if (error instanceof FileError) {
      return error.message;
    }
    return '发生未知错误，请稍后再试';
  }

  /**
   * 记录错误日志
   */
  static logError(error: Error, context?: Record<string, any>): void {
    const errorInfo = {
      name: error.name,
      message: error.message,
      stack: error.stack,
      context,
    };

    if (error instanceof APIError) {
      Object.assign(errorInfo, {
        status: error.status,
        code: error.code,
        details: error.details,
      });
    }

    console.error('Chatbot SDK Error:', errorInfo);
  }
}
