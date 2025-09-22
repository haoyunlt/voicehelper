/**
 * VoiceHelper 前端日志系统
 * 提供结构化日志记录，包含网络信息和性能指标
 */

import { ErrorCode, VoiceHelperError, getErrorInfo } from './errors';

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical',
}

export enum LogType {
  STARTUP = 'startup',
  PAGE_VIEW = 'page_view',
  USER_ACTION = 'user_action',
  API_REQUEST = 'api_request',
  API_RESPONSE = 'api_response',
  ERROR = 'error',
  DEBUG = 'debug',
  PERFORMANCE = 'performance',
  SECURITY = 'security',
  BUSINESS = 'business',
  SYSTEM = 'system',
}

export interface NetworkInfo {
  userAgent: string;
  url: string;
  referrer: string;
  language: string;
  platform: string;
  cookieEnabled: boolean;
  onLine: boolean;
  connectionType?: string;
  ip?: string;
  sessionId?: string;
  userId?: string;
}

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  type: LogType;
  service: string;
  module: string;
  message: string;
  errorCode?: ErrorCode;
  network?: NetworkInfo;
  context?: Record<string, any>;
  stack?: string;
  durationMs?: number;
  url?: string;
  statusCode?: number;
  performance?: {
    loadTime?: number;
    renderTime?: number;
    memoryUsage?: number;
  };
}

class VoiceHelperLogger {
  private service: string;
  private module: string;
  private sessionId: string;
  private userId?: string;
  private baseContext: Record<string, any>;

  constructor(service: string = 'voicehelper-frontend', module: string = '') {
    this.service = service;
    this.module = module;
    this.sessionId = this.generateSessionId();
    this.baseContext = {};
    
    // 初始化时记录启动日志
    this.startup('前端应用初始化', {
      service: this.service,
      module: this.module,
      sessionId: this.sessionId,
    });
  }

  private generateSessionId(): string {
    return `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getNetworkInfo(): NetworkInfo {
    const nav = typeof navigator !== 'undefined' ? navigator : {} as any;
    const loc = typeof location !== 'undefined' ? location : {} as any;
    const doc = typeof document !== 'undefined' ? document : {} as any;

    return {
      userAgent: nav.userAgent || '',
      url: loc.href || '',
      referrer: doc.referrer || '',
      language: nav.language || '',
      platform: nav.platform || '',
      cookieEnabled: nav.cookieEnabled || false,
      onLine: nav.onLine !== undefined ? nav.onLine : true,
      connectionType: (nav as any).connection?.effectiveType || undefined,
      sessionId: this.sessionId,
      userId: this.userId,
    };
  }

  private getPerformanceInfo(): LogEntry['performance'] {
    if (typeof performance === 'undefined') return undefined;

    const timing = performance.timing;
    const memory = (performance as any).memory;

    return {
      loadTime: timing ? timing.loadEventEnd - timing.navigationStart : undefined,
      renderTime: timing ? timing.domContentLoadedEventEnd - timing.domContentLoadedEventStart : undefined,
      memoryUsage: memory ? memory.usedJSHeapSize : undefined,
    };
  }

  private buildLogEntry(
    level: LogLevel,
    type: LogType,
    message: string,
    options: {
      errorCode?: ErrorCode;
      context?: Record<string, any>;
      durationMs?: number;
      url?: string;
      statusCode?: number;
      includeStack?: boolean;
      includePerformance?: boolean;
    } = {}
  ): LogEntry {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      type,
      service: this.service,
      module: this.module,
      message,
      network: this.getNetworkInfo(),
      context: { ...this.baseContext, ...options.context },
    };

    if (options.errorCode) {
      entry.errorCode = options.errorCode;
    }

    if (options.durationMs !== undefined) {
      entry.durationMs = options.durationMs;
    }

    if (options.url) {
      entry.url = options.url;
    }

    if (options.statusCode) {
      entry.statusCode = options.statusCode;
    }

    if (options.includeStack || level === LogLevel.ERROR || level === LogLevel.CRITICAL) {
      entry.stack = new Error().stack;
    }

    if (options.includePerformance || type === LogType.PERFORMANCE) {
      entry.performance = this.getPerformanceInfo();
    }

    return entry;
  }

  private log(entry: LogEntry) {
    const logMessage = JSON.stringify(entry);
    
    // 根据日志级别选择输出方式
    switch (entry.level) {
      case LogLevel.DEBUG:
        console.debug(logMessage);
        break;
      case LogLevel.INFO:
        console.info(logMessage);
        break;
      case LogLevel.WARNING:
        console.warn(logMessage);
        break;
      case LogLevel.ERROR:
      case LogLevel.CRITICAL:
        console.error(logMessage);
        break;
      default:
        console.log(logMessage);
    }

    // 发送到远程日志服务 (可选)
    this.sendToRemote(entry);
  }

  private async sendToRemote(entry: LogEntry) {
    // 只在生产环境或配置了远程日志服务时发送
    const logEndpoint = process.env.NEXT_PUBLIC_LOG_ENDPOINT;
    if (!logEndpoint || process.env.NODE_ENV !== 'production') {
      return;
    }

    try {
      await fetch(logEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(entry),
      });
    } catch (error) {
      // 静默处理远程日志发送失败，避免影响主流程
      console.warn('Failed to send log to remote service:', error);
    }
  }

  // 基础日志方法
  debug(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.DEBUG, LogType.DEBUG, message, { context });
    this.log(entry);
  }

  info(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.SYSTEM, message, { context });
    this.log(entry);
  }

  warning(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.WARNING, LogType.SYSTEM, message, { context });
    this.log(entry);
  }

  error(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, { 
      context, 
      includeStack: true 
    });
    this.log(entry);
  }

  critical(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.CRITICAL, LogType.ERROR, message, { 
      context, 
      includeStack: true 
    });
    this.log(entry);
  }

  // 特定类型日志方法
  startup(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.STARTUP, message, { 
      context,
      includePerformance: true 
    });
    this.log(entry);
  }

  pageView(path: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.PAGE_VIEW, `Page view: ${path}`, {
      context: { ...context, path },
      url: typeof location !== 'undefined' ? location.href : path,
      includePerformance: true,
    });
    this.log(entry);
  }

  userAction(action: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.USER_ACTION, `User action: ${action}`, {
      context: { ...context, action },
    });
    this.log(entry);
  }

  apiRequest(method: string, url: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.API_REQUEST, `${method} ${url}`, {
      context: { ...context, method, url },
      url,
    });
    this.log(entry);
  }

  apiResponse(
    method: string, 
    url: string, 
    statusCode: number, 
    durationMs: number, 
    context?: Record<string, any>
  ) {
    const level = statusCode >= 500 ? LogLevel.ERROR : statusCode >= 400 ? LogLevel.WARNING : LogLevel.INFO;
    const entry = this.buildLogEntry(level, LogType.API_RESPONSE, `${method} ${url} - ${statusCode}`, {
      context: { ...context, method, url },
      url,
      statusCode,
      durationMs,
    });
    this.log(entry);
  }

  errorWithCode(errorCode: ErrorCode, message: string, context?: Record<string, any>) {
    const errorInfo = getErrorInfo(errorCode);
    const entry = this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, {
      errorCode,
      context: { ...context, errorInfo },
      includeStack: true,
    });
    this.log(entry);
  }

  performance(operation: string, durationMs: number, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.PERFORMANCE, `Performance: ${operation}`, {
      context: { ...context, operation },
      durationMs,
      includePerformance: true,
    });
    this.log(entry);
  }

  security(event: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.WARNING, LogType.SECURITY, `Security Event: ${event}`, {
      context: { ...context, event },
    });
    this.log(entry);
  }

  business(event: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.BUSINESS, `Business Event: ${event}`, {
      context: { ...context, event },
    });
    this.log(entry);
  }

  exception(message: string, error: Error, context?: Record<string, any>) {
    let errorCode: ErrorCode | undefined;
    if (error instanceof VoiceHelperError) {
      errorCode = error.code;
    }

    const entry = this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, {
      errorCode,
      context: {
        ...context,
        errorName: error.name,
        errorMessage: error.message,
        errorStack: error.stack,
      },
      includeStack: true,
    });
    this.log(entry);
  }

  // 上下文管理
  setUserId(userId: string) {
    this.userId = userId;
  }

  setContext(context: Record<string, any>) {
    this.baseContext = { ...this.baseContext, ...context };
  }

  withModule(module: string): VoiceHelperLogger {
    return new VoiceHelperLogger(this.service, module);
  }

  withContext(context: Record<string, any>): VoiceHelperLogger {
    const newLogger = new VoiceHelperLogger(this.service, this.module);
    newLogger.baseContext = { ...this.baseContext, ...context };
    newLogger.userId = this.userId;
    newLogger.sessionId = this.sessionId;
    return newLogger;
  }
}

// 全局日志器实例
let defaultLogger: VoiceHelperLogger | null = null;

export function initLogger(service: string = 'voicehelper-frontend'): VoiceHelperLogger {
  defaultLogger = new VoiceHelperLogger(service);
  return defaultLogger;
}

export function getLogger(module: string = ''): VoiceHelperLogger {
  if (!defaultLogger) {
    defaultLogger = new VoiceHelperLogger();
  }
  
  if (module) {
    return defaultLogger.withModule(module);
  }
  
  return defaultLogger;
}

// 便利函数
export function debug(message: string, context?: Record<string, any>) {
  getLogger().debug(message, context);
}

export function info(message: string, context?: Record<string, any>) {
  getLogger().info(message, context);
}

export function warning(message: string, context?: Record<string, any>) {
  getLogger().warning(message, context);
}

export function error(message: string, context?: Record<string, any>) {
  getLogger().error(message, context);
}

export function critical(message: string, context?: Record<string, any>) {
  getLogger().critical(message, context);
}

export function startup(message: string, context?: Record<string, any>) {
  getLogger().startup(message, context);
}

export function pageView(path: string, context?: Record<string, any>) {
  getLogger().pageView(path, context);
}

export function userAction(action: string, context?: Record<string, any>) {
  getLogger().userAction(action, context);
}

export function errorWithCode(errorCode: ErrorCode, message: string, context?: Record<string, any>) {
  getLogger().errorWithCode(errorCode, message, context);
}

export function performance(operation: string, durationMs: number, context?: Record<string, any>) {
  getLogger().performance(operation, durationMs, context);
}

export function security(event: string, context?: Record<string, any>) {
  getLogger().security(event, context);
}

export function business(event: string, context?: Record<string, any>) {
  getLogger().business(event, context);
}

export function exception(message: string, error: Error, context?: Record<string, any>) {
  getLogger().exception(message, error, context);
}

// React Hook for logging
export function useLogger(module?: string) {
  return module ? getLogger(module) : getLogger();
}

// API 请求拦截器
export function createApiLogger() {
  return {
    request: (method: string, url: string, context?: Record<string, any>) => {
      getLogger('api').apiRequest(method, url, context);
      return Date.now(); // 返回开始时间
    },
    response: (
      method: string, 
      url: string, 
      statusCode: number, 
      startTime: number, 
      context?: Record<string, any>
    ) => {
      const durationMs = Date.now() - startTime;
      getLogger('api').apiResponse(method, url, statusCode, durationMs, context);
    },
  };
}

export default VoiceHelperLogger;
