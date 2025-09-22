/**
 * VoiceHelper JavaScript SDK 日志系统
 */

import { ErrorCode, VoiceHelperSDKError, getErrorInfo } from './errors';

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical',
}

export enum LogType {
  STARTUP = 'startup',
  SHUTDOWN = 'shutdown',
  API_REQUEST = 'api_request',
  API_RESPONSE = 'api_response',
  WEBSOCKET = 'websocket',
  STORAGE = 'storage',
  AUTHENTICATION = 'authentication',
  ERROR = 'error',
  DEBUG = 'debug',
  PERFORMANCE = 'performance',
  SECURITY = 'security',
  BUSINESS = 'business',
  SYSTEM = 'system',
}

interface BrowserInfo {
  userAgent: string;
  language: string;
  platform: string;
  cookieEnabled: boolean;
  onLine: boolean;
  screenWidth: number;
  screenHeight: number;
  colorDepth: number;
  pixelRatio: number;
  timezone: string;
  url: string;
  referrer: string;
}

interface NetworkInfo {
  url?: string;
  method?: string;
  host?: string;
  port?: number;
  protocol?: string;
  requestId?: string;
}

interface PerformanceInfo {
  memoryUsage?: any;
  timing?: any;
  navigation?: any;
  connection?: any;
}

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  type: LogType;
  service: string;
  module: string;
  message: string;
  errorCode?: ErrorCode;
  browser?: BrowserInfo;
  network?: NetworkInfo;
  performance?: PerformanceInfo;
  context?: Record<string, any>;
  stack?: string;
  durationMs?: number;
  apiEndpoint?: string;
  requestId?: string;
  sessionId?: string;
  userId?: string;
}

class VoiceHelperSDKLogger {
  private service: string;
  private module: string;
  private baseContext: Record<string, any> = {};
  private sessionId: string;
  private userId?: string;

  constructor(service: string = 'voicehelper-javascript-sdk', module: string = '') {
    this.service = service;
    this.module = module;
    this.sessionId = this.generateSessionId();
    
    // 初始化时记录启动日志
    this.startup('JavaScript SDK日志系统初始化', {
      service: this.service,
      module: this.module,
      sessionId: this.sessionId,
    });
  }

  private generateSessionId(): string {
    return `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getBrowserInfo(): BrowserInfo {
    return {
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
      cookieEnabled: navigator.cookieEnabled,
      onLine: navigator.onLine,
      screenWidth: screen.width,
      screenHeight: screen.height,
      colorDepth: screen.colorDepth,
      pixelRatio: window.devicePixelRatio || 1,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      url: window.location.href,
      referrer: document.referrer,
    };
  }

  private getPerformanceInfo(): PerformanceInfo | undefined {
    if (typeof performance === 'undefined') return undefined;

    const info: PerformanceInfo = {};

    // 内存信息
    if ('memory' in performance) {
      info.memoryUsage = (performance as any).memory;
    }

    // 时间信息
    if (performance.timing) {
      info.timing = {
        navigationStart: performance.timing.navigationStart,
        loadEventEnd: performance.timing.loadEventEnd,
        domContentLoadedEventEnd: performance.timing.domContentLoadedEventEnd,
      };
    }

    // 导航信息
    if (performance.navigation) {
      info.navigation = {
        type: performance.navigation.type,
        redirectCount: performance.navigation.redirectCount,
      };
    }

    // 网络连接信息
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      info.connection = {
        effectiveType: connection?.effectiveType,
        downlink: connection?.downlink,
        rtt: connection?.rtt,
      };
    }

    return info;
  }

  private buildLogEntry(
    level: LogLevel,
    type: LogType,
    message: string,
    options: {
      errorCode?: ErrorCode;
      context?: Record<string, any>;
      durationMs?: number;
      apiEndpoint?: string;
      requestId?: string;
      includeStack?: boolean;
      includeBrowser?: boolean;
      includePerformance?: boolean;
      networkInfo?: NetworkInfo;
    } = {}
  ): LogEntry {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      type,
      service: this.service,
      module: this.module,
      message,
      sessionId: this.sessionId,
      userId: this.userId,
      context: { ...this.baseContext, ...options.context },
    };

    if (options.errorCode !== undefined) {
      entry.errorCode = options.errorCode;
    }

    if (options.durationMs !== undefined) {
      entry.durationMs = options.durationMs;
    }

    if (options.apiEndpoint) {
      entry.apiEndpoint = options.apiEndpoint;
    }

    if (options.requestId) {
      entry.requestId = options.requestId;
    }

    if (options.includeStack || level === LogLevel.ERROR || level === LogLevel.CRITICAL) {
      entry.stack = new Error().stack;
    }

    if (options.includeBrowser || type === LogType.STARTUP || type === LogType.SYSTEM) {
      entry.browser = this.getBrowserInfo();
    }

    if (options.includePerformance || type === LogType.PERFORMANCE) {
      entry.performance = this.getPerformanceInfo();
    }

    if (options.networkInfo) {
      entry.network = options.networkInfo;
    }

    return entry;
  }

  private log(entry: LogEntry) {
    const logMessage = JSON.stringify(entry);

    // 控制台输出
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

    // 存储到本地存储
    this.storeToLocalStorage(entry);

    // 发送到远程服务
    this.sendToRemote(entry);
  }

  private storeToLocalStorage(entry: LogEntry) {
    try {
      const storageKey = `voicehelper_logs_${new Date().toISOString().split('T')[0]}`;
      const existingLogs = JSON.parse(localStorage.getItem(storageKey) || '[]');
      
      existingLogs.push(entry);
      
      // 限制每天最多存储1000条日志
      if (existingLogs.length > 1000) {
        existingLogs.splice(0, existingLogs.length - 1000);
      }
      
      localStorage.setItem(storageKey, JSON.stringify(existingLogs));
    } catch (error) {
      console.warn('存储日志到本地存储失败:', error);
    }
  }

  private async sendToRemote(entry: LogEntry) {
    // 只发送重要日志到远程服务
    if (entry.level === LogLevel.ERROR || entry.level === LogLevel.CRITICAL || entry.type === LogType.STARTUP) {
      try {
        const logEndpoint = process.env.VOICEHELPER_LOG_ENDPOINT;
        if (logEndpoint) {
          await fetch(logEndpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(entry),
          });
        }
      } catch (error) {
        console.warn('发送日志到远程服务失败:', error);
      }
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
      includeStack: true,
      includeBrowser: true 
    });
    this.log(entry);
  }

  critical(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.CRITICAL, LogType.ERROR, message, { 
      context, 
      includeStack: true,
      includeBrowser: true 
    });
    this.log(entry);
  }

  // 特定类型日志方法
  startup(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.STARTUP, message, { 
      context,
      includeBrowser: true,
      includePerformance: true 
    });
    this.log(entry);
  }

  shutdown(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.SHUTDOWN, message, { 
      context,
      includePerformance: true 
    });
    this.log(entry);
  }

  apiRequest(method: string, url: string, requestId?: string, context?: Record<string, any>) {
    const message = `API Request: ${method} ${url}`;
    const networkInfo: NetworkInfo = { url, method, requestId };
    const entry = this.buildLogEntry(LogLevel.DEBUG, LogType.API_REQUEST, message, {
      context,
      apiEndpoint: url,
      requestId,
      networkInfo
    });
    this.log(entry);
  }

  apiResponse(
    method: string, 
    url: string, 
    statusCode: number, 
    durationMs: number, 
    requestId?: string,
    context?: Record<string, any>
  ) {
    const message = `API Response: ${method} ${url} - ${statusCode}`;
    const level = statusCode >= 500 ? LogLevel.ERROR : statusCode >= 400 ? LogLevel.WARNING : LogLevel.DEBUG;
    const networkInfo: NetworkInfo = { url, method, requestId };
    
    const mergedContext = { statusCode, ...context };
    
    const entry = this.buildLogEntry(level, LogType.API_RESPONSE, message, {
      context: mergedContext,
      durationMs,
      apiEndpoint: url,
      requestId,
      networkInfo
    });
    this.log(entry);
  }

  websocket(action: string, url: string, context?: Record<string, any>) {
    const message = `WebSocket ${action}: ${url}`;
    const networkInfo: NetworkInfo = { url };
    const entry = this.buildLogEntry(LogLevel.DEBUG, LogType.WEBSOCKET, message, {
      context: { action, ...context },
      networkInfo
    });
    this.log(entry);
  }

  storage(operation: string, key: string, success: boolean, context?: Record<string, any>) {
    const message = `Storage ${operation}: ${key} - ${success ? 'success' : 'failed'}`;
    const level = success ? LogLevel.DEBUG : LogLevel.WARNING;
    const entry = this.buildLogEntry(level, LogType.STORAGE, message, {
      context: { operation, key, success, ...context }
    });
    this.log(entry);
  }

  authentication(action: string, success: boolean, context?: Record<string, any>) {
    const message = `Authentication ${action}: ${success ? 'success' : 'failed'}`;
    const level = success ? LogLevel.INFO : LogLevel.WARNING;
    const entry = this.buildLogEntry(level, LogType.AUTHENTICATION, message, {
      context: { action, success, ...context }
    });
    this.log(entry);
  }

  errorWithCode(errorCode: ErrorCode, message: string, context?: Record<string, any>) {
    const errorInfo = getErrorInfo(errorCode);
    const mergedContext = {
      ...context,
      errorInfo: {
        code: errorInfo.code,
        message: errorInfo.message,
        description: errorInfo.description,
        category: errorInfo.category,
        service: errorInfo.service
      }
    };
    const entry = this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, {
      errorCode,
      context: mergedContext,
      includeStack: true,
      includeBrowser: true
    });
    this.log(entry);
  }

  performance(operation: string, durationMs: number, context?: Record<string, any>) {
    const message = `Performance: ${operation}`;
    const mergedContext = { operation, ...context };
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.PERFORMANCE, message, {
      context: mergedContext,
      durationMs,
      includePerformance: true
    });
    this.log(entry);
  }

  security(event: string, context?: Record<string, any>) {
    const message = `Security Event: ${event}`;
    const mergedContext = { event, ...context };
    const entry = this.buildLogEntry(LogLevel.WARNING, LogType.SECURITY, message, {
      context: mergedContext,
      includeBrowser: true
    });
    this.log(entry);
  }

  business(event: string, context?: Record<string, any>) {
    const message = `Business Event: ${event}`;
    const mergedContext = { event, ...context };
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.BUSINESS, message, {
      context: mergedContext
    });
    this.log(entry);
  }

  exception(message: string, error: Error, context?: Record<string, any>) {
    let errorCode: ErrorCode | undefined;
    if (error instanceof VoiceHelperSDKError) {
      errorCode = error.code;
    }

    const mergedContext = {
      ...context,
      errorName: error.name,
      errorMessage: error.message,
      errorStack: error.stack
    };

    const entry = this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, {
      errorCode,
      context: mergedContext,
      includeStack: true,
      includeBrowser: true
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

  withModule(module: string): VoiceHelperSDKLogger {
    const newLogger = new VoiceHelperSDKLogger(this.service, module);
    newLogger.baseContext = { ...this.baseContext };
    newLogger.userId = this.userId;
    newLogger.sessionId = this.sessionId;
    return newLogger;
  }

  withContext(context: Record<string, any>): VoiceHelperSDKLogger {
    const newLogger = new VoiceHelperSDKLogger(this.service, this.module);
    newLogger.baseContext = { ...this.baseContext, ...context };
    newLogger.userId = this.userId;
    newLogger.sessionId = this.sessionId;
    return newLogger;
  }

  // 日志管理
  getStoredLogs(date?: string): LogEntry[] {
    try {
      const storageKey = `voicehelper_logs_${date || new Date().toISOString().split('T')[0]}`;
      return JSON.parse(localStorage.getItem(storageKey) || '[]');
    } catch (error) {
      console.error('获取存储的日志失败:', error);
      return [];
    }
  }

  clearStoredLogs(daysToKeep: number = 7) {
    try {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
      
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith('voicehelper_logs_')) {
          const dateStr = key.replace('voicehelper_logs_', '');
          const logDate = new Date(dateStr);
          
          if (logDate < cutoffDate) {
            localStorage.removeItem(key);
            this.info('删除过期日志', { logKey: key, age: daysToKeep });
          }
        }
      }
    } catch (error) {
      console.error('清理过期日志失败:', error);
    }
  }
}

// 全局日志器实例
let defaultLogger: VoiceHelperSDKLogger | null = null;

export function initLogger(service: string = 'voicehelper-javascript-sdk'): VoiceHelperSDKLogger {
  defaultLogger = new VoiceHelperSDKLogger(service);
  return defaultLogger;
}

export function getLogger(module: string = ''): VoiceHelperSDKLogger {
  if (!defaultLogger) {
    defaultLogger = new VoiceHelperSDKLogger();
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

export function shutdown(message: string, context?: Record<string, any>) {
  getLogger().shutdown(message, context);
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

export default VoiceHelperSDKLogger;
