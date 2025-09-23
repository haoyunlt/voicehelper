/**
 * VoiceHelper 桌面应用日志系统
 * 提供结构化日志记录，包含系统信息和性能指标
 */

import { app, ipcMain } from 'electron';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
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
  SHUTDOWN = 'shutdown',
  WINDOW = 'window',
  IPC = 'ipc',
  FILE_SYSTEM = 'file_system',
  NETWORK = 'network',
  ERROR = 'error',
  DEBUG = 'debug',
  PERFORMANCE = 'performance',
  SECURITY = 'security',
  BUSINESS = 'business',
  SYSTEM = 'system',
  UPDATE = 'update',
}

export interface SystemInfo {
  platform: string;
  arch: string;
  osVersion: string;
  nodeVersion: string;
  electronVersion: string;
  appVersion: string;
  cpuCount: number;
  totalMemory: number;
  freeMemory: number;
  userDataPath: string;
  execPath: string;
  pid: number;
  ppid?: number;
}

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  type: LogType;
  service: string;
  module: string;
  message: string;
  errorCode?: ErrorCode;
  system?: SystemInfo;
  context?: Record<string, any>;
  stack?: string;
  durationMs?: number;
  processId?: number;
  windowId?: number;
  ipcChannel?: string;
  filePath?: string;
  networkUrl?: string;
  performance?: {
    cpuUsage?: NodeJS.CpuUsage;
    memoryUsage?: NodeJS.MemoryUsage;
    uptime?: number;
  };
}

class DesktopLogger {
  private service: string;
  private module: string;
  private logFilePath: string;
  private baseContext: Record<string, any>;
  private logStream?: fs.WriteStream;

  constructor(service: string = 'voicehelper-desktop', module: string = '') {
    this.service = service;
    this.module = module;
    this.baseContext = {};
    
    // 设置日志文件路径
    const userDataPath = app.getPath('userData');
    const logsDir = path.join(userDataPath, 'logs');
    
    // 确保日志目录存在
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    
    const logFileName = `voicehelper-${new Date().toISOString().split('T')[0]}.log`;
    this.logFilePath = path.join(logsDir, logFileName);
    
    // 创建日志文件流
    this.initLogStream();
    
    // 初始化时记录启动日志
    this.startup('桌面应用日志系统初始化', {
      service: this.service,
      module: this.module,
      logFilePath: this.logFilePath,
    });
  }

  private initLogStream() {
    try {
      this.logStream = fs.createWriteStream(this.logFilePath, { flags: 'a' });
      
      this.logStream.on('error', (error) => {
        console.error('日志文件写入错误:', error);
      });
    } catch (error) {
      console.error('创建日志文件流失败:', error);
    }
  }

  private getSystemInfo(): SystemInfo {
    return {
      platform: os.platform(),
      arch: os.arch(),
      osVersion: os.release(),
      nodeVersion: process.version,
      electronVersion: process.versions.electron || 'unknown',
      appVersion: app.getVersion(),
      cpuCount: os.cpus().length,
      totalMemory: os.totalmem(),
      freeMemory: os.freemem(),
      userDataPath: app.getPath('userData'),
      execPath: process.execPath,
      pid: process.pid,
      ppid: process.ppid,
    };
  }

  private getPerformanceInfo(): LogEntry['performance'] {
    return {
      cpuUsage: process.cpuUsage(),
      memoryUsage: process.memoryUsage(),
      uptime: process.uptime(),
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
      processId?: number;
      windowId?: number;
      ipcChannel?: string;
      filePath?: string;
      networkUrl?: string;
      includeStack?: boolean;
      includeSystem?: boolean;
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
      context: { ...this.baseContext, ...options.context },
    };

    if (options.errorCode) {
      entry.errorCode = options.errorCode;
    }

    if (options.durationMs !== undefined) {
      entry.durationMs = options.durationMs;
    }

    if (options.processId) {
      entry.processId = options.processId;
    }

    if (options.windowId) {
      entry.windowId = options.windowId;
    }

    if (options.ipcChannel) {
      entry.ipcChannel = options.ipcChannel;
    }

    if (options.filePath) {
      entry.filePath = options.filePath;
    }

    if (options.networkUrl) {
      entry.networkUrl = options.networkUrl;
    }

    if (options.includeStack || level === LogLevel.ERROR || level === LogLevel.CRITICAL) {
      entry.stack = new Error().stack;
    }

    if (options.includeSystem || type === LogType.STARTUP || type === LogType.SYSTEM) {
      entry.system = this.getSystemInfo();
    }

    if (options.includePerformance || type === LogType.PERFORMANCE) {
      entry.performance = this.getPerformanceInfo();
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

    // 写入日志文件
    if (this.logStream) {
      this.logStream.write(logMessage + '\n');
    }

    // 发送到渲染进程 (如果需要)
    this.sendToRenderer(entry);
  }

  private sendToRenderer(entry: LogEntry) {
    // 只发送重要日志到渲染进程
    if (entry.level === LogLevel.ERROR || entry.level === LogLevel.CRITICAL || entry.type === LogType.STARTUP) {
      try {
        // 使用webContents.send发送到所有窗口
        const { BrowserWindow } = require('electron');
        const windows = BrowserWindow.getAllWindows();
        
        windows.forEach(window => {
          if (window && !window.isDestroyed()) {
            window.webContents.send('log-entry', entry);
          }
        });
      } catch (error) {
        console.error('发送日志到渲染进程失败:', error);
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
      includeSystem: true,
    });
    this.log(entry);
  }

  critical(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.CRITICAL, LogType.ERROR, message, { 
      context, 
      includeStack: true,
      includeSystem: true,
    });
    this.log(entry);
  }

  // 特定类型日志方法
  startup(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.STARTUP, message, { 
      context,
      includeSystem: true,
      includePerformance: true,
    });
    this.log(entry);
  }

  shutdown(message: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.SHUTDOWN, message, { 
      context,
      includeSystem: true,
      includePerformance: true,
    });
    this.log(entry);
    
    // 确保日志写入完成
    if (this.logStream) {
      this.logStream.end();
    }
  }

  window(action: string, windowId: number, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.WINDOW, `Window ${action}`, {
      context: { ...context, action },
      windowId,
    });
    this.log(entry);
  }

  ipc(channel: string, direction: 'send' | 'receive', context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.DEBUG, LogType.IPC, `IPC ${direction}: ${channel}`, {
      context: { ...context, direction },
      ipcChannel: channel,
    });
    this.log(entry);
  }

  fileSystem(operation: string, filePath: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.DEBUG, LogType.FILE_SYSTEM, `File ${operation}: ${filePath}`, {
      context: { ...context, operation },
      filePath,
    });
    this.log(entry);
  }

  network(method: string, url: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.DEBUG, LogType.NETWORK, `${method} ${url}`, {
      context: { ...context, method },
      networkUrl: url,
    });
    this.log(entry);
  }

  errorWithCode(errorCode: ErrorCode, message: string, context?: Record<string, any>) {
    const errorInfo = getErrorInfo(errorCode);
    const entry = this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, {
      errorCode,
      context: { ...context, errorInfo },
      includeStack: true,
      includeSystem: true,
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
      includeSystem: true,
    });
    this.log(entry);
  }

  business(event: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.BUSINESS, `Business Event: ${event}`, {
      context: { ...context, event },
    });
    this.log(entry);
  }

  update(event: string, context?: Record<string, any>) {
    const entry = this.buildLogEntry(LogLevel.INFO, LogType.UPDATE, `Update Event: ${event}`, {
      context: { ...context, event },
      includeSystem: true,
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
      includeSystem: true,
    });
    this.log(entry);
  }

  // 上下文管理
  setContext(context: Record<string, any>) {
    this.baseContext = { ...this.baseContext, ...context };
  }

  withModule(module: string): DesktopLogger {
    const newLogger = new DesktopLogger(this.service, module);
    newLogger.baseContext = { ...this.baseContext };
    return newLogger;
  }

  withContext(context: Record<string, any>): DesktopLogger {
    const newLogger = new DesktopLogger(this.service, this.module);
    newLogger.baseContext = { ...this.baseContext, ...context };
    return newLogger;
  }

  // 日志文件管理
  getLogFilePath(): string {
    return this.logFilePath;
  }

  async getLogFiles(): Promise<string[]> {
    try {
      const logsDir = path.dirname(this.logFilePath);
      const files = await fs.promises.readdir(logsDir);
      return files
        .filter(file => file.startsWith('voicehelper-') && file.endsWith('.log'))
        .map(file => path.join(logsDir, file))
        .sort();
    } catch (error) {
      this.exception('获取日志文件列表失败', error as Error);
      return [];
    }
  }

  async cleanOldLogs(daysToKeep: number = 7): Promise<void> {
    try {
      const logFiles = await this.getLogFiles();
      const cutoffTime = Date.now() - (daysToKeep * 24 * 60 * 60 * 1000);

      for (const logFile of logFiles) {
        const stats = await fs.promises.stat(logFile);
        if (stats.mtime.getTime() < cutoffTime) {
          await fs.promises.unlink(logFile);
          this.info('删除过期日志文件', { logFile, age: daysToKeep });
        }
      }
    } catch (error) {
      this.exception('清理过期日志失败', error as Error);
    }
  }
}

// 全局日志器实例
let defaultLogger: DesktopLogger | null = null;

export function initLogger(service: string = 'voicehelper-desktop'): DesktopLogger {
  defaultLogger = new DesktopLogger(service);
  
  // 设置IPC处理器
  setupIpcHandlers();
  
  return defaultLogger;
}

export function getLogger(module: string = ''): DesktopLogger {
  if (!defaultLogger) {
    defaultLogger = new DesktopLogger();
  }
  
  if (module) {
    return defaultLogger.withModule(module);
  }
  
  return defaultLogger;
}

// 设置IPC处理器
function setupIpcHandlers() {
  // 处理来自渲染进程的日志请求
  ipcMain.handle('log-from-renderer', (event, logData) => {
    const logger = getLogger('renderer');
    const { level, message, context } = logData;
    
    switch (level) {
      case 'debug':
        logger.debug(message, context);
        break;
      case 'info':
        logger.info(message, context);
        break;
      case 'warning':
        logger.warning(message, context);
        break;
      case 'error':
        logger.error(message, context);
        break;
      case 'critical':
        logger.critical(message, context);
        break;
      default:
        logger.info(message, context);
    }
  });

  // 获取日志文件路径
  ipcMain.handle('get-log-file-path', () => {
    return defaultLogger?.getLogFilePath() || '';
  });

  // 获取日志文件列表
  ipcMain.handle('get-log-files', async () => {
    return defaultLogger?.getLogFiles() || [];
  });

  // 清理过期日志
  ipcMain.handle('clean-old-logs', async (event, daysToKeep: number) => {
    return defaultLogger?.cleanOldLogs(daysToKeep);
  });
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

export default DesktopLogger;
