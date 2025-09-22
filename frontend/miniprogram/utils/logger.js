/**
 * VoiceHelper 微信小程序日志系统
 * 提供结构化日志记录，包含设备信息和性能指标
 */

const { ErrorCode, getErrorInfo, VoiceHelperError } = require('./errors.js');

// 日志级别
const LogLevel = {
  DEBUG: 'debug',
  INFO: 'info',
  WARNING: 'warning',
  ERROR: 'error',
  CRITICAL: 'critical'
};

// 日志类型
const LogType = {
  STARTUP: 'startup',
  SHUTDOWN: 'shutdown',
  PAGE: 'page',
  COMPONENT: 'component',
  API: 'api',
  NETWORK: 'network',
  STORAGE: 'storage',
  PERMISSION: 'permission',
  PAYMENT: 'payment',
  SHARE: 'share',
  LOCATION: 'location',
  CAMERA: 'camera',
  RECORDER: 'recorder',
  ERROR: 'error',
  DEBUG: 'debug',
  PERFORMANCE: 'performance',
  SECURITY: 'security',
  BUSINESS: 'business',
  SYSTEM: 'system'
};

/**
 * 获取设备信息
 */
function getDeviceInfo() {
  const systemInfo = wx.getSystemInfoSync();
  const accountInfo = wx.getAccountInfoSync();
  
  return {
    brand: systemInfo.brand,
    model: systemInfo.model,
    system: systemInfo.system,
    platform: systemInfo.platform,
    version: systemInfo.version,
    SDKVersion: systemInfo.SDKVersion,
    language: systemInfo.language,
    pixelRatio: systemInfo.pixelRatio,
    screenWidth: systemInfo.screenWidth,
    screenHeight: systemInfo.screenHeight,
    windowWidth: systemInfo.windowWidth,
    windowHeight: systemInfo.windowHeight,
    statusBarHeight: systemInfo.statusBarHeight,
    safeArea: systemInfo.safeArea,
    locationEnabled: systemInfo.locationEnabled,
    wifiEnabled: systemInfo.wifiEnabled,
    cameraAuthorized: systemInfo.cameraAuthorized,
    locationAuthorized: systemInfo.locationAuthorized,
    microphoneAuthorized: systemInfo.microphoneAuthorized,
    notificationAuthorized: systemInfo.notificationAuthorized,
    bluetoothEnabled: systemInfo.bluetoothEnabled,
    locationReducedAccuracy: systemInfo.locationReducedAccuracy,
    theme: systemInfo.theme,
    host: systemInfo.host,
    enableDebug: systemInfo.enableDebug,
    deviceOrientation: systemInfo.deviceOrientation,
    
    // 小程序信息
    appId: accountInfo.miniProgram.appId,
    envVersion: accountInfo.miniProgram.envVersion,
    version: accountInfo.miniProgram.version
  };
}

/**
 * 获取网络信息
 */
function getNetworkInfo() {
  return new Promise((resolve) => {
    wx.getNetworkType({
      success: (res) => {
        resolve({
          networkType: res.networkType,
          isConnected: res.networkType !== 'none'
        });
      },
      fail: () => {
        resolve({
          networkType: 'unknown',
          isConnected: false
        });
      }
    });
  });
}

/**
 * 获取位置信息
 */
function getLocationInfo() {
  return new Promise((resolve) => {
    wx.getLocation({
      type: 'wgs84',
      success: (res) => {
        resolve({
          latitude: res.latitude,
          longitude: res.longitude,
          speed: res.speed,
          accuracy: res.accuracy,
          altitude: res.altitude,
          verticalAccuracy: res.verticalAccuracy,
          horizontalAccuracy: res.horizontalAccuracy
        });
      },
      fail: () => {
        resolve(null);
      }
    });
  });
}

/**
 * 获取性能信息
 */
function getPerformanceInfo() {
  const performance = wx.getPerformance();
  const entries = performance.getEntries();
  
  return {
    navigationStart: performance.navigationStart,
    entries: entries.map(entry => ({
      name: entry.name,
      entryType: entry.entryType,
      startTime: entry.startTime,
      duration: entry.duration
    })),
    memoryUsage: wx.getMemoryInfo ? wx.getMemoryInfo() : null
  };
}

/**
 * VoiceHelper日志器类
 */
class VoiceHelperLogger {
  constructor(service = 'voicehelper-miniprogram', module = '') {
    this.service = service;
    this.module = module;
    this.baseContext = {};
    this.sessionId = this.generateSessionId();
    this.userId = null;
    
    // 初始化时记录启动日志
    this.startup('小程序日志系统初始化', {
      service: this.service,
      module: this.module,
      sessionId: this.sessionId
    });
  }

  generateSessionId() {
    return `sess_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  async buildLogEntry(level, type, message, options = {}) {
    const timestamp = new Date().toISOString();
    
    const entry = {
      timestamp,
      level,
      type,
      service: this.service,
      module: this.module,
      message,
      sessionId: this.sessionId,
      userId: this.userId,
      context: { ...this.baseContext, ...options.context }
    };

    // 添加可选字段
    if (options.errorCode !== undefined) {
      entry.errorCode = options.errorCode;
    }
    
    if (options.durationMs !== undefined) {
      entry.durationMs = options.durationMs;
    }
    
    if (options.pagePath) {
      entry.pagePath = options.pagePath;
    }
    
    if (options.componentName) {
      entry.componentName = options.componentName;
    }
    
    if (options.apiName) {
      entry.apiName = options.apiName;
    }
    
    if (options.networkUrl) {
      entry.networkUrl = options.networkUrl;
    }
    
    if (options.permissionType) {
      entry.permissionType = options.permissionType;
    }
    
    if (options.includeStack || level === LogLevel.ERROR || level === LogLevel.CRITICAL) {
      entry.stack = new Error().stack;
    }
    
    if (options.includeDevice || type === LogType.STARTUP || type === LogType.SYSTEM) {
      entry.device = getDeviceInfo();
    }
    
    if (options.includeNetwork) {
      entry.network = await getNetworkInfo();
    }
    
    if (options.includeLocation) {
      entry.location = await getLocationInfo();
    }
    
    if (options.includePerformance || type === LogType.PERFORMANCE) {
      entry.performance = getPerformanceInfo();
    }

    return entry;
  }

  async log(entry) {
    // 控制台输出
    const logMessage = JSON.stringify(entry);
    
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

    // 写入本地存储
    this.writeToStorage(entry);

    // 发送到远程服务
    this.sendToRemote(entry);
  }

  writeToStorage(entry) {
    try {
      const storageKey = `voicehelper_logs_${new Date().toISOString().split('T')[0]}`;
      let logs = wx.getStorageSync(storageKey) || [];
      
      logs.push(entry);
      
      // 限制每天最多存储1000条日志
      if (logs.length > 1000) {
        logs = logs.slice(-1000);
      }
      
      wx.setStorageSync(storageKey, logs);
    } catch (error) {
      console.error('写入日志到本地存储失败:', error);
    }
  }

  sendToRemote(entry) {
    // 只在生产环境发送重要日志
    if (entry.level === LogLevel.ERROR || entry.level === LogLevel.CRITICAL || entry.type === LogType.STARTUP) {
      // 这里可以实现发送到远程日志服务的逻辑
      // wx.request({
      //   url: 'https://api.voicehelper.ai/logs',
      //   method: 'POST',
      //   data: entry,
      //   fail: (error) => {
      //     console.warn('发送日志到远程服务失败:', error);
      //   }
      // });
    }
  }

  // 基础日志方法
  async debug(message, context) {
    const entry = await this.buildLogEntry(LogLevel.DEBUG, LogType.DEBUG, message, { context });
    this.log(entry);
  }

  async info(message, context) {
    const entry = await this.buildLogEntry(LogLevel.INFO, LogType.SYSTEM, message, { context });
    this.log(entry);
  }

  async warning(message, context) {
    const entry = await this.buildLogEntry(LogLevel.WARNING, LogType.SYSTEM, message, { context });
    this.log(entry);
  }

  async error(message, context) {
    const entry = await this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, { 
      context, 
      includeStack: true,
      includeDevice: true 
    });
    this.log(entry);
  }

  async critical(message, context) {
    const entry = await this.buildLogEntry(LogLevel.CRITICAL, LogType.ERROR, message, { 
      context, 
      includeStack: true,
      includeDevice: true 
    });
    this.log(entry);
  }

  // 特定类型日志方法
  async startup(message, context) {
    const entry = await this.buildLogEntry(LogLevel.INFO, LogType.STARTUP, message, { 
      context,
      includeDevice: true,
      includeNetwork: true,
      includePerformance: true 
    });
    this.log(entry);
  }

  async shutdown(message, context) {
    const entry = await this.buildLogEntry(LogLevel.INFO, LogType.SHUTDOWN, message, { 
      context,
      includePerformance: true 
    });
    this.log(entry);
  }

  async page(action, pagePath, context) {
    const message = `Page ${action}: ${pagePath}`;
    const entry = await this.buildLogEntry(LogLevel.INFO, LogType.PAGE, message, {
      context: { ...context, action },
      pagePath
    });
    this.log(entry);
  }

  async component(action, componentName, context) {
    const message = `Component ${action}: ${componentName}`;
    const entry = await this.buildLogEntry(LogLevel.DEBUG, LogType.COMPONENT, message, {
      context: { ...context, action },
      componentName
    });
    this.log(entry);
  }

  async api(apiName, success, context) {
    const message = `API ${apiName}: ${success ? 'success' : 'fail'}`;
    const level = success ? LogLevel.DEBUG : LogLevel.WARNING;
    const entry = await this.buildLogEntry(level, LogType.API, message, {
      context: { ...context, success },
      apiName
    });
    this.log(entry);
  }

  async network(method, url, statusCode, durationMs, context) {
    const message = `${method} ${url} - ${statusCode}`;
    const level = statusCode >= 500 ? LogLevel.ERROR : statusCode >= 400 ? LogLevel.WARNING : LogLevel.DEBUG;
    const entry = await this.buildLogEntry(level, LogType.NETWORK, message, {
      context: { ...context, method, statusCode },
      networkUrl: url,
      durationMs,
      includeNetwork: true
    });
    this.log(entry);
  }

  async permission(permissionType, granted, context) {
    const message = `Permission ${permissionType}: ${granted ? 'granted' : 'denied'}`;
    const level = granted ? LogLevel.INFO : LogLevel.WARNING;
    const entry = await this.buildLogEntry(level, LogType.PERMISSION, message, {
      context: { ...context, granted },
      permissionType
    });
    this.log(entry);
  }

  async payment(action, result, context) {
    const message = `Payment ${action}: ${result}`;
    const level = result === 'success' ? LogLevel.INFO : LogLevel.WARNING;
    const entry = await this.buildLogEntry(level, LogType.PAYMENT, message, {
      context: { ...context, action, result }
    });
    this.log(entry);
  }

  async share(platform, success, context) {
    const message = `Share to ${platform}: ${success ? 'success' : 'fail'}`;
    const level = success ? LogLevel.INFO : LogLevel.WARNING;
    const entry = await this.buildLogEntry(level, LogType.SHARE, message, {
      context: { ...context, platform, success }
    });
    this.log(entry);
  }

  async errorWithCode(errorCode, message, context) {
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
    const entry = await this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, {
      errorCode,
      context: mergedContext,
      includeStack: true,
      includeDevice: true
    });
    this.log(entry);
  }

  async performance(operation, durationMs, context) {
    const message = `Performance: ${operation}`;
    const mergedContext = { ...context, operation };
    const entry = await this.buildLogEntry(LogLevel.INFO, LogType.PERFORMANCE, message, {
      context: mergedContext,
      durationMs,
      includePerformance: true
    });
    this.log(entry);
  }

  async security(event, context) {
    const message = `Security Event: ${event}`;
    const mergedContext = { ...context, event };
    const entry = await this.buildLogEntry(LogLevel.WARNING, LogType.SECURITY, message, {
      context: mergedContext,
      includeDevice: true
    });
    this.log(entry);
  }

  async business(event, context) {
    const message = `Business Event: ${event}`;
    const mergedContext = { ...context, event };
    const entry = await this.buildLogEntry(LogLevel.INFO, LogType.BUSINESS, message, {
      context: mergedContext
    });
    this.log(entry);
  }

  async exception(message, error, context) {
    let errorCode;
    if (error instanceof VoiceHelperError) {
      errorCode = error.code;
    }

    const mergedContext = {
      ...context,
      errorName: error.name,
      errorMessage: error.message,
      errorStack: error.stack
    };

    const entry = await this.buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, {
      errorCode,
      context: mergedContext,
      includeStack: true,
      includeDevice: true
    });
    this.log(entry);
  }

  // 上下文管理
  setUserId(userId) {
    this.userId = userId;
  }

  setContext(context) {
    this.baseContext = { ...this.baseContext, ...context };
  }

  withModule(module) {
    const newLogger = new VoiceHelperLogger(this.service, module);
    newLogger.baseContext = { ...this.baseContext };
    newLogger.userId = this.userId;
    newLogger.sessionId = this.sessionId;
    return newLogger;
  }

  withContext(context) {
    const newLogger = new VoiceHelperLogger(this.service, this.module);
    newLogger.baseContext = { ...this.baseContext, ...context };
    newLogger.userId = this.userId;
    newLogger.sessionId = this.sessionId;
    return newLogger;
  }

  // 日志管理
  getStoredLogs(date) {
    try {
      const storageKey = `voicehelper_logs_${date || new Date().toISOString().split('T')[0]}`;
      return wx.getStorageSync(storageKey) || [];
    } catch (error) {
      console.error('获取存储的日志失败:', error);
      return [];
    }
  }

  clearStoredLogs(daysToKeep = 7) {
    try {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
      
      const storageInfo = wx.getStorageInfoSync();
      const logKeys = storageInfo.keys.filter(key => key.startsWith('voicehelper_logs_'));
      
      logKeys.forEach(key => {
        const dateStr = key.replace('voicehelper_logs_', '');
        const logDate = new Date(dateStr);
        
        if (logDate < cutoffDate) {
          wx.removeStorageSync(key);
          this.info('删除过期日志', { logKey: key, age: daysToKeep });
        }
      });
    } catch (error) {
      console.error('清理过期日志失败:', error);
    }
  }
}

// 全局日志器实例
let defaultLogger = null;

/**
 * 初始化日志器
 */
function initLogger(service = 'voicehelper-miniprogram') {
  defaultLogger = new VoiceHelperLogger(service);
  return defaultLogger;
}

/**
 * 获取日志器
 */
function getLogger(module = '') {
  if (!defaultLogger) {
    defaultLogger = new VoiceHelperLogger();
  }
  
  if (module) {
    return defaultLogger.withModule(module);
  }
  
  return defaultLogger;
}

// 便利函数
async function debug(message, context) {
  await getLogger().debug(message, context);
}

async function info(message, context) {
  await getLogger().info(message, context);
}

async function warning(message, context) {
  await getLogger().warning(message, context);
}

async function error(message, context) {
  await getLogger().error(message, context);
}

async function critical(message, context) {
  await getLogger().critical(message, context);
}

async function startup(message, context) {
  await getLogger().startup(message, context);
}

async function shutdown(message, context) {
  await getLogger().shutdown(message, context);
}

async function errorWithCode(errorCode, message, context) {
  await getLogger().errorWithCode(errorCode, message, context);
}

async function performance(operation, durationMs, context) {
  await getLogger().performance(operation, durationMs, context);
}

async function security(event, context) {
  await getLogger().security(event, context);
}

async function business(event, context) {
  await getLogger().business(event, context);
}

async function exception(message, error, context) {
  await getLogger().exception(message, error, context);
}

module.exports = {
  LogLevel,
  LogType,
  VoiceHelperLogger,
  initLogger,
  getLogger,
  debug,
  info,
  warning,
  error,
  critical,
  startup,
  shutdown,
  errorWithCode,
  performance,
  security,
  business,
  exception
};
