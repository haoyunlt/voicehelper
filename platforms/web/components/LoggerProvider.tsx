/**
 * 日志提供者组件
 * 初始化全局日志系统并提供上下文
 */

'use client';

import React, { createContext, useContext, useEffect, ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import { initLogger, getLogger, pageView } from '@/lib/logger';
import type { Logger } from '@/lib/logger';

interface LoggerContextType {
  logger: Logger;
}

const LoggerContext = createContext<LoggerContextType | null>(null);

interface LoggerProviderProps {
  children: ReactNode;
}

export function LoggerProvider({ children }: LoggerProviderProps) {
  const pathname = usePathname();
  
  useEffect(() => {
    // 初始化日志系统
    initLogger();
    const logger = getLogger();
    
    // 记录应用启动
    logger.info('前端应用启动', {
      pathname,
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
      cookieEnabled: navigator.cookieEnabled,
      onLine: navigator.onLine,
      screenResolution: `${screen.width}x${screen.height}`,
      viewportSize: `${window.innerWidth}x${window.innerHeight}`,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      referrer: document.referrer,
    });

    // 监听页面可见性变化
    const handleVisibilityChange = () => {
      logger.info(document.hidden ? 'page_hidden' : 'page_visible', {
        pathname,
        timestamp: new Date().toISOString(),
      });
    };

    // 监听页面卸载
    const handleBeforeUnload = () => {
      logger.info('page_unload', {
        pathname,
        timestamp: new Date().toISOString(),
      });
    };

    // 监听网络状态变化
    const handleOnline = () => {
      logger.info('network_online', {
        pathname,
        timestamp: new Date().toISOString(),
      });
    };

    const handleOffline = () => {
      logger.info('network_offline', {
        pathname,
        timestamp: new Date().toISOString(),
      });
    };

    // 监听未捕获的错误
    const handleError = (event: ErrorEvent) => {
      logger.error('未捕获的JavaScript错误', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack,
        pathname,
      });
    };

    // 监听未处理的Promise拒绝
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      logger.error('未处理的Promise拒绝', {
        reason: event.reason,
        pathname,
      });
    };

    // 添加事件监听器
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    // 清理函数
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  // 监听路由变化
  useEffect(() => {
    pageView(pathname);
  }, [pathname]);

  const logger = getLogger();

  return (
    <LoggerContext.Provider value={{ logger }}>
      {children}
    </LoggerContext.Provider>
  );
}

export function useLoggerContext(): LoggerContextType {
  const context = useContext(LoggerContext);
  if (!context) {
    throw new Error('useLoggerContext must be used within a LoggerProvider');
  }
  return context;
}
