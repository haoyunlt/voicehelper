/**
 * V2架构前端SDK基类
 * 提供统一的流式客户端抽象
 */

import { BaseResponse, ErrorResponse, APIError } from '@/types';

export interface EventSink {
  on(event: string, data: any): void;
}

export interface StreamOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
}

export interface RequestConfig {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  headers?: Record<string, string>;
  body?: any;
  timeout?: number;
}

export abstract class BaseStreamClient {
  protected headers: Record<string, string> = {};
  protected baseURL: string;
  
  constructor(baseURL: string = '', token?: string) {
    this.baseURL = baseURL || this.getDefaultBaseURL();
    if (token) {
      this.headers.Authorization = `Bearer ${token}`;
    }
  }
  
  private getDefaultBaseURL(): string {
    if (typeof window !== 'undefined') {
      return `${window.location.protocol}//${window.location.host}`;
    }
    return 'http://localhost:3000';
  }
  
  protected generateTraceId(): string {
    return `trace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  protected addTraceHeaders(headers: Record<string, string> = {}): Record<string, string> {
    return {
      ...this.headers,
      'X-Trace-ID': this.generateTraceId(),
      'X-Tenant-ID': 'default',
      ...headers,
    };
  }
  
  protected async retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        
        if (attempt === maxRetries) {
          break;
        }
        
        // 指数退避
        const delay = baseDelay * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError!;
  }

  protected async request<T extends BaseResponse>(
    endpoint: string,
    config: RequestConfig = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const headers = this.addTraceHeaders(config.headers);
    
    const requestInit: RequestInit = {
      method: config.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      body: config.body ? JSON.stringify(config.body) : null,
    };

    try {
      const controller = new AbortController();
      const timeoutId = config.timeout 
        ? setTimeout(() => controller.abort(), config.timeout)
        : null;

      const response = await fetch(url, {
        ...requestInit,
        signal: controller.signal,
      });

      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      if (!response.ok) {
        const errorData: ErrorResponse = await response.json();
        throw new APIError(
          errorData.error.message,
          response.status,
          errorData.error.code,
          errorData.error.details
        );
      }

      return await response.json() as T;
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      
      throw new APIError(
        error instanceof Error ? error.message : 'Unknown error',
        0,
        'NETWORK_ERROR',
        error
      );
    }
  }

  protected createEventSource(
    endpoint: string,
    options: StreamOptions = {}
  ): EventSource {
    const url = new URL(`${this.baseURL}${endpoint}`);
    
    // 添加认证参数到URL（如果需要）
    if (this.headers.Authorization) {
      url.searchParams.set('token', this.headers.Authorization.replace('Bearer ', ''));
    }

    const eventSource = new EventSource(url.toString());
    
    // 设置超时
    if (options.timeout) {
      setTimeout(() => {
        eventSource.close();
      }, options.timeout);
    }

    return eventSource;
  }

  protected createWebSocket(
    endpoint: string,
    protocols?: string[]
  ): WebSocket {
    const wsProtocol = this.baseURL.startsWith('https') ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${this.baseURL.replace(/^https?:\/\//, '')}${endpoint}`;
    
    return new WebSocket(wsUrl, protocols);
  }
}
