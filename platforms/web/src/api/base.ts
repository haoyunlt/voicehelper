/**
 * V2架构前端SDK基类
 * 提供统一的流式客户端抽象
 */

export interface EventSink {
  on(event: string, data: any): void;
}

export interface StreamOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
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
}
