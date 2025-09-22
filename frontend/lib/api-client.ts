/**
 * API 客户端
 * 集成日志记录和错误处理
 */

import { ErrorCode, VoiceHelperError, getErrorInfo } from './errors';
import { getLogger, createApiLogger } from './logger';

export interface ApiResponse<T = any> {
  data?: T;
  error?: {
    code: ErrorCode;
    message: string;
    description: string;
    details?: Record<string, any>;
  };
  success: boolean;
  timestamp: string;
}

export interface ApiRequestOptions extends RequestInit {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
}

class ApiClient {
  private baseURL: string;
  private defaultTimeout: number;
  private logger = getLogger('ApiClient');
  private apiLogger = createApiLogger();

  constructor(baseURL?: string, timeout: number = 30000) {
    this.baseURL = baseURL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080/api/v1';
    this.defaultTimeout = timeout;

    this.logger.startup('API客户端初始化', {
      baseURL: this.baseURL,
      timeout: this.defaultTimeout,
    });
  }

  private async request<T>(
    endpoint: string,
    options: ApiRequestOptions = {}
  ): Promise<ApiResponse<T>> {
    const {
      timeout = this.defaultTimeout,
      retries = 3,
      retryDelay = 1000,
      ...fetchOptions
    } = options;

    const url = `${this.baseURL}${endpoint}`;
    const method = fetchOptions.method || 'GET';
    
    // 记录请求开始
    const startTime = this.apiLogger.request(method, url, {
      headers: fetchOptions.headers,
      body: fetchOptions.body ? 'present' : undefined,
    });

    // 设置默认headers
    const headers = {
      'Content-Type': 'application/json',
      'X-Request-ID': this.generateRequestId(),
      'X-Client-Version': process.env.NEXT_PUBLIC_APP_VERSION || '1.0.0',
      'X-Client-Platform': 'web',
      ...fetchOptions.headers,
    };

    let lastError: Error;
    
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        this.logger.debug(`API请求尝试 ${attempt + 1}/${retries + 1}`, {
          method,
          url,
          attempt: attempt + 1,
          headers,
        });

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await fetch(url, {
          ...fetchOptions,
          headers,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // 记录响应
        this.apiLogger.response(method, url, response.status, startTime, {
          attempt: attempt + 1,
          contentType: response.headers.get('content-type'),
        });

        const responseData = await this.parseResponse<T>(response);

        if (response.ok) {
          this.logger.debug('API请求成功', {
            method,
            url,
            status: response.status,
            attempt: attempt + 1,
          });

          return {
            data: responseData,
            success: true,
            timestamp: new Date().toISOString(),
          };
        } else {
          // 处理HTTP错误
          const error = this.handleHttpError(response, responseData);
          
          if (attempt === retries || !this.shouldRetry(response.status)) {
            throw error;
          }

          this.logger.warning(`API请求失败，准备重试`, {
            method,
            url,
            status: response.status,
            attempt: attempt + 1,
            nextAttemptIn: retryDelay,
            error: error.message,
          });

          await this.delay(retryDelay);
          lastError = error;
          continue;
        }
      } catch (error) {
        lastError = error as Error;

        if (error instanceof DOMException && error.name === 'AbortError') {
          // 超时错误
          const timeoutError = new VoiceHelperError(
            ErrorCode.NETWORK_TIMEOUT,
            `请求超时 (${timeout}ms)`,
            { method, url, timeout }
          );

          this.logger.errorWithCode(ErrorCode.NETWORK_TIMEOUT, '请求超时', {
            method,
            url,
            timeout,
            attempt: attempt + 1,
          });

          if (attempt === retries) {
            throw timeoutError;
          }
        } else if (error instanceof TypeError && error.message.includes('fetch')) {
          // 网络错误
          const networkError = new VoiceHelperError(
            ErrorCode.FRONTEND_NETWORK_ERROR,
            '网络连接失败',
            { method, url, originalError: error.message }
          );

          this.logger.errorWithCode(ErrorCode.FRONTEND_NETWORK_ERROR, '网络连接失败', {
            method,
            url,
            attempt: attempt + 1,
            error: error.message,
          });

          if (attempt === retries) {
            throw networkError;
          }
        } else {
          // 其他错误
          this.logger.exception('API请求异常', error as Error, {
            method,
            url,
            attempt: attempt + 1,
          });

          if (attempt === retries) {
            throw error;
          }
        }

        if (attempt < retries) {
          this.logger.info(`API请求失败，${retryDelay}ms后重试`, {
            method,
            url,
            attempt: attempt + 1,
            error: (error as Error).message,
          });
          await this.delay(retryDelay);
        }
      }
    }

    throw lastError;
  }

  private async parseResponse<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type');
    
    if (contentType?.includes('application/json')) {
      return await response.json();
    } else if (contentType?.includes('text/')) {
      return await response.text() as unknown as T;
    } else {
      return await response.blob() as unknown as T;
    }
  }

  private handleHttpError(response: Response, responseData: any): VoiceHelperError {
    // 尝试从响应中提取错误信息
    let errorCode = ErrorCode.FRONTEND_API_ERROR;
    let message = `HTTP ${response.status}`;
    let details: Record<string, any> = {
      status: response.status,
      statusText: response.statusText,
      url: response.url,
    };

    if (responseData && typeof responseData === 'object') {
      if (responseData.error && responseData.error.code) {
        errorCode = responseData.error.code;
        message = responseData.error.message || message;
        details = { ...details, ...responseData.error.details };
      } else if (responseData.code) {
        errorCode = responseData.code;
        message = responseData.message || message;
      }
    }

    return new VoiceHelperError(errorCode, message, details);
  }

  private shouldRetry(status: number): boolean {
    // 只对特定状态码进行重试
    return status >= 500 || status === 408 || status === 429;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // HTTP 方法
  async get<T>(endpoint: string, options?: ApiRequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any, options?: ApiRequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T>(endpoint: string, data?: any, options?: ApiRequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async patch<T>(endpoint: string, data?: any, options?: ApiRequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string, options?: ApiRequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }

  // 流式请求 (SSE)
  async stream(
    endpoint: string,
    onMessage: (data: any) => void,
    onError?: (error: Error) => void,
    options?: ApiRequestOptions
  ): Promise<void> {
    const url = `${this.baseURL}${endpoint}`;
    const startTime = this.apiLogger.request('GET', url, { stream: true });

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
          ...options?.headers,
        },
      });

      this.apiLogger.response('GET', url, response.status, startTime, { stream: true });

      if (!response.ok) {
        throw new VoiceHelperError(
          ErrorCode.FRONTEND_API_ERROR,
          `流式请求失败: ${response.status}`,
          { url, status: response.status }
        );
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new VoiceHelperError(
          ErrorCode.FRONTEND_API_ERROR,
          '无法创建流读取器',
          { url }
        );
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onMessage(data);
            } catch (error) {
              this.logger.warning('解析SSE数据失败', {
                line,
                error: (error as Error).message,
              });
            }
          }
        }
      }
    } catch (error) {
      this.logger.exception('流式请求异常', error as Error, { url });
      if (onError) {
        onError(error as Error);
      } else {
        throw error;
      }
    }
  }

  // 文件上传
  async upload<T>(
    endpoint: string,
    file: File,
    onProgress?: (progress: number) => void,
    options?: Omit<ApiRequestOptions, 'body'>
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    const startTime = this.apiLogger.request('POST', url, { 
      upload: true,
      fileName: file.name,
      fileSize: file.size,
    });

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append('file', file);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = (event.loaded / event.total) * 100;
          onProgress(progress);
        }
      };

      xhr.onload = () => {
        this.apiLogger.response('POST', url, xhr.status, startTime, {
          upload: true,
          fileName: file.name,
          fileSize: file.size,
        });

        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve({
              data,
              success: true,
              timestamp: new Date().toISOString(),
            });
          } catch (error) {
            reject(new VoiceHelperError(
              ErrorCode.FRONTEND_API_ERROR,
              '解析上传响应失败',
              { fileName: file.name }
            ));
          }
        } else {
          reject(new VoiceHelperError(
            ErrorCode.FRONTEND_API_ERROR,
            `文件上传失败: ${xhr.status}`,
            { fileName: file.name, status: xhr.status }
          ));
        }
      };

      xhr.onerror = () => {
        this.logger.errorWithCode(ErrorCode.FRONTEND_NETWORK_ERROR, '文件上传网络错误', {
          fileName: file.name,
          fileSize: file.size,
        });
        reject(new VoiceHelperError(
          ErrorCode.FRONTEND_NETWORK_ERROR,
          '文件上传网络错误',
          { fileName: file.name }
        ));
      };

      xhr.open('POST', url);
      
      // 设置headers (不包括Content-Type，让浏览器自动设置)
      const headers = {
        'X-Request-ID': this.generateRequestId(),
        ...options?.headers,
      };
      
      Object.entries(headers).forEach(([key, value]) => {
        if (value) xhr.setRequestHeader(key, value);
      });

      xhr.send(formData);
    });
  }
}

// 创建默认实例
export const apiClient = new ApiClient();

// 导出类型和实例
export default ApiClient;
