import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import WebSocket from 'ws';
import {
  ChatbotConfig,
  LoginRequest,
  LoginResponse,
  Conversation,
  CreateConversationRequest,
  UpdateConversationRequest,
  Message,
  SendMessageRequest,
  StreamEvent,
  TranscribeRequest,
  TranscribeResponse,
  SynthesizeRequest,
  Dataset,
  CreateDatasetRequest,
  Document,
  UploadDocumentRequest,
  SearchRequest,
  SearchResponse,
  PaginationParams,
  PaginatedResponse,
  HealthResponse,
  EventCallback,
  ErrorCallback,
} from './types';
import { ChatbotError, APIError, NetworkError } from './errors';

/**
 * Chatbot API客户端
 */
export class ChatbotClient {
  private http: AxiosInstance;
  private config: Required<ChatbotConfig>;

  constructor(config: ChatbotConfig) {
    this.config = {
      baseURL: 'https://api.chatbot.ai/v1',
      timeout: 30000,
      tenantId: 'default',
      ...config,
    };

    this.http = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Authorization': `Bearer ${this.config.apiKey}`,
        'Content-Type': 'application/json',
        'X-Tenant-ID': this.config.tenantId,
      },
    });

    // 响应拦截器
    this.http.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response) {
          throw new APIError(
            error.response.data?.error || 'API Error',
            error.response.status,
            error.response.data?.code
          );
        } else if (error.request) {
          throw new NetworkError('Network Error: ' + error.message);
        } else {
          throw new ChatbotError('Request Error: ' + error.message);
        }
      }
    );
  }

  /**
   * 认证相关方法
   */
  auth = {
    /**
     * 微信小程序登录
     */
    wechatLogin: async (request: LoginRequest): Promise<LoginResponse> => {
      const response = await this.http.post('/auth/wechat/login', request);
      return response.data;
    },

    /**
     * 刷新令牌
     */
    refreshToken: async (refreshToken: string): Promise<{ token: string; expires_at: string }> => {
      const response = await this.http.post('/auth/refresh', {
        refresh_token: refreshToken,
      });
      return response.data;
    },
  };

  /**
   * 对话相关方法
   */
  conversations = {
    /**
     * 获取对话列表
     */
    list: async (params?: PaginationParams & { status?: string }): Promise<PaginatedResponse<Conversation>> => {
      const response = await this.http.get('/conversations', { params });
      return {
        data: response.data.conversations,
        pagination: response.data.pagination,
      };
    },

    /**
     * 创建对话
     */
    create: async (request: CreateConversationRequest): Promise<Conversation> => {
      const response = await this.http.post('/conversations', request);
      return response.data;
    },

    /**
     * 获取对话详情
     */
    get: async (conversationId: string): Promise<Conversation> => {
      const response = await this.http.get(`/conversations/${conversationId}`);
      return response.data;
    },

    /**
     * 更新对话
     */
    update: async (conversationId: string, request: UpdateConversationRequest): Promise<Conversation> => {
      const response = await this.http.put(`/conversations/${conversationId}`, request);
      return response.data;
    },

    /**
     * 删除对话
     */
    delete: async (conversationId: string): Promise<void> => {
      await this.http.delete(`/conversations/${conversationId}`);
    },
  };

  /**
   * 消息相关方法
   */
  messages = {
    /**
     * 获取消息列表
     */
    list: async (
      conversationId: string,
      params?: PaginationParams & { before?: string }
    ): Promise<PaginatedResponse<Message>> => {
      const response = await this.http.get(`/conversations/${conversationId}/messages`, { params });
      return {
        data: response.data.messages,
        pagination: response.data.pagination,
      };
    },

    /**
     * 发送消息 (非流式)
     */
    send: async (conversationId: string, request: SendMessageRequest): Promise<Message> => {
      const response = await this.http.post(`/conversations/${conversationId}/messages`, {
        ...request,
        stream: false,
      });
      return response.data;
    },

    /**
     * 发送消息 (流式)
     */
    sendStream: async (
      conversationId: string,
      request: SendMessageRequest,
      onMessage: EventCallback<StreamEvent>,
      onError?: ErrorCallback
    ): Promise<void> => {
      try {
        const response = await this.http.post(`/conversations/${conversationId}/messages`, {
          ...request,
          stream: true,
        }, {
          responseType: 'stream',
        });

        const stream = response.data;
        let buffer = '';

        stream.on('data', (chunk: Buffer) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                onMessage(data);
              } catch (error) {
                console.warn('Failed to parse SSE data:', line);
              }
            }
          }
        });

        stream.on('end', () => {
          // Stream ended
        });

        stream.on('error', (error: Error) => {
          if (onError) {
            onError(error);
          }
        });
      } catch (error) {
        if (onError) {
          onError(error as Error);
        }
      }
    },
  };

  /**
   * 语音相关方法
   */
  voice = {
    /**
     * 语音转文字
     */
    transcribe: async (request: TranscribeRequest): Promise<TranscribeResponse> => {
      const formData = new FormData();
      formData.append('audio', request.audio);
      if (request.language) {
        formData.append('language', request.language);
      }

      const response = await this.http.post('/voice/transcribe', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    },

    /**
     * 文字转语音
     */
    synthesize: async (request: SynthesizeRequest): Promise<ArrayBuffer> => {
      const response = await this.http.post('/voice/synthesize', request, {
        responseType: 'arraybuffer',
      });
      return response.data;
    },
  };

  /**
   * 数据集相关方法
   */
  datasets = {
    /**
     * 获取数据集列表
     */
    list: async (params?: PaginationParams): Promise<PaginatedResponse<Dataset>> => {
      const response = await this.http.get('/datasets', { params });
      return {
        data: response.data.datasets,
        pagination: response.data.pagination,
      };
    },

    /**
     * 创建数据集
     */
    create: async (request: CreateDatasetRequest): Promise<Dataset> => {
      const response = await this.http.post('/datasets', request);
      return response.data;
    },

    /**
     * 获取数据集详情
     */
    get: async (datasetId: string): Promise<Dataset> => {
      const response = await this.http.get(`/datasets/${datasetId}`);
      return response.data;
    },

    /**
     * 上传文档
     */
    uploadDocument: async (datasetId: string, request: UploadDocumentRequest): Promise<Document> => {
      const formData = new FormData();
      formData.append('file', request.file);
      if (request.name) {
        formData.append('name', request.name);
      }

      const response = await this.http.post(`/datasets/${datasetId}/documents`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    },
  };

  /**
   * 搜索相关方法
   */
  search = {
    /**
     * 知识搜索
     */
    query: async (request: SearchRequest): Promise<SearchResponse> => {
      const response = await this.http.post('/search', request);
      return response.data;
    },
  };

  /**
   * 系统相关方法
   */
  system = {
    /**
     * 健康检查
     */
    health: async (): Promise<HealthResponse> => {
      const response = await this.http.get('/health');
      return response.data;
    },

    /**
     * 获取系统指标
     */
    metrics: async (): Promise<string> => {
      const response = await this.http.get('/metrics', {
        headers: {
          'Accept': 'text/plain',
        },
      });
      return response.data;
    },
  };

  /**
   * WebSocket连接 (用于实时通信)
   */
  createWebSocket(conversationId: string): WebSocket {
    const wsUrl = this.config.baseURL.replace(/^http/, 'ws') + `/conversations/${conversationId}/ws`;
    const ws = new WebSocket(wsUrl, {
      headers: {
        'Authorization': `Bearer ${this.config.apiKey}`,
        'X-Tenant-ID': this.config.tenantId,
      },
    });

    return ws;
  }

  /**
   * 更新配置
   */
  updateConfig(newConfig: Partial<ChatbotConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // 更新HTTP客户端配置
    if (newConfig.apiKey) {
      this.http.defaults.headers['Authorization'] = `Bearer ${newConfig.apiKey}`;
    }
    if (newConfig.tenantId) {
      this.http.defaults.headers['X-Tenant-ID'] = newConfig.tenantId;
    }
    if (newConfig.baseURL) {
      this.http.defaults.baseURL = newConfig.baseURL;
    }
    if (newConfig.timeout) {
      this.http.defaults.timeout = newConfig.timeout;
    }
  }
}
