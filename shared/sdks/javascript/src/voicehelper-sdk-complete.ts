/**
 * VoiceHelper AI JavaScript SDK
 * 完整的TypeScript/JavaScript客户端SDK
 * 
 * @version 1.9.0
 * @author VoiceHelper Team
 * @license MIT
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import EventSource from 'eventsource';

// ==================== 类型定义 ====================

export interface VoiceHelperConfig {
  apiKey: string;
  baseURL?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  debug?: boolean;
}

export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentPart[];
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface ContentPart {
  type: 'text' | 'image_url' | 'audio_url';
  text?: string;
  image_url?: {
    url: string;
    detail?: 'low' | 'high' | 'auto';
  };
  audio_url?: {
    url: string;
    format?: 'mp3' | 'wav' | 'ogg' | 'm4a';
  };
}

export interface ChatCompletionRequest {
  messages: Message[];
  model?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: Tool[];
  tool_choice?: 'none' | 'auto' | ToolChoice;
}

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Choice[];
  usage?: Usage;
}

export interface Choice {
  index: number;
  message: Message;
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter';
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface Tool {
  type: 'function';
  function: FunctionDefinition;
}

export interface FunctionDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ToolChoice {
  type: 'function';
  function: {
    name: string;
  };
}

export interface TranscriptionRequest {
  file: File | Blob;
  model?: string;
  language?: string;
  response_format?: 'json' | 'text' | 'srt' | 'verbose_json';
}

export interface TranscriptionResponse {
  text: string;
  language?: string;
  duration?: number;
  segments?: TranscriptionSegment[];
}

export interface TranscriptionSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  confidence: number;
}

export interface SynthesisRequest {
  text: string;
  voice?: string;
  response_format?: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm';
  speed?: number;
  emotion?: 'neutral' | 'happy' | 'sad' | 'angry' | 'excited' | 'calm';
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  status: 'active' | 'archived' | 'deleted';
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  created_at: string;
  document_count: number;
}

export interface SearchRequest {
  query: string;
  datasets?: string[];
  limit?: number;
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface SearchResult {
  content: string;
  score: number;
  metadata: Record<string, any>;
}

export interface Service {
  id: string;
  name: string;
  category: string;
  description: string;
  status: string;
}

export interface Connection {
  id: string;
  service_id: string;
  name: string;
  status: 'active' | 'inactive' | 'error';
  created_at: string;
}

export interface UserProfile {
  id: string;
  email: string;
  name: string;
  avatar_url?: string;
  plan: 'free' | 'pro' | 'enterprise';
  created_at: string;
}

export interface UsageStats {
  period: {
    start: string;
    end: string;
  };
  requests: {
    total: number;
    by_endpoint: Record<string, number>;
  };
  tokens: {
    input: number;
    output: number;
    total: number;
  };
}

export interface VoiceHelperError {
  message: string;
  type: string;
  code: string;
  param?: string;
}

// ==================== 主要SDK类 ====================

export class VoiceHelperSDK {
  private client: AxiosInstance;
  private config: Required<VoiceHelperConfig>;

  constructor(config: VoiceHelperConfig) {
    this.config = {
      baseURL: 'https://api.voicehelper.ai/v1',
      timeout: 30000,
      maxRetries: 3,
      retryDelay: 1000,
      debug: false,
      ...config,
    };

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Authorization': `Bearer ${this.config.apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': 'VoiceHelper-JS-SDK/1.9.0',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        if (this.config.debug) {
          console.log('VoiceHelper Request:', config);
        }
        return config;
      },
      (error) => {
        if (this.config.debug) {
          console.error('VoiceHelper Request Error:', error);
        }
        return Promise.reject(error);
      }
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => {
        if (this.config.debug) {
          console.log('VoiceHelper Response:', response);
        }
        return response;
      },
      async (error) => {
        if (this.config.debug) {
          console.error('VoiceHelper Response Error:', error);
        }

        // 自动重试逻辑
        if (this.shouldRetry(error) && error.config && !error.config._retry) {
          error.config._retry = true;
          error.config._retryCount = (error.config._retryCount || 0) + 1;

          if (error.config._retryCount <= this.config.maxRetries) {
            await this.delay(this.config.retryDelay * error.config._retryCount);
            return this.client.request(error.config);
          }
        }

        return Promise.reject(this.formatError(error));
      }
    );
  }

  private shouldRetry(error: any): boolean {
    return (
      error.code === 'ECONNABORTED' ||
      error.code === 'ENOTFOUND' ||
      error.code === 'ECONNRESET' ||
      (error.response && [408, 429, 500, 502, 503, 504].includes(error.response.status))
    );
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private formatError(error: any): VoiceHelperError {
    if (error.response?.data?.error) {
      return error.response.data.error;
    }

    return {
      message: error.message || 'Unknown error occurred',
      type: 'client_error',
      code: error.code || 'unknown_error',
    };
  }

  // ==================== 对话接口 ====================

  /**
   * 创建对话完成
   */
  async createChatCompletion(request: ChatCompletionRequest): Promise<ChatCompletionResponse> {
    const response = await this.client.post<ChatCompletionResponse>('/chat/completions', request);
    return response.data;
  }

  /**
   * 创建流式对话完成
   */
  async createChatCompletionStream(
    request: ChatCompletionRequest,
    onChunk: (chunk: any) => void,
    onError?: (error: VoiceHelperError) => void,
    onComplete?: () => void
  ): Promise<void> {
    const streamRequest = { ...request, stream: true };
    
    try {
      const response = await this.client.post('/chat/completions', streamRequest, {
        responseType: 'stream',
        headers: {
          'Accept': 'text/event-stream',
        },
      });

      const eventSource = new EventSource(`${this.config.baseURL}/chat/completions`, {
        headers: {
          'Authorization': `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
      });

      eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
          eventSource.close();
          onComplete?.();
          return;
        }

        try {
          const chunk = JSON.parse(event.data);
          onChunk(chunk);
        } catch (error) {
          console.error('Failed to parse chunk:', error);
        }
      };

      eventSource.onerror = (error) => {
        eventSource.close();
        onError?.(this.formatError(error));
      };

    } catch (error) {
      onError?.(this.formatError(error));
    }
  }

  /**
   * 获取对话列表
   */
  async listConversations(params?: {
    limit?: number;
    offset?: number;
    status?: 'active' | 'archived' | 'deleted';
  }): Promise<{ conversations: Conversation[]; total: number; has_more: boolean }> {
    const response = await this.client.get('/chat/conversations', { params });
    return response.data;
  }

  /**
   * 创建新对话
   */
  async createConversation(data: {
    title?: string;
    initial_message?: string;
  }): Promise<Conversation> {
    const response = await this.client.post<Conversation>('/chat/conversations', data);
    return response.data;
  }

  /**
   * 获取对话详情
   */
  async getConversation(conversationId: string): Promise<Conversation & { messages: Message[] }> {
    const response = await this.client.get(`/chat/conversations/${conversationId}`);
    return response.data;
  }

  /**
   * 删除对话
   */
  async deleteConversation(conversationId: string): Promise<void> {
    await this.client.delete(`/chat/conversations/${conversationId}`);
  }

  // ==================== 语音接口 ====================

  /**
   * 语音转文字
   */
  async transcribeAudio(request: TranscriptionRequest): Promise<TranscriptionResponse> {
    const formData = new FormData();
    formData.append('file', request.file);
    
    if (request.model) formData.append('model', request.model);
    if (request.language) formData.append('language', request.language);
    if (request.response_format) formData.append('response_format', request.response_format);

    const response = await this.client.post<TranscriptionResponse>('/voice/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  /**
   * 文字转语音
   */
  async synthesizeText(request: SynthesisRequest): Promise<ArrayBuffer> {
    const response = await this.client.post('/voice/synthesize', request, {
      responseType: 'arraybuffer',
    });

    return response.data;
  }

  /**
   * 创建实时语音连接
   */
  createRealtimeVoiceConnection(params?: {
    model?: string;
    voice?: string;
  }): WebSocket {
    const wsUrl = this.config.baseURL.replace('http', 'ws') + '/voice/realtime';
    const url = new URL(wsUrl);
    
    if (params?.model) url.searchParams.set('model', params.model);
    if (params?.voice) url.searchParams.set('voice', params.voice);
    url.searchParams.set('authorization', `Bearer ${this.config.apiKey}`);

    return new WebSocket(url.toString());
  }

  // ==================== 知识库接口 ====================

  /**
   * 获取数据集列表
   */
  async listDatasets(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ datasets: Dataset[]; total: number }> {
    const response = await this.client.get('/knowledge/datasets', { params });
    return response.data;
  }

  /**
   * 创建数据集
   */
  async createDataset(data: {
    name: string;
    description?: string;
  }): Promise<Dataset> {
    const response = await this.client.post<Dataset>('/knowledge/datasets', data);
    return response.data;
  }

  /**
   * 上传文档
   */
  async uploadDocument(
    datasetId: string,
    file: File | Blob,
    metadata?: Record<string, any>
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    const response = await this.client.post(
      `/knowledge/datasets/${datasetId}/documents`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  /**
   * 知识检索
   */
  async searchKnowledge(request: SearchRequest): Promise<SearchResponse> {
    const response = await this.client.post<SearchResponse>('/knowledge/search', request);
    return response.data;
  }

  // ==================== 服务集成接口 ====================

  /**
   * 获取可用服务列表
   */
  async listAvailableServices(params?: {
    category?: string;
    status?: string;
  }): Promise<{ services: Service[] }> {
    const response = await this.client.get('/integrations/services', { params });
    return response.data;
  }

  /**
   * 获取已连接服务
   */
  async listConnections(): Promise<{ connections: Connection[] }> {
    const response = await this.client.get('/integrations/connections');
    return response.data;
  }

  /**
   * 创建服务连接
   */
  async createConnection(data: {
    service_id: string;
    name?: string;
    config: Record<string, any>;
  }): Promise<Connection> {
    const response = await this.client.post<Connection>('/integrations/connections', data);
    return response.data;
  }

  /**
   * 执行服务操作
   */
  async executeServiceOperation(
    connectionId: string,
    operation: string,
    parameters?: Record<string, any>
  ): Promise<any> {
    const response = await this.client.post(
      `/integrations/connections/${connectionId}/execute`,
      { operation, parameters }
    );
    return response.data;
  }

  // ==================== 用户管理接口 ====================

  /**
   * 获取用户资料
   */
  async getUserProfile(): Promise<UserProfile> {
    const response = await this.client.get<UserProfile>('/users/profile');
    return response.data;
  }

  /**
   * 更新用户资料
   */
  async updateUserProfile(data: {
    name?: string;
    avatar_url?: string;
  }): Promise<UserProfile> {
    const response = await this.client.put<UserProfile>('/users/profile', data);
    return response.data;
  }

  /**
   * 获取使用统计
   */
  async getUserUsage(params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<UsageStats> {
    const response = await this.client.get<UsageStats>('/users/usage', { params });
    return response.data;
  }

  // ==================== 工具方法 ====================

  /**
   * 设置API密钥
   */
  setApiKey(apiKey: string): void {
    this.config.apiKey = apiKey;
    this.client.defaults.headers['Authorization'] = `Bearer ${apiKey}`;
  }

  /**
   * 设置基础URL
   */
  setBaseURL(baseURL: string): void {
    this.config.baseURL = baseURL;
    this.client.defaults.baseURL = baseURL;
  }

  /**
   * 获取当前配置
   */
  getConfig(): VoiceHelperConfig {
    return { ...this.config };
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<{ status: string; version: string }> {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw this.formatError(error);
    }
  }
}

// ==================== 便捷函数 ====================

/**
 * 创建VoiceHelper SDK实例
 */
export function createVoiceHelperClient(config: VoiceHelperConfig): VoiceHelperSDK {
  return new VoiceHelperSDK(config);
}

/**
 * 简单的聊天函数
 */
export async function chat(
  apiKey: string,
  message: string,
  options?: {
    model?: string;
    temperature?: number;
    baseURL?: string;
  }
): Promise<string> {
  const client = new VoiceHelperSDK({
    apiKey,
    baseURL: options?.baseURL,
  });

  const response = await client.createChatCompletion({
    messages: [{ role: 'user', content: message }],
    model: options?.model || 'gpt-4',
    temperature: options?.temperature || 0.7,
  });

  return response.choices[0]?.message?.content || '';
}

/**
 * 简单的语音转文字函数
 */
export async function transcribe(
  apiKey: string,
  audioFile: File | Blob,
  options?: {
    model?: string;
    language?: string;
    baseURL?: string;
  }
): Promise<string> {
  const client = new VoiceHelperSDK({
    apiKey,
    baseURL: options?.baseURL,
  });

  const response = await client.transcribeAudio({
    file: audioFile,
    model: options?.model,
    language: options?.language,
  });

  return response.text;
}

/**
 * 简单的文字转语音函数
 */
export async function synthesize(
  apiKey: string,
  text: string,
  options?: {
    voice?: string;
    speed?: number;
    emotion?: string;
    baseURL?: string;
  }
): Promise<ArrayBuffer> {
  const client = new VoiceHelperSDK({
    apiKey,
    baseURL: options?.baseURL,
  });

  return await client.synthesizeText({
    text,
    voice: options?.voice,
    speed: options?.speed,
    emotion: options?.emotion as any,
  });
}

// ==================== 导出 ====================

export default VoiceHelperSDK;

// 类型导出
export type {
  VoiceHelperConfig,
  Message,
  ContentPart,
  ChatCompletionRequest,
  ChatCompletionResponse,
  Choice,
  Usage,
  Tool,
  FunctionDefinition,
  ToolCall,
  ToolChoice,
  TranscriptionRequest,
  TranscriptionResponse,
  TranscriptionSegment,
  SynthesisRequest,
  Conversation,
  Dataset,
  SearchRequest,
  SearchResponse,
  SearchResult,
  Service,
  Connection,
  UserProfile,
  UsageStats,
  VoiceHelperError,
};
