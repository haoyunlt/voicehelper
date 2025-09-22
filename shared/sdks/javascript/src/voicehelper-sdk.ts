/**
 * VoiceHelper SDK - v1.9.0
 * 智能聊天机器人系统 JavaScript/TypeScript SDK
 */

export interface VoiceHelperConfig {
  apiKey: string;
  baseURL?: string;
  timeout?: number;
  retries?: number;
  debug?: boolean;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  attachments?: Attachment[];
}

export interface Attachment {
  type: 'image' | 'audio' | 'video' | 'file';
  url?: string;
  data?: string; // base64
  metadata?: Record<string, any>;
}

export interface ChatCompletionOptions {
  model?: string;
  stream?: boolean;
  maxTokens?: number;
  temperature?: number;
  multimodalConfig?: MultimodalConfig;
}

export interface MultimodalConfig {
  enableVision?: boolean;
  enableAudio?: boolean;
  enableEmotionDetection?: boolean;
  fusionStrategy?: 'hierarchical' | 'adaptive_attention' | 'cross_transformer';
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Choice[];
  usage: Usage;
  processingMetrics: ProcessingMetrics;
}

export interface Choice {
  index: number;
  message: ChatMessage;
  finishReason: string;
}

export interface Usage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface ProcessingMetrics {
  totalTimeMs: number;
  voiceProcessingTimeMs?: number;
  visionProcessingTimeMs?: number;
  fusionTimeMs?: number;
  modelInferenceTimeMs?: number;
}

export interface VoiceSynthesisOptions {
  voice?: string;
  emotion?: 'neutral' | 'happy' | 'sad' | 'angry' | 'excited' | 'calm';
  speed?: number;
  language?: string;
  streaming?: boolean;
}

export interface VoiceRecognitionOptions {
  language?: string;
  enableEmotionDetection?: boolean;
  enableSpeakerSeparation?: boolean;
  noiseReduction?: boolean;
}

export interface VisionAnalysisOptions {
  tasks?: string[];
  query?: string;
}

export interface MCPServiceCallOptions {
  operation: string;
  params: Record<string, any>;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: Record<string, ComponentHealth>;
  timestamp: string;
}

export interface ComponentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTimeMs: number;
  errorRate: number;
  lastCheck: string;
}

export class VoiceHelperError extends Error {
  constructor(
    message: string,
    public code: string,
    public type: string,
    public details?: any
  ) {
    super(message);
    this.name = 'VoiceHelperError';
  }
}

export class VoiceHelperSDK {
  private config: Required<VoiceHelperConfig>;
  private baseHeaders: Record<string, string>;

  constructor(config: VoiceHelperConfig) {
    this.config = {
      baseURL: 'https://api.voicehelper.com/v1',
      timeout: 30000,
      retries: 3,
      debug: false,
      ...config,
    };

    this.baseHeaders = {
      'Content-Type': 'application/json',
      'X-API-Key': this.config.apiKey,
      'User-Agent': 'VoiceHelper-SDK-JS/1.9.0',
    };
  }

  // ==================== 聊天对话 API ====================

  /**
   * 创建聊天完成
   */
  async createChatCompletion(
    messages: ChatMessage[],
    options: ChatCompletionOptions = {}
  ): Promise<ChatCompletionResponse> {
    const requestBody = {
      messages,
      model: options.model || 'gpt-4-turbo',
      stream: options.stream || false,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      multimodal_config: options.multimodalConfig,
    };

    if (options.stream) {
      throw new Error('Streaming not supported in this method. Use createChatCompletionStream instead.');
    }

    return this.makeRequest<ChatCompletionResponse>('POST', '/chat/completions', requestBody);
  }

  /**
   * 创建流式聊天完成
   */
  async *createChatCompletionStream(
    messages: ChatMessage[],
    options: ChatCompletionOptions = {}
  ): AsyncGenerator<any, void, unknown> {
    const requestBody = {
      messages,
      model: options.model || 'gpt-4-turbo',
      stream: true,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      multimodal_config: options.multimodalConfig,
    };

    const response = await fetch(`${this.config.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        ...this.baseHeaders,
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw await this.handleErrorResponse(response);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new VoiceHelperError('No response body', 'no_response_body', 'stream_error');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              return;
            }
            try {
              yield JSON.parse(data);
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * 获取对话列表
   */
  async getConversations(options: {
    limit?: number;
    offset?: number;
    filter?: 'all' | 'active' | 'archived';
  } = {}): Promise<any> {
    const params = new URLSearchParams();
    if (options.limit) params.set('limit', options.limit.toString());
    if (options.offset) params.set('offset', options.offset.toString());
    if (options.filter) params.set('filter', options.filter);

    const url = `/chat/conversations${params.toString() ? '?' + params.toString() : ''}`;
    return this.makeRequest('GET', url);
  }

  /**
   * 获取对话详情
   */
  async getConversation(conversationId: string): Promise<any> {
    return this.makeRequest('GET', `/chat/conversations/${conversationId}`);
  }

  // ==================== 语音处理 API ====================

  /**
   * 语音合成
   */
  async synthesizeVoice(
    text: string,
    options: VoiceSynthesisOptions = {}
  ): Promise<{ audioUrl?: string; audioData?: ArrayBuffer; metadata: any }> {
    const requestBody = {
      text,
      voice: options.voice || 'alloy',
      emotion: options.emotion || 'neutral',
      speed: options.speed || 1.0,
      language: options.language || 'zh-CN',
      streaming: options.streaming || false,
    };

    const response = await fetch(`${this.config.baseURL}/voice/synthesize`, {
      method: 'POST',
      headers: this.baseHeaders,
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw await this.handleErrorResponse(response);
    }

    const contentType = response.headers.get('content-type');
    
    if (contentType?.includes('audio/')) {
      // Binary audio response
      const audioData = await response.arrayBuffer();
      return {
        audioData,
        metadata: {
          contentType,
          size: audioData.byteLength,
        },
      };
    } else {
      // JSON response with URL
      const result = await response.json();
      return {
        audioUrl: result.audio_url,
        metadata: result,
      };
    }
  }

  /**
   * 语音识别
   */
  async recognizeVoice(
    audioFile: File | Blob,
    options: VoiceRecognitionOptions = {}
  ): Promise<any> {
    const formData = new FormData();
    formData.append('audio', audioFile);
    
    if (Object.keys(options).length > 0) {
      formData.append('config', JSON.stringify(options));
    }

    const headers = { ...this.baseHeaders };
    delete headers['Content-Type']; // Let browser set it for FormData

    const response = await fetch(`${this.config.baseURL}/voice/recognize`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw await this.handleErrorResponse(response);
    }

    return response.json();
  }

  // ==================== 图像理解 API ====================

  /**
   * 图像分析
   */
  async analyzeImage(
    imageFile: File | Blob,
    options: VisionAnalysisOptions = {}
  ): Promise<any> {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    if (options.tasks) {
      options.tasks.forEach(task => formData.append('tasks', task));
    }
    
    if (options.query) {
      formData.append('query', options.query);
    }

    const headers = { ...this.baseHeaders };
    delete headers['Content-Type'];

    const response = await fetch(`${this.config.baseURL}/vision/analyze`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw await this.handleErrorResponse(response);
    }

    return response.json();
  }

  // ==================== 多模态融合 API ====================

  /**
   * 多模态融合
   */
  async fuseModalities(
    modalityInputs: any[],
    options: {
      fusionStrategy?: string;
      context?: Record<string, any>;
    } = {}
  ): Promise<any> {
    const requestBody = {
      modality_inputs: modalityInputs,
      fusion_strategy: options.fusionStrategy || 'hierarchical',
      context: options.context,
    };

    return this.makeRequest('POST', '/multimodal/fuse', requestBody);
  }

  // ==================== MCP服务集成 API ====================

  /**
   * 获取可用服务列表
   */
  async getServices(options: {
    category?: string;
    status?: string;
  } = {}): Promise<any> {
    const params = new URLSearchParams();
    if (options.category) params.set('category', options.category);
    if (options.status) params.set('status', options.status);

    const url = `/mcp/services${params.toString() ? '?' + params.toString() : ''}`;
    return this.makeRequest('GET', url);
  }

  /**
   * 调用MCP服务
   */
  async callMCPService(
    serviceName: string,
    options: MCPServiceCallOptions
  ): Promise<any> {
    return this.makeRequest('POST', `/mcp/services/${serviceName}/call`, options);
  }

  // ==================== 数据集管理 API ====================

  /**
   * 获取数据集列表
   */
  async getDatasets(): Promise<any> {
    return this.makeRequest('GET', '/datasets');
  }

  /**
   * 创建数据集
   */
  async createDataset(name: string, description?: string, config?: any): Promise<any> {
    const requestBody = {
      name,
      description,
      config,
    };

    return this.makeRequest('POST', '/datasets', requestBody);
  }

  /**
   * 数据摄取
   */
  async ingestData(
    datasetId: string,
    files: File[],
    metadata?: Record<string, any>
  ): Promise<any> {
    const formData = new FormData();
    
    files.forEach(file => formData.append('files', file));
    
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }

    const headers = { ...this.baseHeaders };
    delete headers['Content-Type'];

    const response = await fetch(`${this.config.baseURL}/datasets/${datasetId}/ingest`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw await this.handleErrorResponse(response);
    }

    return response.json();
  }

  // ==================== 系统监控 API ====================

  /**
   * 系统健康检查
   */
  async getSystemHealth(): Promise<SystemHealth> {
    return this.makeRequest<SystemHealth>('GET', '/system/health');
  }

  /**
   * 系统指标
   */
  async getSystemMetrics(): Promise<any> {
    return this.makeRequest('GET', '/system/metrics');
  }

  // ==================== 工具方法 ====================

  /**
   * 通用请求方法
   */
  private async makeRequest<T = any>(
    method: string,
    endpoint: string,
    body?: any
  ): Promise<T> {
    const url = `${this.config.baseURL}${endpoint}`;
    let attempt = 0;

    while (attempt < this.config.retries) {
      try {
        if (this.config.debug) {
          console.log(`[VoiceHelper SDK] ${method} ${url}`, body);
        }

        const response = await fetch(url, {
          method,
          headers: this.baseHeaders,
          body: body ? JSON.stringify(body) : undefined,
          signal: AbortSignal.timeout(this.config.timeout),
        });

        if (!response.ok) {
          throw await this.handleErrorResponse(response);
        }

        const result = await response.json();
        
        if (this.config.debug) {
          console.log(`[VoiceHelper SDK] Response:`, result);
        }

        return result;
      } catch (error) {
        attempt++;
        
        if (attempt >= this.config.retries) {
          throw error;
        }

        // Exponential backoff
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw new VoiceHelperError('Max retries exceeded', 'max_retries', 'network_error');
  }

  /**
   * 处理错误响应
   */
  private async handleErrorResponse(response: Response): Promise<VoiceHelperError> {
    try {
      const errorData = await response.json();
      const error = errorData.error || {};
      
      return new VoiceHelperError(
        error.message || `HTTP ${response.status}`,
        error.code || 'unknown_error',
        error.type || 'api_error',
        error.details
      );
    } catch {
      return new VoiceHelperError(
        `HTTP ${response.status} ${response.statusText}`,
        'http_error',
        'api_error'
      );
    }
  }

  /**
   * 设置API密钥
   */
  setApiKey(apiKey: string): void {
    this.config.apiKey = apiKey;
    this.baseHeaders['X-API-Key'] = apiKey;
  }

  /**
   * 设置基础URL
   */
  setBaseURL(baseURL: string): void {
    this.config.baseURL = baseURL;
  }

  /**
   * 启用/禁用调试模式
   */
  setDebug(debug: boolean): void {
    this.config.debug = debug;
  }
}

// ==================== 便捷方法 ====================

/**
 * 创建VoiceHelper SDK实例
 */
export function createVoiceHelperSDK(config: VoiceHelperConfig): VoiceHelperSDK {
  return new VoiceHelperSDK(config);
}

/**
 * 快速聊天方法
 */
export async function quickChat(
  apiKey: string,
  message: string,
  options: ChatCompletionOptions = {}
): Promise<string> {
  const sdk = new VoiceHelperSDK({ apiKey });
  const response = await sdk.createChatCompletion([
    { role: 'user', content: message }
  ], options);
  
  return response.choices[0]?.message?.content || '';
}

/**
 * 快速语音合成
 */
export async function quickTTS(
  apiKey: string,
  text: string,
  options: VoiceSynthesisOptions = {}
): Promise<ArrayBuffer | string> {
  const sdk = new VoiceHelperSDK({ apiKey });
  const result = await sdk.synthesizeVoice(text, options);
  
  return result.audioData || result.audioUrl || '';
}

/**
 * 快速图像分析
 */
export async function quickVision(
  apiKey: string,
  imageFile: File | Blob,
  query?: string
): Promise<string> {
  const sdk = new VoiceHelperSDK({ apiKey });
  const result = await sdk.analyzeImage(imageFile, { query });
  
  return result.description || '';
}

// ==================== 类型导出 ====================
export * from './types';

// 默认导出
export default VoiceHelperSDK;