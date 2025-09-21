/**
 * TypeScript类型定义
 */

// 基础配置
export interface ChatbotConfig {
  /** API密钥或JWT令牌 */
  apiKey: string;
  /** API基础URL */
  baseURL?: string;
  /** 请求超时时间(毫秒) */
  timeout?: number;
  /** 租户ID */
  tenantId?: string;
}

// 用户相关
export interface User {
  id: string;
  nickname: string;
  avatar?: string;
  created_at: string;
  last_login?: string;
}

export interface LoginRequest {
  code: string;
  nickname?: string;
  avatar?: string;
}

export interface LoginResponse {
  token: string;
  refresh_token: string;
  expires_at: string;
  user: User;
}

// 对话相关
export interface Conversation {
  id: string;
  title: string;
  summary?: string;
  status: 'active' | 'archived' | 'deleted';
  msg_count: number;
  token_count: number;
  created_at: string;
  updated_at: string;
  last_msg_at?: string;
  metadata?: Record<string, any>;
}

export interface CreateConversationRequest {
  title?: string;
  metadata?: Record<string, any>;
}

export interface UpdateConversationRequest {
  title?: string;
  metadata?: Record<string, any>;
}

// 消息相关
export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  modality: 'text' | 'voice' | 'image';
  token_count: number;
  references?: Reference[];
  created_at: string;
  metadata?: Record<string, any>;
}

export interface Reference {
  title: string;
  url?: string;
  snippet?: string;
  score?: number;
}

export interface SendMessageRequest {
  content: string;
  modality?: 'text' | 'voice';
  stream?: boolean;
  metadata?: Record<string, any>;
}

export interface StreamEvent {
  type: 'delta' | 'done' | 'error' | 'typing';
  content?: string;
  message_id?: string;
  seq?: number;
  error?: string;
}

// 语音相关
export interface TranscribeRequest {
  audio: File | Blob;
  language?: string;
}

export interface TranscribeResponse {
  text: string;
  confidence: number;
  duration: number;
}

export interface SynthesizeRequest {
  text: string;
  voice?: 'female' | 'male' | 'child';
  speed?: number;
}

// 数据集相关
export interface Dataset {
  id: string;
  name: string;
  description?: string;
  type: 'document' | 'qa' | 'custom';
  status: 'active' | 'inactive' | 'processing';
  doc_count: number;
  chunk_count: number;
  token_count: number;
  created_at: string;
  updated_at: string;
}

export interface CreateDatasetRequest {
  name: string;
  description?: string;
  type?: 'document' | 'qa' | 'custom';
}

export interface Document {
  id: string;
  dataset_id: string;
  name: string;
  source: string;
  type: string;
  size: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  chunk_count: number;
  token_count: number;
  created_at: string;
  updated_at: string;
}

export interface UploadDocumentRequest {
  file: File;
  name?: string;
}

// 搜索相关
export interface SearchRequest {
  query: string;
  dataset_ids?: string[];
  top_k?: number;
  threshold?: number;
}

export interface SearchResult {
  content: string;
  score: number;
  source: {
    document_id: string;
    document_name: string;
    chunk_id: string;
  };
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
}

// 分页相关
export interface PaginationParams {
  page?: number;
  limit?: number;
}

export interface Pagination {
  page: number;
  limit: number;
  total: number;
  pages: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: Pagination;
}

// API响应
export interface APIResponse<T = any> {
  data?: T;
  error?: string;
  code?: string;
  details?: Record<string, any>;
}

// 健康检查
export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  timestamp: number;
  service: string;
  version?: string;
}

// 事件回调
export type EventCallback<T = any> = (data: T) => void;
export type ErrorCallback = (error: Error) => void;

// WebSocket事件
export interface WebSocketEvents {
  'message': EventCallback<StreamEvent>;
  'error': ErrorCallback;
  'open': EventCallback<Event>;
  'close': EventCallback<CloseEvent>;
}
