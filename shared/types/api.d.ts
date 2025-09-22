/**
 * VoiceHelper API 类型定义
 * 统一的API接口类型，供前端、后端、SDK使用
 */

// ==================== 基础类型 ====================

export interface BaseResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ErrorInfo;
  message?: string;
  timestamp: string;
  trace_id?: string;
}

export interface ErrorInfo {
  code: string;
  message: string;
  details?: Record<string, any>;
  stack?: string;
}

export interface PaginationParams {
  page: number;
  page_size: number;
  total?: number;
}

export interface PaginatedResponse<T> extends BaseResponse<T[]> {
  pagination: {
    page: number;
    page_size: number;
    total: number;
    total_pages: number;
    has_next: boolean;
    has_prev: boolean;
  };
}

// ==================== 用户相关 ====================

export interface User {
  user_id: string;
  username: string;
  nickname?: string;
  email?: string;
  avatar_url?: string;
  created_at: string;
  updated_at: string;
  last_login?: string;
  status: 'active' | 'inactive' | 'banned';
  preferences?: UserPreferences;
}

export interface UserPreferences {
  language: string;
  theme: 'light' | 'dark' | 'auto';
  voice_settings: VoiceSettings;
  notification_settings: NotificationSettings;
}

export interface VoiceSettings {
  preferred_voice: string;
  speech_rate: number;
  volume: number;
  auto_play: boolean;
  voice_activation: boolean;
}

export interface NotificationSettings {
  email_notifications: boolean;
  push_notifications: boolean;
  sound_notifications: boolean;
}

// ==================== 会话相关 ====================

export interface Conversation {
  conversation_id: string;
  user_id: string;
  title?: string;
  status: 'active' | 'ended' | 'archived';
  created_at: string;
  updated_at: string;
  ended_at?: string;
  message_count: number;
  metadata?: ConversationMetadata;
}

export interface ConversationMetadata {
  tags?: string[];
  category?: string;
  priority?: 'low' | 'normal' | 'high';
  custom_fields?: Record<string, any>;
}

export interface Message {
  message_id: string;
  conversation_id: string;
  user_id?: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  content_type: 'text' | 'audio' | 'image' | 'file' | 'tool_call' | 'tool_result';
  created_at: string;
  metadata?: MessageMetadata;
  attachments?: Attachment[];
  tool_calls?: ToolCall[];
  references?: Reference[];
}

export interface MessageMetadata {
  response_time_ms?: number;
  model_used?: string;
  token_count?: {
    input: number;
    output: number;
    total: number;
  };
  confidence_score?: number;
  emotion?: EmotionAnalysis;
  intent?: IntentAnalysis;
}

export interface Attachment {
  attachment_id: string;
  filename: string;
  content_type: string;
  size: number;
  url: string;
  thumbnail_url?: string;
}

// ==================== 语音相关 ====================

export interface VoiceSession {
  session_id: string;
  user_id: string;
  conversation_id?: string;
  status: 'active' | 'paused' | 'ended';
  created_at: string;
  ended_at?: string;
  settings: VoiceSessionSettings;
  metrics?: VoiceMetrics;
}

export interface VoiceSessionSettings {
  language: string;
  voice_id: string;
  sample_rate: number;
  channels: number;
  format: 'pcm' | 'opus' | 'mp3' | 'wav';
  vad_enabled: boolean;
  noise_suppression: boolean;
  echo_cancellation: boolean;
}

export interface VoiceMetrics {
  total_duration_ms: number;
  speech_duration_ms: number;
  silence_duration_ms: number;
  interruption_count: number;
  average_latency_ms: number;
  audio_quality_score: number;
}

export interface AudioChunk {
  chunk_id: string;
  session_id: string;
  sequence: number;
  data: ArrayBuffer | string; // base64 encoded
  timestamp_ms: number;
  duration_ms: number;
  is_final: boolean;
}

export interface TranscriptionResult {
  text: string;
  confidence: number;
  language: string;
  is_final: boolean;
  alternatives?: TranscriptionAlternative[];
  word_timestamps?: WordTimestamp[];
}

export interface TranscriptionAlternative {
  text: string;
  confidence: number;
}

export interface WordTimestamp {
  word: string;
  start_time_ms: number;
  end_time_ms: number;
  confidence: number;
}

// ==================== 工具调用相关 ====================

export interface ToolCall {
  tool_call_id: string;
  tool_name: string;
  parameters: Record<string, any>;
  created_at: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: ToolResult;
  error?: ErrorInfo;
}

export interface ToolResult {
  success: boolean;
  data: any;
  message?: string;
  execution_time_ms: number;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, ParameterDefinition>;
    required?: string[];
  };
  examples?: ToolExample[];
}

export interface ParameterDefinition {
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  description: string;
  enum?: any[];
  default?: any;
  minimum?: number;
  maximum?: number;
  pattern?: string;
}

export interface ToolExample {
  description: string;
  parameters: Record<string, any>;
  expected_result: any;
}

// ==================== 分析相关 ====================

export interface EmotionAnalysis {
  primary_emotion: string;
  confidence: number;
  emotions: Record<string, number>;
  valence: number; // -1 to 1
  arousal: number; // 0 to 1
}

export interface IntentAnalysis {
  intent: string;
  confidence: number;
  entities: Entity[];
  slots: Record<string, any>;
}

export interface Entity {
  entity: string;
  value: string;
  confidence: number;
  start: number;
  end: number;
}

export interface Reference {
  type: 'document' | 'url' | 'conversation' | 'tool_result';
  id: string;
  title?: string;
  url?: string;
  snippet?: string;
  relevance_score?: number;
}

// ==================== 数据集相关 ====================

export interface Dataset {
  dataset_id: string;
  name: string;
  description?: string;
  type: 'knowledge_base' | 'training_data' | 'evaluation_set';
  status: 'active' | 'processing' | 'inactive' | 'error';
  created_at: string;
  updated_at: string;
  document_count: number;
  size_bytes: number;
  metadata?: DatasetMetadata;
}

export interface DatasetMetadata {
  tags?: string[];
  category?: string;
  language?: string;
  version?: string;
  source?: string;
  custom_fields?: Record<string, any>;
}

export interface Document {
  document_id: string;
  dataset_id: string;
  title: string;
  content: string;
  content_type: string;
  url?: string;
  created_at: string;
  updated_at: string;
  metadata?: DocumentMetadata;
  chunks?: DocumentChunk[];
}

export interface DocumentMetadata {
  author?: string;
  source?: string;
  language?: string;
  tags?: string[];
  custom_fields?: Record<string, any>;
}

export interface DocumentChunk {
  chunk_id: string;
  document_id: string;
  content: string;
  start_index: number;
  end_index: number;
  embedding?: number[];
  metadata?: ChunkMetadata;
}

export interface ChunkMetadata {
  section?: string;
  page_number?: number;
  relevance_score?: number;
  custom_fields?: Record<string, any>;
}

// ==================== 系统监控相关 ====================

export interface SystemMetrics {
  timestamp: string;
  service: string;
  metrics: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    network_io: NetworkIO;
    request_count: number;
    error_count: number;
    average_response_time: number;
  };
}

export interface NetworkIO {
  bytes_sent: number;
  bytes_received: number;
  packets_sent: number;
  packets_received: number;
}

export interface HealthCheck {
  service: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  checks: HealthCheckItem[];
}

export interface HealthCheckItem {
  name: string;
  status: 'pass' | 'fail' | 'warn';
  message?: string;
  duration_ms?: number;
}

// ==================== API请求/响应类型 ====================

// 聊天相关
export interface ChatRequest {
  message: string;
  conversation_id?: string;
  stream?: boolean;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  tools?: ToolDefinition[];
}

export interface ChatResponse extends BaseResponse<Message> {}

export interface ChatStreamChunk {
  type: 'message' | 'tool_call' | 'error' | 'done';
  data: any;
  timestamp: string;
}

// 语音相关
export interface VoiceStartRequest {
  language?: string;
  voice_id?: string;
  conversation_id?: string;
  settings?: Partial<VoiceSessionSettings>;
}

export interface VoiceStartResponse extends BaseResponse<VoiceSession> {}

export interface VoiceEndRequest {
  session_id: string;
}

export interface VoiceEndResponse extends BaseResponse<VoiceMetrics> {}

// 数据集相关
export interface DatasetCreateRequest {
  name: string;
  description?: string;
  type: Dataset['type'];
  metadata?: DatasetMetadata;
}

export interface DatasetCreateResponse extends BaseResponse<Dataset> {}

export interface DocumentUploadRequest {
  dataset_id: string;
  title: string;
  content?: string;
  file?: File;
  url?: string;
  metadata?: DocumentMetadata;
}

export interface DocumentUploadResponse extends BaseResponse<Document> {}

// 搜索相关
export interface SearchRequest {
  query: string;
  dataset_ids?: string[];
  limit?: number;
  threshold?: number;
  filters?: Record<string, any>;
}

export interface SearchResponse extends BaseResponse<SearchResult[]> {}

export interface SearchResult {
  document_id: string;
  chunk_id?: string;
  title: string;
  content: string;
  relevance_score: number;
  metadata?: Record<string, any>;
}
