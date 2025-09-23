/**
 * VoiceHelper 事件类型定义
 * 统一的事件系统类型，用于WebSocket、SSE、内部事件总线
 */

// ==================== 基础事件类型 ====================

export interface BaseEvent {
  event_id: string;
  event_type: string;
  timestamp: string;
  source: string;
  trace_id?: string;
  user_id?: string;
  session_id?: string;
  conversation_id?: string;
}

export interface EventPayload<T = any> extends BaseEvent {
  data: T;
  metadata?: Record<string, any>;
}

// ==================== 连接事件 ====================

export interface ConnectionEvent extends BaseEvent {
  event_type: 'connection.opened' | 'connection.closed' | 'connection.error' | 'connection.reconnect';
}

export interface ConnectionOpenedEvent extends ConnectionEvent {
  event_type: 'connection.opened';
  data: {
    connection_id: string;
    protocol: 'websocket' | 'sse' | 'webrtc';
    client_info: ClientInfo;
  };
}

export interface ConnectionClosedEvent extends ConnectionEvent {
  event_type: 'connection.closed';
  data: {
    connection_id: string;
    reason: string;
    code?: number;
    duration_ms: number;
  };
}

export interface ConnectionErrorEvent extends ConnectionEvent {
  event_type: 'connection.error';
  data: {
    connection_id: string;
    error: ErrorInfo;
    is_recoverable: boolean;
  };
}

export interface ClientInfo {
  user_agent: string;
  platform: string;
  version: string;
  capabilities: string[];
  network_type?: string;
}

// ==================== 聊天事件 ====================

export interface ChatEvent extends BaseEvent {
  event_type: 'chat.message.start' | 'chat.message.chunk' | 'chat.message.end' | 'chat.message.error' | 'chat.typing.start' | 'chat.typing.stop';
}

export interface ChatMessageStartEvent extends ChatEvent {
  event_type: 'chat.message.start';
  data: {
    message_id: string;
    role: 'user' | 'assistant';
    model?: string;
    estimated_tokens?: number;
  };
}

export interface ChatMessageChunkEvent extends ChatEvent {
  event_type: 'chat.message.chunk';
  data: {
    message_id: string;
    chunk: string;
    chunk_index: number;
    is_final: boolean;
    delta?: MessageDelta;
  };
}

export interface ChatMessageEndEvent extends ChatEvent {
  event_type: 'chat.message.end';
  data: {
    message_id: string;
    message: Message;
    metrics: MessageMetrics;
  };
}

export interface ChatMessageErrorEvent extends ChatEvent {
  event_type: 'chat.message.error';
  data: {
    message_id: string;
    error: ErrorInfo;
    retry_count: number;
    is_retryable: boolean;
  };
}

export interface ChatTypingEvent extends ChatEvent {
  event_type: 'chat.typing.start' | 'chat.typing.stop';
  data: {
    user_id: string;
    typing_indicator: boolean;
  };
}

export interface MessageDelta {
  content?: string;
  tool_calls?: ToolCallDelta[];
  metadata?: Partial<MessageMetadata>;
}

export interface ToolCallDelta {
  tool_call_id: string;
  tool_name?: string;
  parameters?: Record<string, any>;
  status?: ToolCall['status'];
}

export interface MessageMetrics {
  processing_time_ms: number;
  token_count: {
    input: number;
    output: number;
    total: number;
  };
  model_used: string;
  cost_estimate?: number;
}

// ==================== 语音事件 ====================

export interface VoiceEvent extends BaseEvent {
  event_type: 'voice.session.start' | 'voice.session.end' | 'voice.audio.chunk' | 'voice.transcription' | 'voice.synthesis' | 'voice.vad' | 'voice.error' | 'voice.interrupt';
}

export interface VoiceSessionStartEvent extends VoiceEvent {
  event_type: 'voice.session.start';
  data: {
    session_id: string;
    settings: VoiceSessionSettings;
    capabilities: VoiceCapabilities;
  };
}

export interface VoiceSessionEndEvent extends VoiceEvent {
  event_type: 'voice.session.end';
  data: {
    session_id: string;
    duration_ms: number;
    metrics: VoiceMetrics;
    reason: 'user_ended' | 'timeout' | 'error' | 'system_shutdown';
  };
}

export interface VoiceAudioChunkEvent extends VoiceEvent {
  event_type: 'voice.audio.chunk';
  data: {
    session_id: string;
    chunk: AudioChunk;
    direction: 'input' | 'output';
  };
}

export interface VoiceTranscriptionEvent extends VoiceEvent {
  event_type: 'voice.transcription';
  data: {
    session_id: string;
    transcription: TranscriptionResult;
    audio_duration_ms: number;
    processing_time_ms: number;
  };
}

export interface VoiceSynthesisEvent extends VoiceEvent {
  event_type: 'voice.synthesis';
  data: {
    session_id: string;
    text: string;
    audio_chunk: AudioChunk;
    voice_id: string;
    processing_time_ms: number;
  };
}

export interface VoiceVADEvent extends VoiceEvent {
  event_type: 'voice.vad';
  data: {
    session_id: string;
    is_speech: boolean;
    confidence: number;
    timestamp_ms: number;
    energy_level: number;
  };
}

export interface VoiceInterruptEvent extends VoiceEvent {
  event_type: 'voice.interrupt';
  data: {
    session_id: string;
    interrupted_at_ms: number;
    reason: 'user_speech' | 'manual' | 'timeout';
    recovery_action: 'resume' | 'restart' | 'cancel';
  };
}

export interface VoiceErrorEvent extends VoiceEvent {
  event_type: 'voice.error';
  data: {
    session_id: string;
    error: ErrorInfo;
    component: 'asr' | 'tts' | 'vad' | 'audio_processing' | 'network';
    is_recoverable: boolean;
  };
}

export interface VoiceCapabilities {
  supported_languages: string[];
  supported_voices: VoiceInfo[];
  supported_formats: AudioFormat[];
  features: VoiceFeature[];
}

export interface VoiceInfo {
  voice_id: string;
  name: string;
  language: string;
  gender: 'male' | 'female' | 'neutral';
  age_group: 'child' | 'adult' | 'elderly';
  style: string[];
}

export interface AudioFormat {
  format: string;
  sample_rate: number;
  channels: number;
  bit_depth: number;
}

export interface VoiceFeature {
  name: string;
  description: string;
  enabled: boolean;
  parameters?: Record<string, any>;
}

// ==================== 工具调用事件 ====================

export interface ToolEvent extends BaseEvent {
  event_type: 'tool.call.start' | 'tool.call.progress' | 'tool.call.end' | 'tool.call.error';
}

export interface ToolCallStartEvent extends ToolEvent {
  event_type: 'tool.call.start';
  data: {
    tool_call_id: string;
    tool_name: string;
    parameters: Record<string, any>;
    estimated_duration_ms?: number;
  };
}

export interface ToolCallProgressEvent extends ToolEvent {
  event_type: 'tool.call.progress';
  data: {
    tool_call_id: string;
    progress: number; // 0-100
    status_message: string;
    intermediate_results?: any;
  };
}

export interface ToolCallEndEvent extends ToolEvent {
  event_type: 'tool.call.end';
  data: {
    tool_call_id: string;
    result: ToolResult;
    execution_time_ms: number;
  };
}

export interface ToolCallErrorEvent extends ToolEvent {
  event_type: 'tool.call.error';
  data: {
    tool_call_id: string;
    error: ErrorInfo;
    retry_count: number;
    max_retries: number;
  };
}

// ==================== 系统事件 ====================

export interface SystemEvent extends BaseEvent {
  event_type: 'system.health.check' | 'system.metrics.update' | 'system.alert' | 'system.maintenance' | 'system.config.update';
}

export interface SystemHealthCheckEvent extends SystemEvent {
  event_type: 'system.health.check';
  data: {
    service: string;
    health_check: HealthCheck;
    previous_status?: string;
  };
}

export interface SystemMetricsUpdateEvent extends SystemEvent {
  event_type: 'system.metrics.update';
  data: {
    service: string;
    metrics: SystemMetrics;
    alerts?: SystemAlert[];
  };
}

export interface SystemAlertEvent extends SystemEvent {
  event_type: 'system.alert';
  data: {
    alert: SystemAlert;
    affected_services: string[];
    recommended_actions: string[];
  };
}

export interface SystemMaintenanceEvent extends SystemEvent {
  event_type: 'system.maintenance';
  data: {
    maintenance_type: 'scheduled' | 'emergency';
    affected_services: string[];
    start_time: string;
    estimated_duration_ms: number;
    status: 'scheduled' | 'in_progress' | 'completed' | 'cancelled';
  };
}

export interface SystemConfigUpdateEvent extends SystemEvent {
  event_type: 'system.config.update';
  data: {
    config_key: string;
    old_value: any;
    new_value: any;
    updated_by: string;
    requires_restart: boolean;
  };
}

export interface SystemAlert {
  alert_id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  description: string;
  created_at: string;
  resolved_at?: string;
  tags: string[];
  metadata?: Record<string, any>;
}

// ==================== 用户事件 ====================

export interface UserEvent extends BaseEvent {
  event_type: 'user.login' | 'user.logout' | 'user.register' | 'user.profile.update' | 'user.preferences.update' | 'user.activity';
}

export interface UserLoginEvent extends UserEvent {
  event_type: 'user.login';
  data: {
    user_id: string;
    login_method: 'password' | 'oauth' | 'sso' | 'api_key';
    ip_address: string;
    user_agent: string;
    success: boolean;
    failure_reason?: string;
  };
}

export interface UserLogoutEvent extends UserEvent {
  event_type: 'user.logout';
  data: {
    user_id: string;
    session_duration_ms: number;
    logout_type: 'manual' | 'timeout' | 'forced';
  };
}

export interface UserActivityEvent extends UserEvent {
  event_type: 'user.activity';
  data: {
    user_id: string;
    activity_type: string;
    activity_data: Record<string, any>;
    ip_address: string;
    user_agent: string;
  };
}

// ==================== 数据事件 ====================

export interface DataEvent extends BaseEvent {
  event_type: 'data.document.upload' | 'data.document.process' | 'data.document.index' | 'data.search.query' | 'data.backup' | 'data.sync';
}

export interface DocumentUploadEvent extends DataEvent {
  event_type: 'data.document.upload';
  data: {
    document_id: string;
    dataset_id: string;
    filename: string;
    size_bytes: number;
    content_type: string;
    upload_duration_ms: number;
  };
}

export interface DocumentProcessEvent extends DataEvent {
  event_type: 'data.document.process';
  data: {
    document_id: string;
    processing_stage: 'parsing' | 'chunking' | 'embedding' | 'indexing' | 'completed' | 'failed';
    progress: number; // 0-100
    chunks_created?: number;
    processing_time_ms?: number;
    error?: ErrorInfo;
  };
}

export interface SearchQueryEvent extends DataEvent {
  event_type: 'data.search.query';
  data: {
    query_id: string;
    query: string;
    dataset_ids: string[];
    result_count: number;
    search_time_ms: number;
    filters_applied: Record<string, any>;
  };
}

// ==================== 事件订阅和过滤 ====================

export interface EventSubscription {
  subscription_id: string;
  event_types: string[];
  filters?: EventFilter[];
  delivery_mode: 'push' | 'pull';
  endpoint?: string;
  batch_size?: number;
  max_retry_count?: number;
}

export interface EventFilter {
  field: string;
  operator: 'eq' | 'ne' | 'in' | 'nin' | 'gt' | 'gte' | 'lt' | 'lte' | 'contains' | 'regex';
  value: any;
}

export interface EventBatch {
  batch_id: string;
  events: BaseEvent[];
  created_at: string;
  expires_at: string;
}

// ==================== 事件处理结果 ====================

export interface EventProcessingResult {
  event_id: string;
  status: 'processed' | 'failed' | 'skipped' | 'retrying';
  processing_time_ms: number;
  error?: ErrorInfo;
  retry_count?: number;
  next_retry_at?: string;
}

export interface EventDeliveryResult {
  event_id: string;
  subscription_id: string;
  status: 'delivered' | 'failed' | 'pending';
  delivery_time_ms?: number;
  response_code?: number;
  error?: ErrorInfo;
  retry_count: number;
  next_retry_at?: string;
}

// ==================== 类型联合 ====================

export type AnyEvent = 
  | ConnectionEvent
  | ChatEvent
  | VoiceEvent
  | ToolEvent
  | SystemEvent
  | UserEvent
  | DataEvent;

export type EventData<T extends AnyEvent> = T extends { data: infer D } ? D : never;

// ==================== 事件工厂函数类型 ====================

export interface EventFactory {
  createEvent<T extends AnyEvent>(
    eventType: T['event_type'],
    data: EventData<T>,
    options?: Partial<BaseEvent>
  ): T;
}

// ==================== 事件处理器类型 ====================

export interface EventHandler<T extends AnyEvent = AnyEvent> {
  (event: T): Promise<void> | void;
}

export interface EventHandlerRegistry {
  register<T extends AnyEvent>(
    eventType: T['event_type'],
    handler: EventHandler<T>
  ): void;
  
  unregister<T extends AnyEvent>(
    eventType: T['event_type'],
    handler: EventHandler<T>
  ): void;
  
  emit<T extends AnyEvent>(event: T): Promise<void>;
}

// 导入相关类型
import type { 
  ErrorInfo, 
  Message, 
  MessageMetadata, 
  ToolCall, 
  ToolResult, 
  VoiceSessionSettings, 
  VoiceMetrics, 
  AudioChunk, 
  TranscriptionResult, 
  HealthCheck, 
  SystemMetrics 
} from './api';
