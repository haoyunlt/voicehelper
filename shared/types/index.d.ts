/**
 * VoiceHelper 类型定义入口文件
 * 统一导出所有类型定义
 */

// 导出API相关类型
export * from './api';

// 导出事件相关类型
export * from './events';

// 导出通用类型
export * from './common';

// 版本信息
export const TYPES_VERSION = '1.0.0';

// 类型检查工具函数
export function isBaseResponse(obj: any): obj is import('./api').BaseResponse {
  return obj && typeof obj === 'object' && typeof obj.success === 'boolean';
}

export function isErrorInfo(obj: any): obj is import('./api').ErrorInfo {
  return obj && typeof obj === 'object' && typeof obj.code === 'string' && typeof obj.message === 'string';
}

export function isUser(obj: any): obj is import('./api').User {
  return obj && typeof obj === 'object' && typeof obj.user_id === 'string' && typeof obj.username === 'string';
}

export function isMessage(obj: any): obj is import('./api').Message {
  return obj && typeof obj === 'object' && 
         typeof obj.message_id === 'string' && 
         typeof obj.conversation_id === 'string' &&
         ['user', 'assistant', 'system', 'tool'].includes(obj.role) &&
         typeof obj.content === 'string';
}

export function isVoiceSession(obj: any): obj is import('./api').VoiceSession {
  return obj && typeof obj === 'object' && 
         typeof obj.session_id === 'string' && 
         typeof obj.user_id === 'string' &&
         ['active', 'paused', 'ended'].includes(obj.status);
}

export function isBaseEvent(obj: any): obj is import('./events').BaseEvent {
  return obj && typeof obj === 'object' && 
         typeof obj.event_id === 'string' && 
         typeof obj.event_type === 'string' &&
         typeof obj.timestamp === 'string' &&
         typeof obj.source === 'string';
}

export function isChatEvent(obj: any): obj is import('./events').ChatEvent {
  return isBaseEvent(obj) && obj.event_type.startsWith('chat.');
}

export function isVoiceEvent(obj: any): obj is import('./events').VoiceEvent {
  return isBaseEvent(obj) && obj.event_type.startsWith('voice.');
}

export function isSystemEvent(obj: any): obj is import('./events').SystemEvent {
  return isBaseEvent(obj) && obj.event_type.startsWith('system.');
}

// 类型断言辅助函数
export function assertIsBaseResponse(obj: any): asserts obj is import('./api').BaseResponse {
  if (!isBaseResponse(obj)) {
    throw new Error('Object is not a BaseResponse');
  }
}

export function assertIsUser(obj: any): asserts obj is import('./api').User {
  if (!isUser(obj)) {
    throw new Error('Object is not a User');
  }
}

export function assertIsMessage(obj: any): asserts obj is import('./api').Message {
  if (!isMessage(obj)) {
    throw new Error('Object is not a Message');
  }
}

export function assertIsBaseEvent(obj: any): asserts obj is import('./events').BaseEvent {
  if (!isBaseEvent(obj)) {
    throw new Error('Object is not a BaseEvent');
  }
}

// 类型转换工具函数
export function toBaseResponse<T>(data: T, success: boolean = true, message?: string): import('./api').BaseResponse<T> {
  return {
    success,
    data: success ? data : undefined,
    error: success ? undefined : { code: 'UNKNOWN_ERROR', message: message || 'An error occurred' },
    message,
    timestamp: new Date().toISOString(),
  };
}

export function toErrorResponse(error: import('./api').ErrorInfo, message?: string): import('./api').BaseResponse {
  return {
    success: false,
    error,
    message: message || error.message,
    timestamp: new Date().toISOString(),
  };
}

export function createEvent<T extends import('./events').BaseEvent>(
  eventType: T['event_type'],
  data: any,
  options: Partial<import('./events').BaseEvent> = {}
): T {
  return {
    event_id: options.event_id || generateEventId(),
    event_type: eventType,
    timestamp: options.timestamp || new Date().toISOString(),
    source: options.source || 'voicehelper',
    trace_id: options.trace_id,
    user_id: options.user_id,
    session_id: options.session_id,
    conversation_id: options.conversation_id,
    data,
  } as T;
}

// 工具函数
function generateEventId(): string {
  return `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// 常用类型别名
export type ApiResponse<T = any> = import('./api').BaseResponse<T>;
export type ApiError = import('./api').ErrorInfo;
export type ChatMessage = import('./api').Message;
export type VoiceSessionType = import('./api').VoiceSession;
export type EventType = import('./events').AnyEvent;
export type UserType = import('./api').User;
export type ConversationType = import('./api').Conversation;
export type DocumentType = import('./api').Document;
export type DatasetType = import('./api').Dataset;

// 枚举类型
export enum MessageRole {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system',
  TOOL = 'tool',
}

export enum MessageContentType {
  TEXT = 'text',
  AUDIO = 'audio',
  IMAGE = 'image',
  FILE = 'file',
  TOOL_CALL = 'tool_call',
  TOOL_RESULT = 'tool_result',
}

export enum VoiceSessionStatus {
  ACTIVE = 'active',
  PAUSED = 'paused',
  ENDED = 'ended',
}

export enum ConversationStatus {
  ACTIVE = 'active',
  ENDED = 'ended',
  ARCHIVED = 'archived',
}

export enum UserStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  BANNED = 'banned',
}

export enum DatasetType {
  KNOWLEDGE_BASE = 'knowledge_base',
  TRAINING_DATA = 'training_data',
  EVALUATION_SET = 'evaluation_set',
}

export enum ToolCallStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

export enum SystemAlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical',
}

export enum HealthStatus {
  HEALTHY = 'healthy',
  UNHEALTHY = 'unhealthy',
  DEGRADED = 'degraded',
}

// 配置类型
export interface TypesConfig {
  version: string;
  strict_mode: boolean;
  validation_enabled: boolean;
  debug_mode: boolean;
}

export const DEFAULT_TYPES_CONFIG: TypesConfig = {
  version: TYPES_VERSION,
  strict_mode: true,
  validation_enabled: true,
  debug_mode: false,
};

// 验证配置
export interface ValidationConfig {
  max_string_length: number;
  max_array_length: number;
  max_object_depth: number;
  required_fields: string[];
  custom_validators: Record<string, (value: any) => boolean>;
}

export const DEFAULT_VALIDATION_CONFIG: ValidationConfig = {
  max_string_length: 10000,
  max_array_length: 1000,
  max_object_depth: 10,
  required_fields: [],
  custom_validators: {},
};

// 类型元数据
export interface TypeMetadata {
  name: string;
  version: string;
  description: string;
  properties: Record<string, PropertyMetadata>;
  examples: any[];
  deprecated?: boolean;
  since?: string;
}

export interface PropertyMetadata {
  type: string;
  description: string;
  required: boolean;
  default?: any;
  validation?: ValidationRule[];
  deprecated?: boolean;
  since?: string;
}

export interface ValidationRule {
  type: string;
  value?: any;
  message?: string;
}

// 导出类型元数据
export const API_TYPES_METADATA: Record<string, TypeMetadata> = {
  BaseResponse: {
    name: 'BaseResponse',
    version: '1.0.0',
    description: 'Standard API response wrapper',
    properties: {
      success: { type: 'boolean', description: 'Indicates if the request was successful', required: true },
      data: { type: 'any', description: 'Response data', required: false },
      error: { type: 'ErrorInfo', description: 'Error information if request failed', required: false },
      message: { type: 'string', description: 'Human-readable message', required: false },
      timestamp: { type: 'string', description: 'ISO timestamp of response', required: true },
      trace_id: { type: 'string', description: 'Request trace ID for debugging', required: false },
    },
    examples: [
      { success: true, data: { id: 1, name: 'test' }, timestamp: '2023-01-01T00:00:00Z' },
      { success: false, error: { code: 'NOT_FOUND', message: 'Resource not found' }, timestamp: '2023-01-01T00:00:00Z' },
    ],
  },
  User: {
    name: 'User',
    version: '1.0.0',
    description: 'User account information',
    properties: {
      user_id: { type: 'string', description: 'Unique user identifier', required: true },
      username: { type: 'string', description: 'User login name', required: true },
      nickname: { type: 'string', description: 'Display name', required: false },
      email: { type: 'string', description: 'Email address', required: false },
      created_at: { type: 'string', description: 'Account creation timestamp', required: true },
      status: { type: 'string', description: 'Account status', required: true },
    },
    examples: [
      {
        user_id: 'usr_123',
        username: 'john_doe',
        nickname: 'John',
        email: 'john@example.com',
        created_at: '2023-01-01T00:00:00Z',
        status: 'active',
      },
    ],
  },
  // 可以继续添加其他类型的元数据...
};

// 版本兼容性
export interface VersionInfo {
  current: string;
  supported: string[];
  deprecated: string[];
  breaking_changes: Record<string, string[]>;
}

export const VERSION_INFO: VersionInfo = {
  current: TYPES_VERSION,
  supported: ['1.0.0'],
  deprecated: [],
  breaking_changes: {},
};
