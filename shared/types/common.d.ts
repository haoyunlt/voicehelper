/**
 * VoiceHelper 通用类型定义
 * 基础类型、工具类型、常量等
 */

// ==================== 基础工具类型 ====================

export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type Maybe<T> = T | null | undefined;

export type NonEmptyArray<T> = [T, ...T[]];
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

export type KeysOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never;
}[keyof T];

export type Awaited<T> = T extends Promise<infer U> ? U : T;

export type Constructor<T = {}> = new (...args: any[]) => T;

export type Mixin<T extends Constructor> = T & Constructor;

// ==================== 字符串工具类型 ====================

export type StringLiteral<T> = T extends string ? (string extends T ? never : T) : never;

export type Split<S extends string, D extends string> = 
  S extends `${infer T}${D}${infer U}` ? [T, ...Split<U, D>] : [S];

export type Join<T extends readonly string[], D extends string> = 
  T extends readonly [infer F, ...infer R] 
    ? F extends string 
      ? R extends readonly string[]
        ? R['length'] extends 0 
          ? F 
          : `${F}${D}${Join<R, D>}`
        : never
      : never
    : '';

// ==================== 对象工具类型 ====================

export type Pick<T, K extends keyof T> = {
  [P in K]: T[P];
};

export type Omit<T, K extends keyof any> = Pick<T, Exclude<keyof T, K>>;

export type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type RequiredBy<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;

export type Merge<T, U> = Omit<T, keyof U> & U;

export type Override<T, U> = Omit<T, keyof U> & U;

// ==================== 函数工具类型 ====================

export type AsyncFunction<T extends (...args: any[]) => any> = 
  T extends (...args: infer A) => infer R 
    ? (...args: A) => Promise<R>
    : never;

export type SyncFunction<T extends (...args: any[]) => any> = 
  T extends (...args: infer A) => Promise<infer R>
    ? (...args: A) => R
    : T;

export type Parameters<T extends (...args: any) => any> = T extends (...args: infer P) => any ? P : never;

export type ReturnType<T extends (...args: any) => any> = T extends (...args: any) => infer R ? R : any;

// ==================== 时间相关类型 ====================

export type Timestamp = number; // Unix timestamp in milliseconds
export type Duration = number; // Duration in milliseconds
export type ISODateString = string; // ISO 8601 date string

export interface TimeRange {
  start: Timestamp;
  end: Timestamp;
}

export interface DateRange {
  start_date: ISODateString;
  end_date: ISODateString;
}

// ==================== 地理位置类型 ====================

export interface Coordinates {
  latitude: number;
  longitude: number;
  altitude?: number;
  accuracy?: number;
}

export interface Location {
  coordinates: Coordinates;
  address?: string;
  city?: string;
  country?: string;
  timezone?: string;
}

// ==================== 文件相关类型 ====================

export interface FileInfo {
  filename: string;
  size: number;
  mime_type: string;
  extension: string;
  checksum?: string;
  created_at: ISODateString;
  modified_at: ISODateString;
}

export interface FileUpload extends FileInfo {
  content: ArrayBuffer | string;
  encoding?: 'base64' | 'binary';
}

export interface FileReference {
  file_id: string;
  url: string;
  download_url?: string;
  thumbnail_url?: string;
  expires_at?: ISODateString;
}

// ==================== 网络相关类型 ====================

export interface NetworkInfo {
  ip_address: string;
  user_agent: string;
  referer?: string;
  origin?: string;
  connection_type?: 'wifi' | 'cellular' | 'ethernet' | 'unknown';
  bandwidth?: number; // Mbps
  latency?: number; // ms
}

export interface RequestContext {
  request_id: string;
  trace_id: string;
  user_id?: string;
  session_id?: string;
  timestamp: Timestamp;
  network: NetworkInfo;
  client_info: ClientInfo;
}

export interface ClientInfo {
  platform: 'web' | 'mobile' | 'desktop' | 'api';
  os: string;
  browser?: string;
  version: string;
  language: string;
  timezone: string;
  screen_resolution?: string;
  device_type?: 'phone' | 'tablet' | 'desktop' | 'tv' | 'watch';
}

// ==================== 配置相关类型 ====================

export interface ConfigValue {
  value: any;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  description?: string;
  default?: any;
  required?: boolean;
  validation?: ValidationRule[];
}

export interface ValidationRule {
  type: 'required' | 'min' | 'max' | 'pattern' | 'enum' | 'custom';
  value?: any;
  message?: string;
  validator?: (value: any) => boolean | string;
}

export interface ConfigSection {
  [key: string]: ConfigValue | ConfigSection;
}

export interface AppConfig {
  app: {
    name: string;
    version: string;
    environment: 'development' | 'staging' | 'production';
    debug: boolean;
  };
  server: {
    host: string;
    port: number;
    ssl: boolean;
    cors: {
      enabled: boolean;
      origins: string[];
      methods: string[];
      headers: string[];
    };
  };
  database: {
    host: string;
    port: number;
    name: string;
    username: string;
    password: string;
    ssl: boolean;
    pool_size: number;
    timeout: number;
  };
  redis: {
    host: string;
    port: number;
    password?: string;
    database: number;
    ttl: number;
  };
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    format: 'json' | 'text';
    output: 'console' | 'file' | 'both';
    file_path?: string;
    max_size?: number;
    max_files?: number;
  };
  monitoring: {
    enabled: boolean;
    metrics_endpoint: string;
    health_endpoint: string;
    tracing: {
      enabled: boolean;
      service_name: string;
      endpoint: string;
      sample_rate: number;
    };
  };
}

// ==================== 状态管理类型 ====================

export interface State<T = any> {
  data: T;
  loading: boolean;
  error: Error | null;
  last_updated: Timestamp;
}

export interface AsyncState<T = any> extends State<T> {
  promise?: Promise<T>;
  retry_count: number;
  max_retries: number;
}

export interface CacheState<T = any> {
  data: T;
  cached_at: Timestamp;
  expires_at: Timestamp;
  hit_count: number;
  miss_count: number;
}

// ==================== 权限相关类型 ====================

export interface Permission {
  resource: string;
  action: string;
  conditions?: PermissionCondition[];
}

export interface PermissionCondition {
  field: string;
  operator: 'eq' | 'ne' | 'in' | 'nin' | 'gt' | 'gte' | 'lt' | 'lte';
  value: any;
}

export interface Role {
  role_id: string;
  name: string;
  description?: string;
  permissions: Permission[];
  inherits_from?: string[];
}

export interface AccessControl {
  user_id: string;
  roles: string[];
  permissions: Permission[];
  context?: Record<string, any>;
}

// ==================== 队列和任务类型 ====================

export interface Task<T = any> {
  task_id: string;
  type: string;
  payload: T;
  priority: number;
  created_at: Timestamp;
  scheduled_at?: Timestamp;
  started_at?: Timestamp;
  completed_at?: Timestamp;
  failed_at?: Timestamp;
  retry_count: number;
  max_retries: number;
  timeout_ms: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  result?: any;
  error?: Error;
  metadata?: Record<string, any>;
}

export interface Queue {
  name: string;
  size: number;
  processing: number;
  completed: number;
  failed: number;
  waiting: number;
  active: boolean;
  paused: boolean;
  settings: QueueSettings;
}

export interface QueueSettings {
  concurrency: number;
  delay: number;
  attempts: number;
  backoff: {
    type: 'fixed' | 'exponential';
    delay: number;
  };
  remove_on_complete: number;
  remove_on_fail: number;
}

// ==================== 缓存类型 ====================

export interface CacheEntry<T = any> {
  key: string;
  value: T;
  created_at: Timestamp;
  expires_at: Timestamp;
  accessed_at: Timestamp;
  access_count: number;
  size_bytes: number;
  tags?: string[];
}

export interface CacheStats {
  total_entries: number;
  total_size_bytes: number;
  hit_count: number;
  miss_count: number;
  hit_rate: number;
  eviction_count: number;
  expired_count: number;
}

export interface CacheConfig {
  max_size: number;
  max_entries: number;
  default_ttl: Duration;
  eviction_policy: 'lru' | 'lfu' | 'fifo' | 'ttl';
  compression: boolean;
  serialization: 'json' | 'msgpack' | 'protobuf';
}

// ==================== 度量和统计类型 ====================

export interface Metric {
  name: string;
  value: number;
  unit: string;
  timestamp: Timestamp;
  tags?: Record<string, string>;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
}

export interface Counter extends Metric {
  type: 'counter';
  increment?: number;
}

export interface Gauge extends Metric {
  type: 'gauge';
}

export interface Histogram extends Metric {
  type: 'histogram';
  buckets: number[];
  counts: number[];
}

export interface Summary extends Metric {
  type: 'summary';
  quantiles: Record<string, number>; // e.g., { "0.5": 100, "0.95": 200, "0.99": 300 }
  count: number;
  sum: number;
}

export interface Statistics {
  count: number;
  sum: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  std_dev: number;
  percentiles: Record<string, number>;
}

// ==================== 常量类型 ====================

export const HTTP_STATUS_CODES = {
  OK: 200,
  CREATED: 201,
  ACCEPTED: 202,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  METHOD_NOT_ALLOWED: 405,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
  GATEWAY_TIMEOUT: 504,
} as const;

export type HttpStatusCode = typeof HTTP_STATUS_CODES[keyof typeof HTTP_STATUS_CODES];

export const LOG_LEVELS = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
  FATAL: 4,
} as const;

export type LogLevel = keyof typeof LOG_LEVELS;

export const ENVIRONMENTS = ['development', 'staging', 'production'] as const;
export type Environment = typeof ENVIRONMENTS[number];

export const PLATFORMS = ['web', 'mobile', 'desktop', 'api'] as const;
export type Platform = typeof PLATFORMS[number];

export const CONTENT_TYPES = {
  JSON: 'application/json',
  XML: 'application/xml',
  HTML: 'text/html',
  TEXT: 'text/plain',
  CSV: 'text/csv',
  PDF: 'application/pdf',
  IMAGE_JPEG: 'image/jpeg',
  IMAGE_PNG: 'image/png',
  IMAGE_GIF: 'image/gif',
  AUDIO_MP3: 'audio/mpeg',
  AUDIO_WAV: 'audio/wav',
  AUDIO_OGG: 'audio/ogg',
  VIDEO_MP4: 'video/mp4',
  VIDEO_WEBM: 'video/webm',
} as const;

export type ContentType = typeof CONTENT_TYPES[keyof typeof CONTENT_TYPES];

// ==================== 品牌和主题类型 ====================

export interface Theme {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    text_secondary: string;
    border: string;
    error: string;
    warning: string;
    success: string;
    info: string;
  };
  typography: {
    font_family: string;
    font_sizes: Record<string, string>;
    font_weights: Record<string, number>;
    line_heights: Record<string, number>;
  };
  spacing: Record<string, string>;
  breakpoints: Record<string, string>;
  shadows: Record<string, string>;
  border_radius: Record<string, string>;
}

export interface Brand {
  name: string;
  logo_url: string;
  favicon_url: string;
  primary_color: string;
  secondary_color: string;
  font_family: string;
  tagline?: string;
  description?: string;
}

// ==================== 国际化类型 ====================

export interface Locale {
  code: string; // e.g., 'en-US', 'zh-CN'
  name: string;
  native_name: string;
  direction: 'ltr' | 'rtl';
  currency: string;
  date_format: string;
  time_format: string;
  number_format: {
    decimal_separator: string;
    thousands_separator: string;
    currency_symbol: string;
    currency_position: 'before' | 'after';
  };
}

export interface Translation {
  key: string;
  value: string;
  locale: string;
  namespace?: string;
  context?: string;
  pluralization?: Record<string, string>;
}

export interface I18nConfig {
  default_locale: string;
  supported_locales: string[];
  fallback_locale: string;
  load_path: string;
  interpolation: {
    prefix: string;
    suffix: string;
    escape_value: boolean;
  };
}

// ==================== 导出所有类型 ====================

export * from './api';
export * from './events';
