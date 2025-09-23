// VoiceHelper Frontend Types
// 统一的类型定义文件

// ============= 基础类型 =============

export interface BaseResponse {
  status: string;
  timestamp: string;
  trace_id?: string;
}

export interface ErrorResponse extends BaseResponse {
  error: {
    code: string;
    message: string;
    details?: any;
  };
}

// ============= 聊天相关类型 =============

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    session_id?: string;
    trace_id?: string;
    references?: Reference[];
    agent_events?: AgentEvent[];
  };
}

export interface Reference {
  id: string;
  title: string;
  content: string;
  source: string;
  score: number;
  page?: number;
  url?: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  context?: Record<string, any>;
  use_rag?: boolean;
  use_agent?: boolean;
}

export interface ChatStreamEvent {
  type: 'delta' | 'references' | 'agent_event' | 'done' | 'error';
  content?: string;
  data?: any;
  timestamp: string;
}

// ============= Agent相关类型 =============

export interface AgentEvent {
  type: 'plan' | 'step' | 'tool_call' | 'tool_result' | 'summary';
  content: string;
  timestamp: string;
  metadata?: {
    tool_name?: string;
    step_index?: number;
    confidence?: number;
  };
}

export interface AgentCapability {
  name: string;
  description: string;
  enabled: boolean;
  tools: string[];
}

// ============= 语音相关类型 =============

export interface VoiceConfig {
  language: string;
  model: string;
  enable_vad: boolean;
  enable_noise: boolean;
  sample_rate: number;
  channels: number;
}

export interface VoiceMessage {
  type: 'start' | 'audio' | 'stop' | 'config' | 'ping';
  session_id?: string;
  data?: string | VoiceConfig | AudioData;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface AudioData {
  format: 'pcm16' | 'opus' | 'mp3' | 'wav';
  sample_rate: number;
  channels: number;
  data: ArrayBuffer | string; // ArrayBuffer for binary, string for base64
  sequence?: number;
}

export interface VoiceResponse {
  type: 'asr_partial' | 'asr_final' | 'llm_delta' | 'tts_chunk' | 'session_started' | 'session_stopped' | 'error';
  data?: any;
  timestamp: string;
  session_id?: string;
}

export interface TranscriptionResult {
  text: string;
  confidence: number;
  language: string;
  is_final: boolean;
  timestamp: string;
}

export interface TTSResult {
  audio_url?: string;
  audio_data?: ArrayBuffer;
  format: string;
  duration: number;
  timestamp: string;
}

// ============= WebRTC相关类型 =============

export interface WebRTCConfig {
  iceServers: RTCIceServer[];
  room_id: string;
  client_id: string;
}

export interface SignalingMessage {
  type: 'offer' | 'answer' | 'ice_candidate' | 'join_room' | 'leave_room' | 'connected' | 'client_joined' | 'client_left' | 'ping' | 'pong' | 'error';
  data?: any;
  from?: string;
  to?: string;
  room_id?: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface ICECandidate {
  candidate: string;
  sdpMid: string;
  sdpMLineIndex: number;
}

export interface SessionDescription {
  type: 'offer' | 'answer';
  sdp: string;
}

// ============= 对话设计器相关类型 =============

export interface DialogNode {
  id: string;
  type: 'start' | 'intent' | 'condition' | 'action' | 'response' | 'end';
  position: { x: number; y: number };
  data: {
    label: string;
    config?: Record<string, any>;
  };
}

export interface DialogEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
  data?: {
    condition?: string;
    label?: string;
  };
}

export interface DialogFlow {
  id: string;
  name: string;
  description?: string;
  nodes: DialogNode[];
  edges: DialogEdge[];
  metadata: {
    created_at: string;
    updated_at: string;
    version: string;
  };
}

// ============= 监控和指标类型 =============

export interface LatencyMetrics {
  capture_to_asr: number;
  asr_to_llm: number;
  llm_to_tts: number;
  tts_to_play: number;
  end_to_end: number;
  timestamp: string;
}

export interface VoiceMetrics {
  active_sessions: number;
  total_audio_bytes: number;
  average_latency: number;
  error_rate: number;
  barge_in_count: number;
  timestamp: string;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    gateway: ServiceStatus;
    algorithm: ServiceStatus;
    database: ServiceStatus;
    cache: ServiceStatus;
  };
  metrics: {
    cpu_usage: number;
    memory_usage: number;
    active_connections: number;
  };
  timestamp: string;
}

export interface ServiceStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  response_time: number;
  error_rate: number;
  last_check: string;
}

// ============= UI组件类型 =============

export interface ButtonProps {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  disabled?: boolean;
  loading?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
}

export interface InputProps {
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  error?: string;
  className?: string;
}

export interface CardProps {
  title?: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
  actions?: React.ReactNode;
}

// ============= Hook类型 =============

export interface UseWebSocketOptions {
  url: string;
  protocols?: string[];
  onOpen?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export interface UseWebSocketReturn {
  socket: WebSocket | null;
  connectionState: 'Connecting' | 'Open' | 'Closing' | 'Closed';
  sendMessage: (message: any) => void;
  disconnect: () => void;
  reconnect: () => void;
}

export interface UseVoiceOptions {
  config: VoiceConfig;
  onTranscription?: (result: TranscriptionResult) => void;
  onError?: (error: Error) => void;
  autoStart?: boolean;
}

export interface UseVoiceReturn {
  isRecording: boolean;
  isConnected: boolean;
  startRecording: () => void;
  stopRecording: () => void;
  sendAudio: (audioData: AudioData) => void;
  latencyMetrics: LatencyMetrics | null;
}

// ============= API响应类型 =============

export interface QueryResponse extends BaseResponse {
  response: string;
  session_id: string;
  metadata?: {
    use_rag: boolean;
    use_agent: boolean;
    references?: Reference[];
    agent_events?: AgentEvent[];
  };
}

export interface IngestResponse extends BaseResponse {
  document_id: string;
  message: string;
  chunks_created?: number;
  index_size?: number;
}

export interface SearchResponse extends BaseResponse {
  query: string;
  results: Reference[];
  total: number;
}

export interface HealthResponse extends BaseResponse {
  service: string;
  uptime_seconds: number;
  components: Record<string, string>;
  features?: Record<string, boolean>;
}

// ============= 错误类型 =============

export class VoiceHelperError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message);
    this.name = 'VoiceHelperError';
  }
}

export class APIError extends VoiceHelperError {
  constructor(
    message: string,
    public status: number,
    code: string,
    details?: any
  ) {
    super(message, code, details);
    this.name = 'APIError';
  }
}

export class WebSocketError extends VoiceHelperError {
  constructor(
    message: string,
    code: string,
    public event?: Event
  ) {
    super(message, code, event);
    this.name = 'WebSocketError';
  }
}

export class AudioError extends VoiceHelperError {
  constructor(
    message: string,
    code: string,
    public audioContext?: AudioContext
  ) {
    super(message, code, audioContext);
    this.name = 'AudioError';
  }
}

// ============= 常量类型 =============

export const VOICE_EVENTS = {
  START: 'start',
  AUDIO: 'audio',
  STOP: 'stop',
  CONFIG: 'config',
  PING: 'ping'
} as const;

export const CHAT_EVENTS = {
  DELTA: 'delta',
  REFERENCES: 'references',
  AGENT_EVENT: 'agent_event',
  DONE: 'done',
  ERROR: 'error'
} as const;

export const WEBRTC_EVENTS = {
  OFFER: 'offer',
  ANSWER: 'answer',
  ICE_CANDIDATE: 'ice_candidate',
  JOIN_ROOM: 'join_room',
  LEAVE_ROOM: 'leave_room',
  CONNECTED: 'connected',
  CLIENT_JOINED: 'client_joined',
  CLIENT_LEFT: 'client_left',
  PING: 'ping',
  PONG: 'pong',
  ERROR: 'error'
} as const;

export type VoiceEventType = typeof VOICE_EVENTS[keyof typeof VOICE_EVENTS];
export type ChatEventType = typeof CHAT_EVENTS[keyof typeof CHAT_EVENTS];
export type WebRTCEventType = typeof WEBRTC_EVENTS[keyof typeof WEBRTC_EVENTS];
