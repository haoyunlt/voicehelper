/**
 * VoiceHelper SDK Types - v1.9.0
 * 完整的TypeScript类型定义
 */

// ==================== 基础类型 ====================

export type ModelType = 'gpt-4-turbo' | 'gpt-4-vision' | 'claude-3-opus' | 'doubao-pro';

export type MessageRole = 'system' | 'user' | 'assistant';

export type AttachmentType = 'image' | 'audio' | 'video' | 'file';

export type EmotionType = 'neutral' | 'happy' | 'sad' | 'angry' | 'excited' | 'calm';

export type VoiceType = 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer';

export type LanguageCode = 'zh-CN' | 'en-US' | 'ja-JP' | 'ko-KR';

export type ModalityType = 'text' | 'image' | 'audio' | 'video' | 'structured';

export type FusionStrategy = 'hierarchical' | 'adaptive_attention' | 'cross_transformer' | 'dynamic_weighting';

export type ServiceCategory = 
  | 'office_suite' 
  | 'development' 
  | 'social_platform' 
  | 'ecommerce' 
  | 'cloud_service' 
  | 'database' 
  | 'monitoring' 
  | 'security' 
  | 'ai_ml';

export type ServiceStatus = 'active' | 'inactive' | 'beta' | 'maintenance';

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

export type DatasetStatus = 'active' | 'processing' | 'error';

export type ConversationFilter = 'all' | 'active' | 'archived';

export type FinishReason = 'stop' | 'length' | 'content_filter';

// ==================== 配置接口 ====================

export interface VoiceHelperConfig {
  /** API密钥 */
  apiKey: string;
  /** API基础URL，默认为生产环境 */
  baseURL?: string;
  /** 请求超时时间（毫秒），默认30秒 */
  timeout?: number;
  /** 重试次数，默认3次 */
  retries?: number;
  /** 是否启用调试模式 */
  debug?: boolean;
}

export interface MultimodalConfig {
  /** 是否启用视觉理解 */
  enableVision?: boolean;
  /** 是否启用音频处理 */
  enableAudio?: boolean;
  /** 是否启用情感检测 */
  enableEmotionDetection?: boolean;
  /** 融合策略 */
  fusionStrategy?: FusionStrategy;
}

export interface VoiceSynthesisOptions {
  /** 语音类型 */
  voice?: VoiceType;
  /** 情感类型 */
  emotion?: EmotionType;
  /** 语速，0.25-4.0 */
  speed?: number;
  /** 语言 */
  language?: LanguageCode;
  /** 是否流式合成 */
  streaming?: boolean;
}

export interface VoiceRecognitionOptions {
  /** 识别语言 */
  language?: LanguageCode;
  /** 是否启用情感检测 */
  enableEmotionDetection?: boolean;
  /** 是否启用说话人分离 */
  enableSpeakerSeparation?: boolean;
  /** 是否启用噪声抑制 */
  noiseReduction?: boolean;
}

export interface VisionAnalysisOptions {
  /** 分析任务列表 */
  tasks?: VisionTask[];
  /** 用户查询 */
  query?: string;
}

export type VisionTask = 
  | 'description' 
  | 'object_detection' 
  | 'text_extraction' 
  | 'scene_analysis' 
  | 'emotion_detection' 
  | 'brand_recognition';

// ==================== 消息和附件 ====================

export interface ChatMessage {
  /** 消息角色 */
  role: MessageRole;
  /** 消息内容 */
  content: string;
  /** 附件列表 */
  attachments?: Attachment[];
}

export interface Attachment {
  /** 附件类型 */
  type: AttachmentType;
  /** 附件URL */
  url?: string;
  /** Base64编码的数据 */
  data?: string;
  /** 附件元数据 */
  metadata?: Record<string, any>;
}

export interface MessageDelta {
  /** 角色（仅在第一个chunk中） */
  role?: MessageRole;
  /** 增量内容 */
  content?: string;
}

// ==================== 聊天完成相关 ====================

export interface ChatCompletionOptions {
  /** 使用的模型 */
  model?: ModelType;
  /** 是否流式响应 */
  stream?: boolean;
  /** 最大token数 */
  maxTokens?: number;
  /** 温度参数，0-2 */
  temperature?: number;
  /** 多模态配置 */
  multimodalConfig?: MultimodalConfig;
}

export interface ChatCompletionResponse {
  /** 响应ID */
  id: string;
  /** 对象类型 */
  object: 'chat.completion';
  /** 创建时间戳 */
  created: number;
  /** 使用的模型 */
  model: string;
  /** 选择列表 */
  choices: Choice[];
  /** token使用情况 */
  usage: Usage;
  /** 处理指标 */
  processingMetrics: ProcessingMetrics;
}

export interface Choice {
  /** 选择索引 */
  index: number;
  /** 消息内容 */
  message: ChatMessage;
  /** 结束原因 */
  finishReason: FinishReason;
}

export interface Usage {
  /** 提示token数 */
  promptTokens: number;
  /** 完成token数 */
  completionTokens: number;
  /** 总token数 */
  totalTokens: number;
}

export interface ProcessingMetrics {
  /** 总处理时间（毫秒） */
  totalTimeMs: number;
  /** 语音处理时间（毫秒） */
  voiceProcessingTimeMs?: number;
  /** 视觉处理时间（毫秒） */
  visionProcessingTimeMs?: number;
  /** 融合处理时间（毫秒） */
  fusionTimeMs?: number;
  /** 模型推理时间（毫秒） */
  modelInferenceTimeMs?: number;
}

// ==================== 流式响应 ====================

export interface ChatCompletionStreamResponse {
  /** 响应ID */
  id: string;
  /** 对象类型 */
  object: 'chat.completion.chunk';
  /** 创建时间戳 */
  created: number;
  /** 使用的模型 */
  model: string;
  /** 流式选择 */
  choices: StreamChoice[];
}

export interface StreamChoice {
  /** 选择索引 */
  index: number;
  /** 增量消息 */
  delta: MessageDelta;
  /** 结束原因 */
  finishReason?: FinishReason;
}

// ==================== 语音相关 ====================

export interface VoiceSynthesisResponse {
  /** 音频URL（如果返回URL） */
  audioUrl?: string;
  /** 音频时长（毫秒） */
  durationMs: number;
  /** 合成时间（毫秒） */
  synthesisTimeMs: number;
  /** 质量分数 */
  qualityScore: number;
  /** 应用的情感 */
  emotionApplied: string;
}

export interface VoiceRecognitionResponse {
  /** 识别的文本 */
  text: string;
  /** 置信度 */
  confidence: number;
  /** 检测到的语言 */
  languageDetected: string;
  /** 处理时间（毫秒） */
  processingTimeMs: number;
  /** 情感分数列表 */
  emotions?: EmotionScore[];
  /** 说话人片段 */
  speakers?: SpeakerSegment[];
}

export interface EmotionScore {
  /** 情感类型 */
  emotion: string;
  /** 置信度 */
  confidence: number;
  /** 强度 */
  intensity: number;
  /** 模态来源 */
  modality: string;
}

export interface SpeakerSegment {
  /** 说话人ID */
  speakerId: string;
  /** 开始时间（秒） */
  startTime: number;
  /** 结束时间（秒） */
  endTime: number;
  /** 文本内容 */
  text: string;
  /** 置信度 */
  confidence: number;
}

// ==================== 视觉相关 ====================

export interface VisionAnalysisResponse {
  /** 图像类型 */
  imageType: string;
  /** 图像描述 */
  description: string;
  /** 检测到的物体 */
  objects: DetectedObject[];
  /** 提取的文本 */
  texts: ExtractedText[];
  /** 检测到的情感 */
  emotions: EmotionScore[];
  /** 识别的品牌 */
  brands: DetectedBrand[];
  /** 图像质量指标 */
  qualityMetrics: ImageQualityMetrics;
  /** 处理时间（毫秒） */
  processingTimeMs: number;
  /** 准确率估计 */
  accuracyEstimate: number;
}

export interface DetectedObject {
  /** 物体标签 */
  label: string;
  /** 置信度 */
  confidence: number;
  /** 边界框 [x, y, width, height] */
  bbox: [number, number, number, number];
  /** 物体属性 */
  attributes?: Record<string, any>;
}

export interface ExtractedText {
  /** 文本内容 */
  text: string;
  /** 置信度 */
  confidence: number;
  /** 边界框 */
  bbox: [number, number, number, number];
  /** 语言 */
  language: string;
}

export interface DetectedBrand {
  /** 品牌名称 */
  brand: string;
  /** 置信度 */
  confidence: number;
  /** 边界框 */
  bbox: [number, number, number, number];
  /** 品牌分类 */
  category: string;
}

export interface ImageQualityMetrics {
  /** 清晰度 */
  sharpness: number;
  /** 亮度 */
  brightness: number;
  /** 对比度 */
  contrast: number;
  /** 噪声水平 */
  noiseLevel: number;
  /** 整体质量 */
  overallQuality: number;
}

// ==================== 多模态融合 ====================

export interface MultimodalFusionRequest {
  /** 模态输入列表 */
  modalityInputs: ModalityInput[];
  /** 融合策略 */
  fusionStrategy?: FusionStrategy;
  /** 上下文信息 */
  context?: Record<string, any>;
}

export interface ModalityInput {
  /** 模态类型 */
  modality: ModalityType;
  /** 数据内容 */
  data: any;
  /** 置信度 */
  confidence?: number;
  /** 质量分数 */
  qualityScore?: number;
  /** 元数据 */
  metadata?: Record<string, any>;
}

export interface MultimodalFusionResponse {
  /** 融合后的表示向量 */
  fusedRepresentation: number[];
  /** 融合置信度 */
  confidence: number;
  /** 各模态贡献度 */
  modalityContributions: Record<string, number>;
  /** 使用的融合策略 */
  fusionStrategyUsed: FusionStrategy;
  /** 处理时间（毫秒） */
  processingTimeMs: number;
  /** 质量指标 */
  qualityMetrics: Record<string, number>;
  /** 不确定性估计 */
  uncertaintyEstimate: number;
}

// ==================== MCP服务 ====================

export interface ServiceList {
  /** 服务列表 */
  services: ServiceInfo[];
  /** 总数量 */
  totalCount: number;
  /** 分类统计 */
  categories: Record<string, number>;
}

export interface ServiceInfo {
  /** 服务名称 */
  name: string;
  /** 服务分类 */
  category: ServiceCategory;
  /** 版本 */
  version: string;
  /** 描述 */
  description: string;
  /** 提供商 */
  provider: string;
  /** 状态 */
  status: ServiceStatus;
  /** 支持的操作 */
  supportedOperations: string[];
  /** 可靠性分数 */
  reliabilityScore: number;
  /** 响应时间（毫秒） */
  responseTimeMs: number;
}

export interface MCPServiceCallOptions {
  /** 操作名称 */
  operation: string;
  /** 参数 */
  params: Record<string, any>;
}

export interface MCPServiceCallResponse {
  /** 是否成功 */
  success: boolean;
  /** 结果数据 */
  result?: any;
  /** 服务名称 */
  service: string;
  /** 操作名称 */
  operation: string;
  /** 响应时间（毫秒） */
  responseTimeMs: number;
  /** 错误信息 */
  error?: string;
}

// ==================== 数据集管理 ====================

export interface DatasetList {
  /** 数据集列表 */
  datasets: Dataset[];
  /** 总数量 */
  totalCount: number;
}

export interface Dataset {
  /** 数据集ID */
  id: string;
  /** 名称 */
  name: string;
  /** 描述 */
  description: string;
  /** 创建时间 */
  createdAt: string;
  /** 更新时间 */
  updatedAt: string;
  /** 文档数量 */
  documentCount: number;
  /** 大小（字节） */
  sizeBytes: number;
  /** 状态 */
  status: DatasetStatus;
}

export interface CreateDatasetRequest {
  /** 数据集名称 */
  name: string;
  /** 描述 */
  description?: string;
  /** 配置 */
  config?: Record<string, any>;
}

export interface IngestResponse {
  /** 成功摄取的文档数 */
  ingestedDocuments: number;
  /** 失败的文档数 */
  failedDocuments: number;
  /** 处理时间（毫秒） */
  processingTimeMs: number;
  /** 错误列表 */
  errors: string[];
}

// ==================== 对话管理 ====================

export interface ConversationList {
  /** 对话列表 */
  conversations: ConversationSummary[];
  /** 总数量 */
  totalCount: number;
  /** 是否还有更多 */
  hasMore: boolean;
}

export interface ConversationSummary {
  /** 对话ID */
  id: string;
  /** 标题 */
  title: string;
  /** 创建时间 */
  createdAt: string;
  /** 更新时间 */
  updatedAt: string;
  /** 消息数量 */
  messageCount: number;
  /** 状态 */
  status: 'active' | 'archived';
}

export interface Conversation {
  /** 对话ID */
  id: string;
  /** 标题 */
  title: string;
  /** 消息列表 */
  messages: ChatMessage[];
  /** 创建时间 */
  createdAt: string;
  /** 更新时间 */
  updatedAt: string;
  /** 元数据 */
  metadata: Record<string, any>;
}

// ==================== 系统监控 ====================

export interface SystemHealth {
  /** 整体状态 */
  status: HealthStatus;
  /** 各组件状态 */
  components: Record<string, ComponentHealth>;
  /** 检查时间 */
  timestamp: string;
}

export interface ComponentHealth {
  /** 组件状态 */
  status: HealthStatus;
  /** 响应时间（毫秒） */
  responseTimeMs: number;
  /** 错误率 */
  errorRate: number;
  /** 最后检查时间 */
  lastCheck: string;
}

export interface SystemMetrics {
  /** 系统版本 */
  version: string;
  /** 运行时间（秒） */
  uptimeSeconds: number;
  /** 每分钟请求数 */
  requestsPerMinute: number;
  /** 平均响应时间（毫秒） */
  averageResponseTimeMs: number;
  /** 错误率 */
  errorRate: number;
  /** 活跃连接数 */
  activeConnections: number;
  /** 内存使用（MB） */
  memoryUsageMb: number;
  /** CPU使用率（%） */
  cpuUsagePercent: number;
  /** 性能指标 */
  performanceMetrics: PerformanceMetrics;
}

export interface PerformanceMetrics {
  /** 语音处理指标 */
  voiceProcessing: VoiceMetrics;
  /** 视觉处理指标 */
  visionProcessing: VisionMetrics;
  /** 多模态融合指标 */
  multimodalFusion: FusionMetrics;
  /** MCP服务指标 */
  mcpServices: MCPMetrics;
}

export interface VoiceMetrics {
  /** 平均合成时间（毫秒） */
  averageSynthesisTimeMs: number;
  /** 平均识别时间（毫秒） */
  averageRecognitionTimeMs: number;
  /** 成功率 */
  successRate: number;
  /** 目标达成率 */
  targetAchievementRate: number;
}

export interface VisionMetrics {
  /** 平均处理时间（毫秒） */
  averageProcessingTimeMs: number;
  /** 平均准确率 */
  averageAccuracy: number;
  /** 支持的图像类型数 */
  supportedImageTypes: number;
  /** 成功率 */
  successRate: number;
}

export interface FusionMetrics {
  /** 平均融合时间（毫秒） */
  averageFusionTimeMs: number;
  /** 平均准确率 */
  averageAccuracy: number;
  /** 支持的模态数 */
  supportedModalities: number;
  /** 成功率 */
  successRate: number;
}

export interface MCPMetrics {
  /** 总服务数 */
  totalServices: number;
  /** 活跃服务数 */
  activeServices: number;
  /** 平均响应时间（毫秒） */
  averageResponseTimeMs: number;
  /** 成功率 */
  successRate: number;
}

// ==================== 错误类型 ====================

export interface APIError {
  /** 错误码 */
  code: string;
  /** 错误消息 */
  message: string;
  /** 错误类型 */
  type: string;
  /** 错误详情 */
  details?: any;
}

export interface ErrorResponse {
  /** 错误信息 */
  error: APIError;
}

// ==================== 事件类型 ====================

export interface SDKEvent {
  /** 事件类型 */
  type: string;
  /** 事件数据 */
  data: any;
  /** 时间戳 */
  timestamp: number;
}

export type EventHandler<T = any> = (event: SDKEvent & { data: T }) => void;

// ==================== 工具类型 ====================

export type Partial<T> = {
  [P in keyof T]?: T[P];
};

export type Required<T> = {
  [P in keyof T]-?: T[P];
};

export type Pick<T, K extends keyof T> = {
  [P in K]: T[P];
};

export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// ==================== 常量 ====================

export const DEFAULT_CONFIG: Partial<VoiceHelperConfig> = {
  baseURL: 'https://api.voicehelper.com/v1',
  timeout: 30000,
  retries: 3,
  debug: false,
};

export const SUPPORTED_MODELS: ModelType[] = [
  'gpt-4-turbo',
  'gpt-4-vision', 
  'claude-3-opus',
  'doubao-pro'
];

export const SUPPORTED_VOICES: VoiceType[] = [
  'alloy',
  'echo',
  'fable',
  'onyx',
  'nova',
  'shimmer'
];

export const SUPPORTED_EMOTIONS: EmotionType[] = [
  'neutral',
  'happy',
  'sad',
  'angry',
  'excited',
  'calm'
];

export const SUPPORTED_LANGUAGES: LanguageCode[] = [
  'zh-CN',
  'en-US',
  'ja-JP',
  'ko-KR'
];

export const VISION_TASKS: VisionTask[] = [
  'description',
  'object_detection',
  'text_extraction',
  'scene_analysis',
  'emotion_detection',
  'brand_recognition'
];

export const FUSION_STRATEGIES: FusionStrategy[] = [
  'hierarchical',
  'adaptive_attention',
  'cross_transformer',
  'dynamic_weighting'
];

export const SERVICE_CATEGORIES: ServiceCategory[] = [
  'office_suite',
  'development',
  'social_platform',
  'ecommerce',
  'cloud_service',
  'database',
  'monitoring',
  'security',
  'ai_ml'
];