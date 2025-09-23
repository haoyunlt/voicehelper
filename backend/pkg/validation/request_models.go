package validation

import "mime/multipart"

// 语音相关请求模型

// TranscribeRequest 语音转文字请求
type TranscribeRequest struct {
	Language      string                `form:"language" json:"language" binding:"required" validate:"required,language_code"`
	AudioFormat   string                `form:"audio_format" json:"audio_format" binding:"required" validate:"required,audio_format"`
	SampleRate    int                   `form:"sample_rate" json:"sample_rate" validate:"min=8000,max=48000"`
	EnableEmotion bool                  `form:"enable_emotion" json:"enable_emotion"`
	EnableSpeaker bool                  `form:"enable_speaker" json:"enable_speaker"`
	MaxDuration   int                   `form:"max_duration" json:"max_duration" validate:"min=1,max=300"` // 最大5分钟
	AudioFile     *multipart.FileHeader `form:"audio_file" binding:"required"`
}

// SynthesizeRequest 文字转语音请求
type SynthesizeRequest struct {
	Text         string  `json:"text" binding:"required" validate:"required,min=1,max=5000"`
	VoiceID      string  `json:"voice_id" binding:"required" validate:"required,voice_id"`
	Language     string  `json:"language" binding:"required" validate:"required,language_code"`
	Speed        float64 `json:"speed" validate:"min=0.5,max=2.0"`
	Pitch        float64 `json:"pitch" validate:"min=-20,max=20"`
	Volume       float64 `json:"volume" validate:"min=0.1,max=2.0"`
	Emotion      string  `json:"emotion" validate:"emotion"`
	OutputFormat string  `json:"output_format" validate:"oneof=wav mp3 webm"`
}

// VoiceStreamRequest WebSocket语音流请求
type VoiceStreamRequest struct {
	SessionID   string `json:"session_id" binding:"required" validate:"required,min=1"`
	Language    string `json:"language" binding:"required" validate:"required,language_code"`
	AudioFormat string `json:"audio_format" binding:"required" validate:"required,audio_format"`
	SampleRate  int    `json:"sample_rate" validate:"min=8000,max=48000"`
	ChunkSize   int    `json:"chunk_size" validate:"min=512,max=8192"`
	EnableVAD   bool   `json:"enable_vad"` // 语音活动检测
	EnableNR    bool   `json:"enable_nr"`  // 噪音抑制
}

// 对话相关请求模型

// ChatRequest 聊天请求
type ChatRequest struct {
	ConversationID string                 `json:"conversation_id" validate:"min=1"`
	Message        string                 `json:"message" binding:"required" validate:"required,min=1,max=10000"`
	MessageType    string                 `json:"message_type" validate:"oneof=text image audio video file"`
	Context        map[string]interface{} `json:"context"`
	Stream         bool                   `json:"stream"`
	Temperature    float64                `json:"temperature" validate:"min=0,max=2"`
	MaxTokens      int                    `json:"max_tokens" validate:"min=1,max=8192"`
	Model          string                 `json:"model" validate:"min=1"`
}

// CancelChatRequest 取消聊天请求
type CancelChatRequest struct {
	ConversationID string `json:"conversation_id" binding:"required" validate:"required,min=1"`
	Reason         string `json:"reason" validate:"max=500"`
}

// 搜索相关请求模型

// SearchRequest 搜索请求
type SearchRequest struct {
	Query       string                 `json:"query" binding:"required" validate:"required,min=1,max=1000"`
	SearchType  string                 `json:"search_type" validate:"search_type"`
	Limit       int                    `json:"limit" validate:"min=1,max=100"`
	Offset      int                    `json:"offset" validate:"min=0"`
	Filters     map[string]interface{} `json:"filters"`
	IncludeText bool                   `json:"include_text"`
	Language    string                 `json:"language" validate:"language_code"`
}

// SearchSuggestionsRequest 搜索建议请求
type SearchSuggestionsRequest struct {
	Query    string `form:"query" binding:"required" validate:"required,min=1,max=100"`
	Limit    int    `form:"limit" validate:"min=1,max=20"`
	Language string `form:"language" validate:"language_code"`
}

// 文档管理请求模型

// UploadDocumentRequest 上传文档请求
type UploadDocumentRequest struct {
	Title       string                `form:"title" binding:"required" validate:"required,min=1,max=200"`
	Description string                `form:"description" validate:"max=1000"`
	Category    string                `form:"category" validate:"min=1,max=50"`
	Tags        string                `form:"tags" validate:"max=500"` // 逗号分隔的标签
	Language    string                `form:"language" validate:"language_code"`
	IsPublic    bool                  `form:"is_public"`
	File        *multipart.FileHeader `form:"file" binding:"required"`
}

// UpdateDocumentRequest 更新文档请求
type UpdateDocumentRequest struct {
	ID          string `uri:"id" binding:"required" validate:"required,min=1"`
	Title       string `json:"title" validate:"min=1,max=200"`
	Description string `json:"description" validate:"max=1000"`
	Category    string `json:"category" validate:"min=1,max=50"`
	Tags        string `json:"tags" validate:"max=500"`
	IsPublic    bool   `json:"is_public"`
}

// DeleteDocumentRequest 删除文档请求
type DeleteDocumentRequest struct {
	ID string `uri:"id" binding:"required" validate:"required,min=1"`
}

// GetDocumentRequest 获取文档请求
type GetDocumentRequest struct {
	ID          string `uri:"id" binding:"required" validate:"required,min=1"`
	IncludeText bool   `form:"include_text"`
}

// ListDocumentsRequest 列出文档请求
type ListDocumentsRequest struct {
	Page     int    `form:"page" validate:"min=1"`
	PageSize int    `form:"page_size" validate:"min=1,max=100"`
	Category string `form:"category" validate:"max=50"`
	Language string `form:"language" validate:"language_code"`
	Search   string `form:"search" validate:"max=200"`
}

// 会话管理请求模型

// CreateConversationRequest 创建会话请求
type CreateConversationRequest struct {
	Title       string                 `json:"title" validate:"min=1,max=200"`
	Description string                 `json:"description" validate:"max=1000"`
	Type        string                 `json:"type" validate:"oneof=chat voice multimodal"`
	Config      map[string]interface{} `json:"config"`
}

// UpdateConversationRequest 更新会话请求
type UpdateConversationRequest struct {
	ID          string                 `uri:"id" binding:"required" validate:"required,min=1"`
	Title       string                 `json:"title" validate:"min=1,max=200"`
	Description string                 `json:"description" validate:"max=1000"`
	Config      map[string]interface{} `json:"config"`
}

// GetConversationRequest 获取会话请求
type GetConversationRequest struct {
	ID string `uri:"id" binding:"required" validate:"required,min=1"`
}

// DeleteConversationRequest 删除会话请求
type DeleteConversationRequest struct {
	ID string `uri:"id" binding:"required" validate:"required,min=1"`
}

// ListConversationsRequest 列出会话请求
type ListConversationsRequest struct {
	Page     int    `form:"page" validate:"min=1"`
	PageSize int    `form:"page_size" validate:"min=1,max=100"`
	Type     string `form:"type" validate:"oneof=chat voice multimodal"`
	Search   string `form:"search" validate:"max=200"`
}

// GetConversationMessagesRequest 获取会话消息请求
type GetConversationMessagesRequest struct {
	ID       string `uri:"id" binding:"required" validate:"required,min=1"`
	Page     int    `form:"page" validate:"min=1"`
	PageSize int    `form:"page_size" validate:"min=1,max=100"`
	Since    int64  `form:"since" validate:"min=0"`
}

// Agent相关请求模型

// AgentStreamRequest Agent流式请求
type AgentStreamRequest struct {
	Query       string                 `json:"query" binding:"required" validate:"required,min=1,max=5000"`
	SessionID   string                 `json:"session_id" validate:"min=1"`
	Tools       []string               `json:"tools"`
	Context     map[string]interface{} `json:"context"`
	Temperature float64                `json:"temperature" validate:"min=0,max=2"`
	MaxSteps    int                    `json:"max_steps" validate:"min=1,max=20"`
}

// ExecuteAgentToolRequest 执行Agent工具请求
type ExecuteAgentToolRequest struct {
	ToolName   string                 `json:"tool_name" binding:"required" validate:"required,min=1"`
	Parameters map[string]interface{} `json:"parameters" binding:"required"`
	SessionID  string                 `json:"session_id" validate:"min=1"`
	Timeout    int                    `json:"timeout" validate:"min=1,max=300"` // 超时时间（秒）
}

// 认证相关请求模型

// WechatMiniProgramLoginRequest 微信小程序登录请求
type WechatMiniProgramLoginRequest struct {
	Code          string `json:"code" binding:"required" validate:"required,min=1"`
	EncryptedData string `json:"encrypted_data"`
	IV            string `json:"iv"`
	Signature     string `json:"signature"`
	RawData       string `json:"raw_data"`
}

// RefreshTokenRequest 刷新令牌请求
type RefreshTokenRequest struct {
	RefreshToken string `json:"refresh_token" binding:"required" validate:"required,min=1"`
}

// 管理相关请求模型

// SetMaintenanceModeRequest 设置维护模式请求
type SetMaintenanceModeRequest struct {
	Enabled bool   `json:"enabled"`
	Message string `json:"message" validate:"max=500"`
	EndTime int64  `json:"end_time" validate:"min=0"`
}

// 多模态相关请求模型

// MultimodalRequest 多模态请求
type MultimodalRequest struct {
	Text       string                `json:"text" validate:"max=5000"`
	ImageFile  *multipart.FileHeader `form:"image_file"`
	AudioFile  *multipart.FileHeader `form:"audio_file"`
	VideoFile  *multipart.FileHeader `form:"video_file"`
	Query      string                `json:"query" binding:"required" validate:"required,min=1,max=1000"`
	FusionType string                `json:"fusion_type" validate:"oneof=early late attention gated"`
	Language   string                `json:"language" validate:"language_code"`
	OutputType string                `json:"output_type" validate:"oneof=text audio image video"`
}

// 性能监控请求模型

// PerformanceMetricsRequest 性能指标请求
type PerformanceMetricsRequest struct {
	StartTime int64    `form:"start_time" validate:"min=0"`
	EndTime   int64    `form:"end_time" validate:"min=0"`
	Metrics   []string `form:"metrics"`
	Interval  string   `form:"interval" validate:"oneof=1m 5m 15m 1h 1d"`
}

// 批量操作请求模型

// BatchRequest 批量请求
type BatchRequest struct {
	Operations []BatchOperation `json:"operations" binding:"required" validate:"required,min=1,max=100"`
	Parallel   bool             `json:"parallel"`
	Timeout    int              `json:"timeout" validate:"min=1,max=600"`
}

// BatchOperation 批量操作
type BatchOperation struct {
	ID         string                 `json:"id" validate:"required,min=1"`
	Type       string                 `json:"type" binding:"required" validate:"required,oneof=transcribe synthesize search upload"`
	Parameters map[string]interface{} `json:"parameters" binding:"required"`
}

// 健康检查请求模型（通常不需要参数，但为了完整性）

// HealthCheckRequest 健康检查请求
type HealthCheckRequest struct {
	Deep bool `form:"deep"` // 是否进行深度健康检查
}

// 配置相关请求模型

// ReloadConfigurationRequest 重新加载配置请求
type ReloadConfigurationRequest struct {
	Component string `json:"component" validate:"oneof=all auth voice search rag agent"`
	Force     bool   `json:"force"`
}
