package validation

import (
	"time"
)

// ValidationConfig 验证配置
type ValidationConfig struct {
	// 全局配置
	EnableValidation    bool          `json:"enable_validation" yaml:"enable_validation"`
	StrictMode          bool          `json:"strict_mode" yaml:"strict_mode"`
	LogValidationErrors bool          `json:"log_validation_errors" yaml:"log_validation_errors"`
	ValidationTimeout   time.Duration `json:"validation_timeout" yaml:"validation_timeout"`

	// 文件上传限制
	FileUpload FileUploadConfig `json:"file_upload" yaml:"file_upload"`

	// 请求频率限制
	RateLimit RateLimitConfig `json:"rate_limit" yaml:"rate_limit"`

	// 内容过滤
	ContentFilter ContentFilterConfig `json:"content_filter" yaml:"content_filter"`

	// 业务规则
	BusinessRules BusinessRulesConfig `json:"business_rules" yaml:"business_rules"`
}

// FileUploadConfig 文件上传配置
type FileUploadConfig struct {
	MaxFileSize       int64    `json:"max_file_size" yaml:"max_file_size"`             // 最大文件大小（字节）
	MaxAudioSize      int64    `json:"max_audio_size" yaml:"max_audio_size"`           // 最大音频文件大小
	MaxDocumentSize   int64    `json:"max_document_size" yaml:"max_document_size"`     // 最大文档文件大小
	MaxImageSize      int64    `json:"max_image_size" yaml:"max_image_size"`           // 最大图片文件大小
	AllowedAudioTypes []string `json:"allowed_audio_types" yaml:"allowed_audio_types"` // 允许的音频类型
	AllowedDocTypes   []string `json:"allowed_doc_types" yaml:"allowed_doc_types"`     // 允许的文档类型
	AllowedImageTypes []string `json:"allowed_image_types" yaml:"allowed_image_types"` // 允许的图片类型
	ScanForVirus      bool     `json:"scan_for_virus" yaml:"scan_for_virus"`           // 是否扫描病毒
	CheckFileContent  bool     `json:"check_file_content" yaml:"check_file_content"`   // 是否检查文件内容
}

// RateLimitConfig 频率限制配置
type RateLimitConfig struct {
	EnableRateLimit bool           `json:"enable_rate_limit" yaml:"enable_rate_limit"`
	DefaultWindow   time.Duration  `json:"default_window" yaml:"default_window"`
	DefaultLimit    int            `json:"default_limit" yaml:"default_limit"`
	EndpointLimits  map[string]int `json:"endpoint_limits" yaml:"endpoint_limits"`
	UserTypeLimits  map[string]int `json:"user_type_limits" yaml:"user_type_limits"`
	IPWhitelist     []string       `json:"ip_whitelist" yaml:"ip_whitelist"`
	BurstAllowance  int            `json:"burst_allowance" yaml:"burst_allowance"`
}

// ContentFilterConfig 内容过滤配置
type ContentFilterConfig struct {
	EnableContentFilter bool     `json:"enable_content_filter" yaml:"enable_content_filter"`
	SensitiveWords      []string `json:"sensitive_words" yaml:"sensitive_words"`
	BlockedDomains      []string `json:"blocked_domains" yaml:"blocked_domains"`
	AllowedLanguages    []string `json:"allowed_languages" yaml:"allowed_languages"`
	MaxTextLength       int      `json:"max_text_length" yaml:"max_text_length"`
	MinTextLength       int      `json:"min_text_length" yaml:"min_text_length"`
}

// BusinessRulesConfig 业务规则配置
type BusinessRulesConfig struct {
	// 对话相关
	MaxConversationLength int           `json:"max_conversation_length" yaml:"max_conversation_length"`
	MaxMessageLength      int           `json:"max_message_length" yaml:"max_message_length"`
	ConversationTimeout   time.Duration `json:"conversation_timeout" yaml:"conversation_timeout"`

	// 语音相关
	MaxAudioDuration     int   `json:"max_audio_duration" yaml:"max_audio_duration"` // 秒
	MinAudioDuration     int   `json:"min_audio_duration" yaml:"min_audio_duration"` // 秒
	SupportedSampleRates []int `json:"supported_sample_rates" yaml:"supported_sample_rates"`
	MaxSynthesisLength   int   `json:"max_synthesis_length" yaml:"max_synthesis_length"` // 字符数

	// 搜索相关
	MaxSearchResults int           `json:"max_search_results" yaml:"max_search_results"`
	MinQueryLength   int           `json:"min_query_length" yaml:"min_query_length"`
	MaxQueryLength   int           `json:"max_query_length" yaml:"max_query_length"`
	SearchTimeout    time.Duration `json:"search_timeout" yaml:"search_timeout"`

	// 文档相关
	MaxDocumentsPerBatch int      `json:"max_documents_per_batch" yaml:"max_documents_per_batch"`
	MaxDocumentSize      int64    `json:"max_document_size" yaml:"max_document_size"`
	AllowedDocFormats    []string `json:"allowed_doc_formats" yaml:"allowed_doc_formats"`

	// Agent相关
	MaxAgentSteps      int           `json:"max_agent_steps" yaml:"max_agent_steps"`
	MaxToolsPerRequest int           `json:"max_tools_per_request" yaml:"max_tools_per_request"`
	AgentTimeout       time.Duration `json:"agent_timeout" yaml:"agent_timeout"`
}

// DefaultValidationConfig 默认验证配置
func DefaultValidationConfig() *ValidationConfig {
	return &ValidationConfig{
		EnableValidation:    true,
		StrictMode:          false,
		LogValidationErrors: true,
		ValidationTimeout:   time.Second * 30,

		FileUpload: FileUploadConfig{
			MaxFileSize:     100 * 1024 * 1024, // 100MB
			MaxAudioSize:    50 * 1024 * 1024,  // 50MB
			MaxDocumentSize: 100 * 1024 * 1024, // 100MB
			MaxImageSize:    10 * 1024 * 1024,  // 10MB
			AllowedAudioTypes: []string{
				"audio/wav", "audio/mpeg", "audio/mp3", "audio/webm",
				"audio/x-m4a", "audio/flac", "audio/ogg",
			},
			AllowedDocTypes: []string{
				"application/pdf", "application/msword",
				"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
				"text/plain", "text/markdown", "text/html",
			},
			AllowedImageTypes: []string{
				"image/jpeg", "image/png", "image/gif", "image/webp",
			},
			ScanForVirus:     false,
			CheckFileContent: true,
		},

		RateLimit: RateLimitConfig{
			EnableRateLimit: true,
			DefaultWindow:   time.Hour,
			DefaultLimit:    1000,
			EndpointLimits: map[string]int{
				"/api/v1/voice/transcribe": 200,
				"/api/v1/voice/synthesize": 500,
				"/api/v1/chat/send":        1000,
				"/api/v1/search":           2000,
				"/api/v1/documents/upload": 50,
				"/api/v1/agent/query":      200,
			},
			UserTypeLimits: map[string]int{
				"free":       100,
				"premium":    1000,
				"enterprise": 10000,
			},
			IPWhitelist:    []string{},
			BurstAllowance: 10,
		},

		ContentFilter: ContentFilterConfig{
			EnableContentFilter: true,
			SensitiveWords:      []string{}, // 应该从配置文件或数据库加载
			BlockedDomains:      []string{},
			AllowedLanguages: []string{
				"zh-CN", "zh-TW", "en-US", "en-GB", "ja-JP", "ko-KR",
				"fr-FR", "de-DE", "es-ES", "it-IT", "pt-BR", "ru-RU",
			},
			MaxTextLength: 10000,
			MinTextLength: 1,
		},

		BusinessRules: BusinessRulesConfig{
			MaxConversationLength: 100,
			MaxMessageLength:      10000,
			ConversationTimeout:   time.Hour * 24,

			MaxAudioDuration:     300, // 5分钟
			MinAudioDuration:     1,   // 1秒
			SupportedSampleRates: []int{8000, 16000, 22050, 44100, 48000},
			MaxSynthesisLength:   5000, // 5000字符

			MaxSearchResults: 100,
			MinQueryLength:   1,
			MaxQueryLength:   1000,
			SearchTimeout:    time.Second * 30,

			MaxDocumentsPerBatch: 100,
			MaxDocumentSize:      100 * 1024 * 1024, // 100MB
			AllowedDocFormats:    []string{"pdf", "doc", "docx", "txt", "md", "html"},

			MaxAgentSteps:      20,
			MaxToolsPerRequest: 10,
			AgentTimeout:       time.Minute * 5,
		},
	}
}

// ValidationRule 验证规则
type ValidationRule struct {
	Field     string      `json:"field"`
	Type      string      `json:"type"`      // required, type, length, range, choice, format, custom
	Value     interface{} `json:"value"`     // 规则值
	Message   string      `json:"message"`   // 错误消息
	Condition string      `json:"condition"` // 条件表达式
	Enabled   bool        `json:"enabled"`   // 是否启用
}

// EndpointValidationRules 端点验证规则
type EndpointValidationRules struct {
	Endpoint string           `json:"endpoint"`
	Method   string           `json:"method"`
	Rules    []ValidationRule `json:"rules"`
	Enabled  bool             `json:"enabled"`
}

// ValidationRuleSet 验证规则集
type ValidationRuleSet struct {
	Name        string                    `json:"name"`
	Description string                    `json:"description"`
	Version     string                    `json:"version"`
	Rules       []EndpointValidationRules `json:"rules"`
	CreatedAt   time.Time                 `json:"created_at"`
	UpdatedAt   time.Time                 `json:"updated_at"`
}

// GetDefaultRuleSet 获取默认验证规则集
func GetDefaultRuleSet() *ValidationRuleSet {
	return &ValidationRuleSet{
		Name:        "default_validation_rules",
		Description: "Default validation rules for VoiceHelper API",
		Version:     "1.0.0",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Rules: []EndpointValidationRules{
			{
				Endpoint: "/api/v1/voice/transcribe",
				Method:   "POST",
				Enabled:  true,
				Rules: []ValidationRule{
					{
						Field:   "language",
						Type:    "required",
						Message: "Language is required",
						Enabled: true,
					},
					{
						Field:   "language",
						Type:    "choice",
						Value:   []string{"zh-CN", "en-US", "ja-JP", "ko-KR"},
						Message: "Unsupported language",
						Enabled: true,
					},
					{
						Field:   "audio_format",
						Type:    "required",
						Message: "Audio format is required",
						Enabled: true,
					},
					{
						Field:   "audio_format",
						Type:    "choice",
						Value:   []string{"wav", "mp3", "webm", "m4a"},
						Message: "Unsupported audio format",
						Enabled: true,
					},
					{
						Field:   "sample_rate",
						Type:    "range",
						Value:   map[string]int{"min": 8000, "max": 48000},
						Message: "Sample rate must be between 8000 and 48000",
						Enabled: true,
					},
					{
						Field:   "max_duration",
						Type:    "range",
						Value:   map[string]int{"min": 1, "max": 300},
						Message: "Max duration must be between 1 and 300 seconds",
						Enabled: true,
					},
				},
			},
			{
				Endpoint: "/api/v1/voice/synthesize",
				Method:   "POST",
				Enabled:  true,
				Rules: []ValidationRule{
					{
						Field:   "text",
						Type:    "required",
						Message: "Text is required",
						Enabled: true,
					},
					{
						Field:   "text",
						Type:    "length",
						Value:   map[string]int{"min": 1, "max": 5000},
						Message: "Text length must be between 1 and 5000 characters",
						Enabled: true,
					},
					{
						Field:   "voice_id",
						Type:    "required",
						Message: "Voice ID is required",
						Enabled: true,
					},
					{
						Field:   "speed",
						Type:    "range",
						Value:   map[string]float64{"min": 0.5, "max": 2.0},
						Message: "Speed must be between 0.5 and 2.0",
						Enabled: true,
					},
					{
						Field:   "pitch",
						Type:    "range",
						Value:   map[string]float64{"min": -20.0, "max": 20.0},
						Message: "Pitch must be between -20 and 20",
						Enabled: true,
					},
				},
			},
			{
				Endpoint: "/api/v1/search",
				Method:   "POST",
				Enabled:  true,
				Rules: []ValidationRule{
					{
						Field:   "query",
						Type:    "required",
						Message: "Query is required",
						Enabled: true,
					},
					{
						Field:   "query",
						Type:    "length",
						Value:   map[string]int{"min": 1, "max": 1000},
						Message: "Query length must be between 1 and 1000 characters",
						Enabled: true,
					},
					{
						Field:   "limit",
						Type:    "range",
						Value:   map[string]int{"min": 1, "max": 100},
						Message: "Limit must be between 1 and 100",
						Enabled: true,
					},
					{
						Field:   "search_type",
						Type:    "choice",
						Value:   []string{"semantic", "keyword", "hybrid", "fuzzy"},
						Message: "Invalid search type",
						Enabled: true,
					},
				},
			},
			{
				Endpoint: "/api/v1/documents/upload",
				Method:   "POST",
				Enabled:  true,
				Rules: []ValidationRule{
					{
						Field:   "title",
						Type:    "required",
						Message: "Document title is required",
						Enabled: true,
					},
					{
						Field:   "title",
						Type:    "length",
						Value:   map[string]int{"min": 1, "max": 200},
						Message: "Title length must be between 1 and 200 characters",
						Enabled: true,
					},
					{
						Field:   "file",
						Type:    "required",
						Message: "Document file is required",
						Enabled: true,
					},
				},
			},
			{
				Endpoint: "/api/v1/chat/send",
				Method:   "POST",
				Enabled:  true,
				Rules: []ValidationRule{
					{
						Field:   "message",
						Type:    "required",
						Message: "Message is required",
						Enabled: true,
					},
					{
						Field:   "message",
						Type:    "length",
						Value:   map[string]int{"min": 1, "max": 10000},
						Message: "Message length must be between 1 and 10000 characters",
						Enabled: true,
					},
					{
						Field:   "temperature",
						Type:    "range",
						Value:   map[string]float64{"min": 0.0, "max": 2.0},
						Message: "Temperature must be between 0.0 and 2.0",
						Enabled: true,
					},
					{
						Field:   "max_tokens",
						Type:    "range",
						Value:   map[string]int{"min": 1, "max": 8192},
						Message: "Max tokens must be between 1 and 8192",
						Enabled: true,
					},
				},
			},
		},
	}
}
