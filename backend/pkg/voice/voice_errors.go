package voice

import (
	"fmt"
	"time"
)

// VoiceErrorType 语音错误类型
type VoiceErrorType string

const (
	// 连接错误
	ErrorConnectionFailed  VoiceErrorType = "connection_failed"
	ErrorConnectionTimeout VoiceErrorType = "connection_timeout"
	ErrorConnectionLost    VoiceErrorType = "connection_lost"

	// 会话错误
	ErrorSessionNotFound    VoiceErrorType = "session_not_found"
	ErrorSessionExists      VoiceErrorType = "session_exists"
	ErrorSessionExpired     VoiceErrorType = "session_expired"
	ErrorMaxSessionsReached VoiceErrorType = "max_sessions_reached"

	// 音频处理错误
	ErrorAudioFormat     VoiceErrorType = "audio_format_error"
	ErrorAudioProcessing VoiceErrorType = "audio_processing_error"
	ErrorASRFailed       VoiceErrorType = "asr_failed"
	ErrorTTSFailed       VoiceErrorType = "tts_failed"

	// 配置错误
	ErrorInvalidConfig     VoiceErrorType = "invalid_config"
	ErrorUnsupportedFormat VoiceErrorType = "unsupported_format"
	ErrorInvalidSampleRate VoiceErrorType = "invalid_sample_rate"

	// 服务错误
	ErrorServiceUnavailable VoiceErrorType = "service_unavailable"
	ErrorServiceTimeout     VoiceErrorType = "service_timeout"
	ErrorServiceOverloaded  VoiceErrorType = "service_overloaded"

	// 权限错误
	ErrorUnauthorized  VoiceErrorType = "unauthorized"
	ErrorForbidden     VoiceErrorType = "forbidden"
	ErrorQuotaExceeded VoiceErrorType = "quota_exceeded"
)

// VoiceError 语音处理错误
type VoiceError struct {
	Type       VoiceErrorType         `json:"type"`
	Message    string                 `json:"message"`
	SessionID  string                 `json:"session_id,omitempty"`
	UserID     string                 `json:"user_id,omitempty"`
	TenantID   string                 `json:"tenant_id,omitempty"`
	Timestamp  time.Time              `json:"timestamp"`
	Details    map[string]interface{} `json:"details,omitempty"`
	Cause      error                  `json:"-"`
	Retryable  bool                   `json:"retryable"`
	RetryAfter time.Duration          `json:"retry_after,omitempty"`
}

// Error 实现error接口
func (e *VoiceError) Error() string {
	if e.SessionID != "" {
		return fmt.Sprintf("[%s] %s (session: %s)", e.Type, e.Message, e.SessionID)
	}
	return fmt.Sprintf("[%s] %s", e.Type, e.Message)
}

// Unwrap 返回原始错误
func (e *VoiceError) Unwrap() error {
	return e.Cause
}

// IsRetryable 是否可重试
func (e *VoiceError) IsRetryable() bool {
	return e.Retryable
}

// GetRetryAfter 获取重试间隔
func (e *VoiceError) GetRetryAfter() time.Duration {
	if e.RetryAfter > 0 {
		return e.RetryAfter
	}

	// 默认重试间隔
	switch e.Type {
	case ErrorConnectionTimeout, ErrorServiceTimeout:
		return 5 * time.Second
	case ErrorServiceOverloaded:
		return 30 * time.Second
	case ErrorServiceUnavailable:
		return 60 * time.Second
	default:
		return 10 * time.Second
	}
}

// NewVoiceError 创建语音错误
func NewVoiceError(errorType VoiceErrorType, message string) *VoiceError {
	return &VoiceError{
		Type:      errorType,
		Message:   message,
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
	}
}

// NewVoiceErrorWithCause 创建带原因的语音错误
func NewVoiceErrorWithCause(errorType VoiceErrorType, message string, cause error) *VoiceError {
	return &VoiceError{
		Type:      errorType,
		Message:   message,
		Timestamp: time.Now(),
		Details:   make(map[string]interface{}),
		Cause:     cause,
	}
}

// WithSession 添加会话信息
func (e *VoiceError) WithSession(sessionID, userID, tenantID string) *VoiceError {
	e.SessionID = sessionID
	e.UserID = userID
	e.TenantID = tenantID
	return e
}

// WithDetails 添加详细信息
func (e *VoiceError) WithDetails(key string, value interface{}) *VoiceError {
	if e.Details == nil {
		e.Details = make(map[string]interface{})
	}
	e.Details[key] = value
	return e
}

// WithRetry 设置重试信息
func (e *VoiceError) WithRetry(retryable bool, retryAfter time.Duration) *VoiceError {
	e.Retryable = retryable
	e.RetryAfter = retryAfter
	return e
}

// 预定义的错误创建函数

// NewConnectionError 连接错误
func NewConnectionError(message string, cause error) *VoiceError {
	return NewVoiceErrorWithCause(ErrorConnectionFailed, message, cause).
		WithRetry(true, 5*time.Second)
}

// NewSessionNotFoundError 会话未找到错误
func NewSessionNotFoundError(sessionID string) *VoiceError {
	return NewVoiceError(ErrorSessionNotFound, "Voice session not found").
		WithDetails("session_id", sessionID)
}

// NewSessionExistsError 会话已存在错误
func NewSessionExistsError(sessionID string) *VoiceError {
	return NewVoiceError(ErrorSessionExists, "Voice session already exists").
		WithDetails("session_id", sessionID)
}

// NewMaxSessionsReachedError 达到最大会话数错误
func NewMaxSessionsReachedError(maxSessions int) *VoiceError {
	return NewVoiceError(ErrorMaxSessionsReached, "Maximum number of voice sessions reached").
		WithDetails("max_sessions", maxSessions).
		WithRetry(true, 30*time.Second)
}

// NewAudioFormatError 音频格式错误
func NewAudioFormatError(format string, supportedFormats []string) *VoiceError {
	return NewVoiceError(ErrorAudioFormat, "Unsupported audio format").
		WithDetails("format", format).
		WithDetails("supported_formats", supportedFormats)
}

// NewASRError ASR处理错误
func NewASRError(message string, cause error) *VoiceError {
	return NewVoiceErrorWithCause(ErrorASRFailed, message, cause).
		WithRetry(true, 2*time.Second)
}

// NewTTSError TTS处理错误
func NewTTSError(message string, cause error) *VoiceError {
	return NewVoiceErrorWithCause(ErrorTTSFailed, message, cause).
		WithRetry(true, 2*time.Second)
}

// NewServiceUnavailableError 服务不可用错误
func NewServiceUnavailableError(service string, cause error) *VoiceError {
	return NewVoiceErrorWithCause(ErrorServiceUnavailable, fmt.Sprintf("Service %s is unavailable", service), cause).
		WithDetails("service", service).
		WithRetry(true, 60*time.Second)
}

// NewServiceTimeoutError 服务超时错误
func NewServiceTimeoutError(service string, timeout time.Duration) *VoiceError {
	return NewVoiceError(ErrorServiceTimeout, fmt.Sprintf("Service %s timeout after %v", service, timeout)).
		WithDetails("service", service).
		WithDetails("timeout", timeout.String()).
		WithRetry(true, 5*time.Second)
}

// NewUnauthorizedError 未授权错误
func NewUnauthorizedError(userID string) *VoiceError {
	return NewVoiceError(ErrorUnauthorized, "Unauthorized access to voice service").
		WithDetails("user_id", userID)
}

// NewQuotaExceededError 配额超出错误
func NewQuotaExceededError(userID string, quotaType string, limit int) *VoiceError {
	return NewVoiceError(ErrorQuotaExceeded, fmt.Sprintf("Voice service quota exceeded for %s", quotaType)).
		WithDetails("user_id", userID).
		WithDetails("quota_type", quotaType).
		WithDetails("limit", limit).
		WithRetry(true, 3600*time.Second) // 1小时后重试
}

// IsVoiceError 检查是否为语音错误
func IsVoiceError(err error) bool {
	_, ok := err.(*VoiceError)
	return ok
}

// AsVoiceError 转换为语音错误
func AsVoiceError(err error) (*VoiceError, bool) {
	if ve, ok := err.(*VoiceError); ok {
		return ve, true
	}
	return nil, false
}

// GetErrorType 获取错误类型
func GetErrorType(err error) VoiceErrorType {
	if ve, ok := AsVoiceError(err); ok {
		return ve.Type
	}
	return ""
}

// IsRetryableError 检查错误是否可重试
func IsRetryableError(err error) bool {
	if ve, ok := AsVoiceError(err); ok {
		return ve.IsRetryable()
	}
	return false
}

// GetRetryAfterFromError 从错误中获取重试间隔
func GetRetryAfterFromError(err error) time.Duration {
	if ve, ok := AsVoiceError(err); ok {
		return ve.GetRetryAfter()
	}
	return 10 * time.Second
}
