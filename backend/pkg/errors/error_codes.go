package errors

import (
	"fmt"
	"net/http"
)

// ErrorCode 错误码类型
type ErrorCode string

// 系统级错误码
const (
	// 通用错误
	ErrInternalServer     ErrorCode = "INTERNAL_SERVER_ERROR"
	ErrInvalidParams      ErrorCode = "INVALID_PARAMS"
	ErrUnauthorized       ErrorCode = "UNAUTHORIZED"
	ErrForbidden          ErrorCode = "FORBIDDEN"
	ErrNotFound           ErrorCode = "NOT_FOUND"
	ErrMethodNotAllowed   ErrorCode = "METHOD_NOT_ALLOWED"
	ErrTooManyRequests    ErrorCode = "TOO_MANY_REQUESTS"
	ErrServiceUnavailable ErrorCode = "SERVICE_UNAVAILABLE"

	// 认证相关
	ErrTokenInvalid     ErrorCode = "TOKEN_INVALID"
	ErrTokenExpired     ErrorCode = "TOKEN_EXPIRED"
	ErrTokenMissing     ErrorCode = "TOKEN_MISSING"
	ErrLoginFailed      ErrorCode = "LOGIN_FAILED"
	ErrPermissionDenied ErrorCode = "PERMISSION_DENIED"

	// 微信登录相关
	ErrWechatLoginFailed     ErrorCode = "WECHAT_LOGIN_FAILED"
	ErrWechatCodeInvalid     ErrorCode = "WECHAT_CODE_INVALID"
	ErrWechatUserInfoInvalid ErrorCode = "WECHAT_USER_INFO_INVALID"
	ErrSignatureInvalid      ErrorCode = "SIGNATURE_INVALID"

	// 用户相关
	ErrUserNotFound     ErrorCode = "USER_NOT_FOUND"
	ErrUserCreateFailed ErrorCode = "USER_CREATE_FAILED"
	ErrUserUpdateFailed ErrorCode = "USER_UPDATE_FAILED"
	ErrUserInactive     ErrorCode = "USER_INACTIVE"

	// 聊天相关
	ErrChatSessionNotFound ErrorCode = "CHAT_SESSION_NOT_FOUND"
	ErrChatMessageInvalid  ErrorCode = "CHAT_MESSAGE_INVALID"
	ErrChatCancelFailed    ErrorCode = "CHAT_CANCEL_FAILED"

	// 语音相关
	ErrVoiceFileInvalid        ErrorCode = "VOICE_FILE_INVALID"
	ErrVoiceFileTooLarge       ErrorCode = "VOICE_FILE_TOO_LARGE"
	ErrVoiceFormatNotSupported ErrorCode = "VOICE_FORMAT_NOT_SUPPORTED"
	ErrVoiceTranscribeFailed   ErrorCode = "VOICE_TRANSCRIBE_FAILED"
	ErrVoiceSynthesizeFailed   ErrorCode = "VOICE_SYNTHESIZE_FAILED"

	// WebSocket相关
	ErrWebSocketUpgradeFailed    ErrorCode = "WEBSOCKET_UPGRADE_FAILED"
	ErrWebSocketConnectionClosed ErrorCode = "WEBSOCKET_CONNECTION_CLOSED"
	ErrWebSocketMessageInvalid   ErrorCode = "WEBSOCKET_MESSAGE_INVALID"

	// 数据库相关
	ErrDatabaseConnection  ErrorCode = "DATABASE_CONNECTION_ERROR"
	ErrDatabaseQuery       ErrorCode = "DATABASE_QUERY_ERROR"
	ErrDatabaseTransaction ErrorCode = "DATABASE_TRANSACTION_ERROR"

	// 缓存相关
	ErrCacheConnection ErrorCode = "CACHE_CONNECTION_ERROR"
	ErrCacheOperation  ErrorCode = "CACHE_OPERATION_ERROR"

	// 第三方服务相关
	ErrExternalServiceUnavailable ErrorCode = "EXTERNAL_SERVICE_UNAVAILABLE"
	ErrExternalServiceTimeout     ErrorCode = "EXTERNAL_SERVICE_TIMEOUT"
	ErrExternalServiceRateLimit   ErrorCode = "EXTERNAL_SERVICE_RATE_LIMIT"

	// 文件相关
	ErrFileNotFound      ErrorCode = "FILE_NOT_FOUND"
	ErrFileUploadFailed  ErrorCode = "FILE_UPLOAD_FAILED"
	ErrFileFormatInvalid ErrorCode = "FILE_FORMAT_INVALID"
	ErrFileSizeExceeded  ErrorCode = "FILE_SIZE_EXCEEDED"
)

// APIError API错误结构
type APIError struct {
	Code       ErrorCode `json:"code"`
	Message    string    `json:"message"`
	Details    string    `json:"details,omitempty"`
	HTTPStatus int       `json:"-"`
	Cause      error     `json:"-"`
}

// Error 实现error接口
func (e *APIError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("%s: %s (%s)", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap 支持错误链
func (e *APIError) Unwrap() error {
	return e.Cause
}

// NewAPIError 创建API错误
func NewAPIError(code ErrorCode, message string, httpStatus int) *APIError {
	return &APIError{
		Code:       code,
		Message:    message,
		HTTPStatus: httpStatus,
	}
}

// NewAPIErrorWithDetails 创建带详情的API错误
func NewAPIErrorWithDetails(code ErrorCode, message, details string, httpStatus int) *APIError {
	return &APIError{
		Code:       code,
		Message:    message,
		Details:    details,
		HTTPStatus: httpStatus,
	}
}

// NewAPIErrorWithCause 创建带原因的API错误
func NewAPIErrorWithCause(code ErrorCode, message string, httpStatus int, cause error) *APIError {
	return &APIError{
		Code:       code,
		Message:    message,
		HTTPStatus: httpStatus,
		Cause:      cause,
	}
}

// 预定义的常用错误
var (
	ErrInternalServerError = NewAPIError(ErrInternalServer, "内部服务器错误", http.StatusInternalServerError)
	ErrBadRequest          = NewAPIError(ErrInvalidParams, "请求参数错误", http.StatusBadRequest)
	ErrUnauthorizedAccess  = NewAPIError(ErrUnauthorized, "未授权访问", http.StatusUnauthorized)
	ErrForbiddenAccess     = NewAPIError(ErrForbidden, "禁止访问", http.StatusForbidden)
	ErrResourceNotFound    = NewAPIError(ErrNotFound, "资源不存在", http.StatusNotFound)
	ErrRateLimitExceeded   = NewAPIError(ErrTooManyRequests, "请求频率超限", http.StatusTooManyRequests)
)

// GetErrorMessage 获取错误码对应的中文消息
func GetErrorMessage(code ErrorCode) string {
	messages := map[ErrorCode]string{
		ErrInternalServer:     "内部服务器错误",
		ErrInvalidParams:      "请求参数错误",
		ErrUnauthorized:       "未授权访问",
		ErrForbidden:          "禁止访问",
		ErrNotFound:           "资源不存在",
		ErrMethodNotAllowed:   "请求方法不允许",
		ErrTooManyRequests:    "请求频率超限",
		ErrServiceUnavailable: "服务不可用",

		ErrTokenInvalid:     "令牌无效",
		ErrTokenExpired:     "令牌已过期",
		ErrTokenMissing:     "缺少令牌",
		ErrLoginFailed:      "登录失败",
		ErrPermissionDenied: "权限不足",

		ErrWechatLoginFailed:     "微信登录失败",
		ErrWechatCodeInvalid:     "微信授权码无效",
		ErrWechatUserInfoInvalid: "微信用户信息无效",
		ErrSignatureInvalid:      "签名验证失败",

		ErrUserNotFound:     "用户不存在",
		ErrUserCreateFailed: "用户创建失败",
		ErrUserUpdateFailed: "用户更新失败",
		ErrUserInactive:     "用户已禁用",

		ErrChatSessionNotFound: "聊天会话不存在",
		ErrChatMessageInvalid:  "聊天消息无效",
		ErrChatCancelFailed:    "取消聊天失败",

		ErrVoiceFileInvalid:        "语音文件无效",
		ErrVoiceFileTooLarge:       "语音文件过大",
		ErrVoiceFormatNotSupported: "语音格式不支持",
		ErrVoiceTranscribeFailed:   "语音转写失败",
		ErrVoiceSynthesizeFailed:   "语音合成失败",

		ErrWebSocketUpgradeFailed:    "WebSocket升级失败",
		ErrWebSocketConnectionClosed: "WebSocket连接已关闭",
		ErrWebSocketMessageInvalid:   "WebSocket消息无效",

		ErrDatabaseConnection:  "数据库连接错误",
		ErrDatabaseQuery:       "数据库查询错误",
		ErrDatabaseTransaction: "数据库事务错误",

		ErrCacheConnection: "缓存连接错误",
		ErrCacheOperation:  "缓存操作错误",

		ErrExternalServiceUnavailable: "外部服务不可用",
		ErrExternalServiceTimeout:     "外部服务超时",
		ErrExternalServiceRateLimit:   "外部服务限流",

		ErrFileNotFound:      "文件不存在",
		ErrFileUploadFailed:  "文件上传失败",
		ErrFileFormatInvalid: "文件格式无效",
		ErrFileSizeExceeded:  "文件大小超限",
	}

	if msg, ok := messages[code]; ok {
		return msg
	}
	return "未知错误"
}

// GetHTTPStatus 获取错误码对应的HTTP状态码
func GetHTTPStatus(code ErrorCode) int {
	statusMap := map[ErrorCode]int{
		ErrInternalServer:     http.StatusInternalServerError,
		ErrInvalidParams:      http.StatusBadRequest,
		ErrUnauthorized:       http.StatusUnauthorized,
		ErrForbidden:          http.StatusForbidden,
		ErrNotFound:           http.StatusNotFound,
		ErrMethodNotAllowed:   http.StatusMethodNotAllowed,
		ErrTooManyRequests:    http.StatusTooManyRequests,
		ErrServiceUnavailable: http.StatusServiceUnavailable,

		ErrTokenInvalid:     http.StatusUnauthorized,
		ErrTokenExpired:     http.StatusUnauthorized,
		ErrTokenMissing:     http.StatusUnauthorized,
		ErrLoginFailed:      http.StatusUnauthorized,
		ErrPermissionDenied: http.StatusForbidden,

		ErrWechatLoginFailed:     http.StatusUnauthorized,
		ErrWechatCodeInvalid:     http.StatusBadRequest,
		ErrWechatUserInfoInvalid: http.StatusBadRequest,
		ErrSignatureInvalid:      http.StatusUnauthorized,

		ErrUserNotFound:     http.StatusNotFound,
		ErrUserCreateFailed: http.StatusInternalServerError,
		ErrUserUpdateFailed: http.StatusInternalServerError,
		ErrUserInactive:     http.StatusForbidden,

		ErrChatSessionNotFound: http.StatusNotFound,
		ErrChatMessageInvalid:  http.StatusBadRequest,
		ErrChatCancelFailed:    http.StatusInternalServerError,

		ErrVoiceFileInvalid:        http.StatusBadRequest,
		ErrVoiceFileTooLarge:       http.StatusRequestEntityTooLarge,
		ErrVoiceFormatNotSupported: http.StatusUnsupportedMediaType,
		ErrVoiceTranscribeFailed:   http.StatusInternalServerError,
		ErrVoiceSynthesizeFailed:   http.StatusInternalServerError,

		ErrWebSocketUpgradeFailed:    http.StatusBadRequest,
		ErrWebSocketConnectionClosed: http.StatusServiceUnavailable,
		ErrWebSocketMessageInvalid:   http.StatusBadRequest,

		ErrDatabaseConnection:  http.StatusInternalServerError,
		ErrDatabaseQuery:       http.StatusInternalServerError,
		ErrDatabaseTransaction: http.StatusInternalServerError,

		ErrCacheConnection: http.StatusInternalServerError,
		ErrCacheOperation:  http.StatusInternalServerError,

		ErrExternalServiceUnavailable: http.StatusServiceUnavailable,
		ErrExternalServiceTimeout:     http.StatusGatewayTimeout,
		ErrExternalServiceRateLimit:   http.StatusTooManyRequests,

		ErrFileNotFound:      http.StatusNotFound,
		ErrFileUploadFailed:  http.StatusInternalServerError,
		ErrFileFormatInvalid: http.StatusUnsupportedMediaType,
		ErrFileSizeExceeded:  http.StatusRequestEntityTooLarge,
	}

	if status, ok := statusMap[code]; ok {
		return status
	}
	return http.StatusInternalServerError
}
