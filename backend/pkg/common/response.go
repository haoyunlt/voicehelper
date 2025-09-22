package common

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// APIResponse 统一API响应结构
type APIResponse struct {
	Success   bool        `json:"success"`
	Code      int         `json:"code"`
	Message   string      `json:"message"`
	Data      interface{} `json:"data,omitempty"`
	Error     *APIError   `json:"error,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
	RequestID string      `json:"request_id,omitempty"`
}

// APIError API错误结构
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

// PaginationResponse 分页响应结构
type PaginationResponse struct {
	Items      interface{} `json:"items"`
	Total      int         `json:"total"`
	Page       int         `json:"page"`
	PageSize   int         `json:"page_size"`
	TotalPages int         `json:"total_pages"`
}

// ResponseBuilder 响应构建器
type ResponseBuilder struct {
	c *gin.Context
}

// NewResponseBuilder 创建响应构建器
func NewResponseBuilder(c *gin.Context) *ResponseBuilder {
	return &ResponseBuilder{c: c}
}

// Success 成功响应
func (rb *ResponseBuilder) Success(data interface{}) {
	rb.JSON(http.StatusOK, true, "Success", data, nil)
}

// SuccessWithMessage 带消息的成功响应
func (rb *ResponseBuilder) SuccessWithMessage(message string, data interface{}) {
	rb.JSON(http.StatusOK, true, message, data, nil)
}

// Error 错误响应
func (rb *ResponseBuilder) Error(code int, message string) {
	rb.ErrorWithDetails(code, message, "")
}

// ErrorWithDetails 带详情的错误响应
func (rb *ResponseBuilder) ErrorWithDetails(code int, message, details string) {
	apiError := &APIError{
		Code:    http.StatusText(code),
		Message: message,
		Details: details,
	}
	rb.JSON(code, false, message, nil, apiError)
}

// BadRequest 400错误
func (rb *ResponseBuilder) BadRequest(message string) {
	rb.Error(http.StatusBadRequest, message)
}

// Unauthorized 401错误
func (rb *ResponseBuilder) Unauthorized(message string) {
	if message == "" {
		message = "Unauthorized"
	}
	rb.Error(http.StatusUnauthorized, message)
}

// Forbidden 403错误
func (rb *ResponseBuilder) Forbidden(message string) {
	if message == "" {
		message = "Forbidden"
	}
	rb.Error(http.StatusForbidden, message)
}

// NotFound 404错误
func (rb *ResponseBuilder) NotFound(message string) {
	if message == "" {
		message = "Resource not found"
	}
	rb.Error(http.StatusNotFound, message)
}

// InternalServerError 500错误
func (rb *ResponseBuilder) InternalServerError(message string) {
	if message == "" {
		message = "Internal server error"
	}
	rb.Error(http.StatusInternalServerError, message)
}

// Pagination 分页响应
func (rb *ResponseBuilder) Pagination(items interface{}, total, page, pageSize int) {
	totalPages := (total + pageSize - 1) / pageSize
	data := PaginationResponse{
		Items:      items,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}
	rb.Success(data)
}

// JSON 通用JSON响应
func (rb *ResponseBuilder) JSON(statusCode int, success bool, message string, data interface{}, apiError *APIError) {
	response := APIResponse{
		Success:   success,
		Code:      statusCode,
		Message:   message,
		Data:      data,
		Error:     apiError,
		Timestamp: time.Now(),
		RequestID: rb.getRequestID(),
	}

	rb.c.JSON(statusCode, response)
}

// getRequestID 获取请求ID
func (rb *ResponseBuilder) getRequestID() string {
	if requestID := rb.c.GetHeader("X-Request-ID"); requestID != "" {
		return requestID
	}
	if requestID := rb.c.GetHeader("X-Trace-ID"); requestID != "" {
		return requestID
	}
	return ""
}

// 全局便利函数
func Success(c *gin.Context, data interface{}) {
	NewResponseBuilder(c).Success(data)
}

func SuccessWithMessage(c *gin.Context, message string, data interface{}) {
	NewResponseBuilder(c).SuccessWithMessage(message, data)
}

func Error(c *gin.Context, code int, message string) {
	NewResponseBuilder(c).Error(code, message)
}

func BadRequest(c *gin.Context, message string) {
	NewResponseBuilder(c).BadRequest(message)
}

func Unauthorized(c *gin.Context, message string) {
	NewResponseBuilder(c).Unauthorized(message)
}

func Forbidden(c *gin.Context, message string) {
	NewResponseBuilder(c).Forbidden(message)
}

func NotFound(c *gin.Context, message string) {
	NewResponseBuilder(c).NotFound(message)
}

func InternalServerError(c *gin.Context, message string) {
	NewResponseBuilder(c).InternalServerError(message)
}

func Pagination(c *gin.Context, items interface{}, total, page, pageSize int) {
	NewResponseBuilder(c).Pagination(items, total, page, pageSize)
}
