package middleware

import (
	"context"
	"fmt"
	"net/http"
	"runtime/debug"
	"time"

	"voicehelper/backend/pkg/errors"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// ErrorResponse 统一错误响应格式
type ErrorResponse struct {
	Error     string    `json:"error"`
	Code      string    `json:"code"`
	Details   string    `json:"details,omitempty"`
	Timestamp time.Time `json:"timestamp"`
	RequestID string    `json:"request_id,omitempty"`
	Path      string    `json:"path"`
}

// ErrorHandler 全局错误处理中间件
func ErrorHandler() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		requestID := c.GetString("request_id")

		// 记录panic信息
		logrus.WithFields(logrus.Fields{
			"request_id": requestID,
			"method":     c.Request.Method,
			"path":       c.Request.URL.Path,
			"panic":      recovered,
			"stack":      string(debug.Stack()),
		}).Error("服务器发生panic")

		// 返回统一的错误响应
		errorResp := ErrorResponse{
			Error:     "内部服务器错误",
			Code:      string(errors.ErrInternalServer),
			Timestamp: time.Now(),
			RequestID: requestID,
			Path:      c.Request.URL.Path,
		}

		c.JSON(http.StatusInternalServerError, errorResp)
		c.Abort()
	})
}

// HandleAPIError 处理API错误
func HandleAPIError(c *gin.Context, err error) {
	requestID := c.GetString("request_id")

	var apiErr *errors.APIError
	var httpStatus int
	var errorCode string
	var message string
	var details string

	// 检查是否是APIError类型
	if e, ok := err.(*errors.APIError); ok {
		apiErr = e
		httpStatus = e.HTTPStatus
		errorCode = string(e.Code)
		message = e.Message
		details = e.Details
	} else {
		// 未知错误，包装为内部服务器错误
		httpStatus = http.StatusInternalServerError
		errorCode = string(errors.ErrInternalServer)
		message = "内部服务器错误"
		details = ""
	}

	// 记录错误日志
	logFields := logrus.Fields{
		"request_id":  requestID,
		"method":      c.Request.Method,
		"path":        c.Request.URL.Path,
		"error_code":  errorCode,
		"http_status": httpStatus,
		"user_agent":  c.Request.UserAgent(),
		"client_ip":   c.ClientIP(),
	}

	if apiErr != nil && apiErr.Cause != nil {
		logFields["cause"] = apiErr.Cause.Error()
	}

	// 根据错误级别记录不同级别的日志
	if httpStatus >= 500 {
		logrus.WithFields(logFields).WithError(err).Error("服务器内部错误")
	} else if httpStatus >= 400 {
		logrus.WithFields(logFields).WithError(err).Warn("客户端请求错误")
	} else {
		logrus.WithFields(logFields).WithError(err).Info("请求处理完成")
	}

	// 构造错误响应
	errorResp := ErrorResponse{
		Error:     message,
		Code:      errorCode,
		Details:   details,
		Timestamp: time.Now(),
		RequestID: requestID,
		Path:      c.Request.URL.Path,
	}

	c.JSON(httpStatus, errorResp)
}

// ValidationErrorHandler 参数验证错误处理
func ValidationErrorHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()

		// 检查是否有绑定错误
		if len(c.Errors) > 0 {
			err := c.Errors.Last()

			// 如果是绑定错误，转换为统一格式
			if err.Type == gin.ErrorTypeBind {
				apiErr := errors.NewAPIErrorWithDetails(
					errors.ErrInvalidParams,
					"请求参数验证失败",
					err.Error(),
					http.StatusBadRequest,
				)
				HandleAPIError(c, apiErr)
				c.Abort()
				return
			}
		}
	}
}

// TimeoutHandler 请求超时处理
func TimeoutHandler(timeout time.Duration) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 设置请求超时
		ctx := c.Request.Context()

		// 创建带超时的context
		timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()

		// 替换请求的context
		c.Request = c.Request.WithContext(timeoutCtx)

		// 使用channel来检测超时
		done := make(chan struct{})

		go func() {
			c.Next()
			close(done)
		}()

		select {
		case <-done:
			// 请求正常完成
			return
		case <-timeoutCtx.Done():
			// 请求超时
			if timeoutCtx.Err() == context.DeadlineExceeded {
				apiErr := errors.NewAPIError(
					errors.ErrExternalServiceTimeout,
					"请求处理超时",
					http.StatusGatewayTimeout,
				)
				HandleAPIError(c, apiErr)
				c.Abort()
				return
			}
		}
	}
}

// NotFoundHandler 404错误处理
func NotFoundHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		apiErr := errors.NewAPIErrorWithDetails(
			errors.ErrNotFound,
			"请求的资源不存在",
			fmt.Sprintf("路径 %s 不存在", c.Request.URL.Path),
			http.StatusNotFound,
		)
		HandleAPIError(c, apiErr)
	}
}

// MethodNotAllowedHandler 405错误处理
func MethodNotAllowedHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		apiErr := errors.NewAPIErrorWithDetails(
			errors.ErrMethodNotAllowed,
			"请求方法不允许",
			fmt.Sprintf("方法 %s 不被路径 %s 支持", c.Request.Method, c.Request.URL.Path),
			http.StatusMethodNotAllowed,
		)
		HandleAPIError(c, apiErr)
	}
}
