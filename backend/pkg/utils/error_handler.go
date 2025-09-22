package utils

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// ErrorHandler 错误处理器
type ErrorHandler struct {
	logger *logrus.Logger
}

// NewErrorHandler 创建错误处理器
func NewErrorHandler(logger *logrus.Logger) *ErrorHandler {
	if logger == nil {
		logger = logrus.New()
	}
	return &ErrorHandler{logger: logger}
}

// HandleError 统一错误处理
func (eh *ErrorHandler) HandleError(err error, context string, fields ...logrus.Fields) error {
	if err == nil {
		return nil
	}

	// 获取调用者信息
	pc, file, line, ok := runtime.Caller(1)
	var caller string
	if ok {
		fn := runtime.FuncForPC(pc)
		caller = fmt.Sprintf("%s:%d %s", file, line, fn.Name())
	}

	// 合并字段
	logFields := logrus.Fields{
		"context": context,
		"caller":  caller,
	}
	for _, field := range fields {
		for k, v := range field {
			logFields[k] = v
		}
	}

	// 根据错误类型进行不同处理
	switch {
	case errors.Is(err, sql.ErrNoRows):
		eh.logger.WithFields(logFields).Warn("No rows found")
		return fmt.Errorf("resource not found")
	case errors.Is(err, context.Canceled):
		eh.logger.WithFields(logFields).Info("Context canceled")
		return fmt.Errorf("operation canceled")
	case errors.Is(err, context.DeadlineExceeded):
		eh.logger.WithFields(logFields).Warn("Context deadline exceeded")
		return fmt.Errorf("operation timeout")
	case strings.Contains(err.Error(), "connection refused"):
		eh.logger.WithFields(logFields).Error("Connection refused")
		return fmt.Errorf("service unavailable")
	case strings.Contains(err.Error(), "timeout"):
		eh.logger.WithFields(logFields).Warn("Operation timeout")
		return fmt.Errorf("operation timeout")
	default:
		eh.logger.WithFields(logFields).Error(err.Error())
		return err
	}
}

// SafeString 安全字符串转换，处理nil指针
func SafeString(ptr *string) string {
	if ptr == nil {
		return ""
	}
	return *ptr
}

// SafeInt 安全整数转换，处理nil指针
func SafeInt(ptr *int) int {
	if ptr == nil {
		return 0
	}
	return *ptr
}

// SafeInt64 安全int64转换，处理nil指针
func SafeInt64(ptr *int64) int64 {
	if ptr == nil {
		return 0
	}
	return *ptr
}

// SafeBool 安全布尔转换，处理nil指针
func SafeBool(ptr *bool) bool {
	if ptr == nil {
		return false
	}
	return *ptr
}

// SafeTime 安全时间转换，处理nil指针
func SafeTime(ptr *time.Time) time.Time {
	if ptr == nil {
		return time.Time{}
	}
	return *ptr
}

// SafeFloat64 安全float64转换，处理nil指针
func SafeFloat64(ptr *float64) float64 {
	if ptr == nil {
		return 0.0
	}
	return *ptr
}

// StringPtr 创建字符串指针
func StringPtr(s string) *string {
	return &s
}

// IntPtr 创建整数指针
func IntPtr(i int) *int {
	return &i
}

// Int64Ptr 创建int64指针
func Int64Ptr(i int64) *int64 {
	return &i
}

// BoolPtr 创建布尔指针
func BoolPtr(b bool) *bool {
	return &b
}

// TimePtr 创建时间指针
func TimePtr(t time.Time) *time.Time {
	return &t
}

// Float64Ptr 创建float64指针
func Float64Ptr(f float64) *float64 {
	return &f
}

// IsNil 检查接口是否为nil
func IsNil(i interface{}) bool {
	if i == nil {
		return true
	}

	v := reflect.ValueOf(i)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

// IsEmpty 检查值是否为空
func IsEmpty(i interface{}) bool {
	if IsNil(i) {
		return true
	}

	v := reflect.ValueOf(i)
	switch v.Kind() {
	case reflect.String:
		return v.String() == ""
	case reflect.Slice, reflect.Map, reflect.Array:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Struct:
		// 对于时间类型的特殊处理
		if t, ok := i.(time.Time); ok {
			return t.IsZero()
		}
		return false
	default:
		return false
	}
}

// DefaultString 返回默认字符串值
func DefaultString(value, defaultValue string) string {
	if value == "" {
		return defaultValue
	}
	return value
}

// DefaultInt 返回默认整数值
func DefaultInt(value, defaultValue int) int {
	if value == 0 {
		return defaultValue
	}
	return value
}

// DefaultInt64 返回默认int64值
func DefaultInt64(value, defaultValue int64) int64 {
	if value == 0 {
		return defaultValue
	}
	return value
}

// DefaultFloat64 返回默认float64值
func DefaultFloat64(value, defaultValue float64) float64 {
	if value == 0.0 {
		return defaultValue
	}
	return value
}

// CoalesceString 返回第一个非空字符串
func CoalesceString(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// CoalesceInt 返回第一个非零整数
func CoalesceInt(values ...int) int {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}

// TryCatch 模拟try-catch语法
func TryCatch(tryFunc func() error, catchFunc func(error) error) error {
	defer func() {
		if r := recover(); r != nil {
			var err error
			switch x := r.(type) {
			case string:
				err = errors.New(x)
			case error:
				err = x
			default:
				err = fmt.Errorf("unknown panic: %v", r)
			}
			if catchFunc != nil {
				catchFunc(err)
			}
		}
	}()

	return tryFunc()
}

// RetryWithBackoff 带退避的重试机制
func RetryWithBackoff(ctx context.Context, maxRetries int, initialDelay time.Duration, fn func() error) error {
	var lastErr error
	delay := initialDelay

	for i := 0; i < maxRetries; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := fn(); err == nil {
			return nil
		} else {
			lastErr = err
		}

		if i < maxRetries-1 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
			delay *= 2 // 指数退避
		}
	}

	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr)
}

// ValidateRequired 验证必需字段
func ValidateRequired(fields map[string]interface{}) error {
	var missing []string

	for name, value := range fields {
		if IsEmpty(value) {
			missing = append(missing, name)
		}
	}

	if len(missing) > 0 {
		return fmt.Errorf("required fields missing: %s", strings.Join(missing, ", "))
	}

	return nil
}

// SafeExecute 安全执行函数，捕获panic
func SafeExecute(fn func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			switch x := r.(type) {
			case string:
				err = errors.New(x)
			case error:
				err = x
			default:
				err = fmt.Errorf("panic: %v", r)
			}
		}
	}()

	return fn()
}

// WrapError 包装错误，添加上下文信息
func WrapError(err error, message string) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", message, err)
}

// ChainErrors 链式错误处理
func ChainErrors(errors ...error) error {
	var messages []string
	for _, err := range errors {
		if err != nil {
			messages = append(messages, err.Error())
		}
	}

	if len(messages) == 0 {
		return nil
	}

	return fmt.Errorf("multiple errors: %s", strings.Join(messages, "; "))
}

// TimeoutContext 创建带超时的上下文
func TimeoutContext(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		timeout = 30 * time.Second // 默认30秒超时
	}
	return context.WithTimeout(parent, timeout)
}

// Must 必须成功，否则panic
func Must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}

// IgnoreError 忽略错误，只返回值
func IgnoreError[T any](value T, err error) T {
	return value
}

// FirstError 返回第一个非nil错误
func FirstError(errors ...error) error {
	for _, err := range errors {
		if err != nil {
			return err
		}
	}
	return nil
}
