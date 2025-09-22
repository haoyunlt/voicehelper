package logger

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"chatbot/common/errors"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// LogLevel 日志级别
type LogLevel string

const (
	DebugLevel LogLevel = "debug"
	InfoLevel  LogLevel = "info"
	WarnLevel  LogLevel = "warn"
	ErrorLevel LogLevel = "error"
	FatalLevel LogLevel = "fatal"
	PanicLevel LogLevel = "panic"
)

// LogType 日志类型
type LogType string

const (
	StartupLog     LogType = "startup"     // 启动日志
	RequestLog     LogType = "request"     // 请求日志
	ResponseLog    LogType = "response"    // 响应日志
	ErrorLog       LogType = "error"       // 错误日志
	DebugLog       LogType = "debug"       // 调试日志
	PerformanceLog LogType = "performance" // 性能日志
	SecurityLog    LogType = "security"    // 安全日志
	BusinessLog    LogType = "business"    // 业务日志
	SystemLog      LogType = "system"      // 系统日志
)

// NetworkInfo 网络信息
type NetworkInfo struct {
	LocalIP    string `json:"local_ip"`
	LocalPort  string `json:"local_port"`
	RemoteIP   string `json:"remote_ip"`
	RemotePort string `json:"remote_port"`
	URL        string `json:"url"`
	Method     string `json:"method"`
	UserAgent  string `json:"user_agent"`
	RequestID  string `json:"request_id"`
}

// LogEntry 日志条目
type LogEntry struct {
	Timestamp    time.Time              `json:"timestamp"`
	Level        LogLevel               `json:"level"`
	Type         LogType                `json:"type"`
	Service      string                 `json:"service"`
	Module       string                 `json:"module"`
	Message      string                 `json:"message"`
	ErrorCode    errors.ErrorCode       `json:"error_code,omitempty"`
	Network      *NetworkInfo           `json:"network,omitempty"`
	Context      map[string]interface{} `json:"context,omitempty"`
	Stack        string                 `json:"stack,omitempty"`
	Duration     time.Duration          `json:"duration,omitempty"`
	RequestSize  int64                  `json:"request_size,omitempty"`
	ResponseSize int64                  `json:"response_size,omitempty"`
	StatusCode   int                    `json:"status_code,omitempty"`
}

// Logger 日志器接口
type Logger interface {
	Debug(msg string, fields ...Field)
	Info(msg string, fields ...Field)
	Warn(msg string, fields ...Field)
	Error(msg string, fields ...Field)
	Fatal(msg string, fields ...Field)
	Panic(msg string, fields ...Field)

	// 特定类型日志
	Startup(msg string, fields ...Field)
	Request(req *http.Request, fields ...Field)
	Response(req *http.Request, statusCode int, duration time.Duration, fields ...Field)
	ErrorWithCode(code errors.ErrorCode, msg string, fields ...Field)
	Performance(operation string, duration time.Duration, fields ...Field)
	Security(event string, fields ...Field)
	Business(event string, fields ...Field)

	// 上下文日志
	WithContext(ctx context.Context) Logger
	WithFields(fields ...Field) Logger
	WithService(service string) Logger
	WithModule(module string) Logger
}

// Field 日志字段
type Field struct {
	Key   string
	Value interface{}
}

// VoiceHelperLogger VoiceHelper日志器实现
type VoiceHelperLogger struct {
	logger  *logrus.Logger
	service string
	module  string
	fields  map[string]interface{}
	ctx     context.Context
}

// NewLogger 创建新的日志器
func NewLogger(service string) Logger {
	logger := logrus.New()

	// 设置日志格式为JSON
	logger.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: time.RFC3339Nano,
		FieldMap: logrus.FieldMap{
			logrus.FieldKeyTime:  "timestamp",
			logrus.FieldKeyLevel: "level",
			logrus.FieldKeyMsg:   "message",
		},
	})

	// 设置日志级别
	level := strings.ToLower(os.Getenv("LOG_LEVEL"))
	switch level {
	case "debug":
		logger.SetLevel(logrus.DebugLevel)
	case "info":
		logger.SetLevel(logrus.InfoLevel)
	case "warn":
		logger.SetLevel(logrus.WarnLevel)
	case "error":
		logger.SetLevel(logrus.ErrorLevel)
	case "fatal":
		logger.SetLevel(logrus.FatalLevel)
	case "panic":
		logger.SetLevel(logrus.PanicLevel)
	default:
		logger.SetLevel(logrus.InfoLevel)
	}

	// 设置输出
	logger.SetOutput(os.Stdout)

	return &VoiceHelperLogger{
		logger:  logger,
		service: service,
		fields:  make(map[string]interface{}),
	}
}

// getNetworkInfo 获取网络信息
func getNetworkInfo(req *http.Request) *NetworkInfo {
	if req == nil {
		return nil
	}

	// 获取本地IP和端口
	localAddr := req.Context().Value(http.LocalAddrContextKey)
	var localIP, localPort string
	if addr, ok := localAddr.(net.Addr); ok {
		host, port, _ := net.SplitHostPort(addr.String())
		localIP = host
		localPort = port
	}

	// 获取远程IP和端口
	remoteIP := getClientIP(req)
	remotePort := ""
	if host, port, err := net.SplitHostPort(req.RemoteAddr); err == nil {
		if remoteIP == "" {
			remoteIP = host
		}
		remotePort = port
	}

	// 获取请求ID
	requestID := req.Header.Get("X-Request-ID")
	if requestID == "" {
		requestID = req.Header.Get("X-Trace-ID")
	}

	return &NetworkInfo{
		LocalIP:    localIP,
		LocalPort:  localPort,
		RemoteIP:   remoteIP,
		RemotePort: remotePort,
		URL:        req.URL.String(),
		Method:     req.Method,
		UserAgent:  req.UserAgent(),
		RequestID:  requestID,
	}
}

// getClientIP 获取客户端真实IP
func getClientIP(req *http.Request) string {
	// 检查X-Forwarded-For头
	if xff := req.Header.Get("X-Forwarded-For"); xff != "" {
		ips := strings.Split(xff, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// 检查X-Real-IP头
	if xri := req.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// 检查X-Client-IP头
	if xci := req.Header.Get("X-Client-IP"); xci != "" {
		return xci
	}

	// 使用RemoteAddr
	if ip, _, err := net.SplitHostPort(req.RemoteAddr); err == nil {
		return ip
	}

	return req.RemoteAddr
}

// buildLogEntry 构建日志条目
func (l *VoiceHelperLogger) buildLogEntry(level LogLevel, logType LogType, msg string, fields []Field) *LogEntry {
	entry := &LogEntry{
		Timestamp: time.Now(),
		Level:     level,
		Type:      logType,
		Service:   l.service,
		Module:    l.module,
		Message:   msg,
		Context:   make(map[string]interface{}),
	}

	// 添加基础字段
	for k, v := range l.fields {
		entry.Context[k] = v
	}

	// 添加传入的字段
	for _, field := range fields {
		entry.Context[field.Key] = field.Value
	}

	// 添加调用栈信息（仅错误级别）
	if level == ErrorLevel || level == FatalLevel || level == PanicLevel {
		entry.Stack = getStackTrace()
	}

	return entry
}

// getStackTrace 获取调用栈
func getStackTrace() string {
	buf := make([]byte, 4096)
	n := runtime.Stack(buf, false)
	return string(buf[:n])
}

// log 记录日志
func (l *VoiceHelperLogger) log(level LogLevel, logType LogType, msg string, fields []Field) {
	entry := l.buildLogEntry(level, logType, msg, fields)

	// 转换为logrus字段
	logrusFields := logrus.Fields{
		"type":    string(entry.Type),
		"service": entry.Service,
		"module":  entry.Module,
	}

	if entry.ErrorCode != 0 {
		logrusFields["error_code"] = entry.ErrorCode
	}

	if entry.Network != nil {
		logrusFields["network"] = entry.Network
	}

	if entry.Context != nil && len(entry.Context) > 0 {
		logrusFields["context"] = entry.Context
	}

	if entry.Stack != "" {
		logrusFields["stack"] = entry.Stack
	}

	if entry.Duration > 0 {
		logrusFields["duration_ms"] = entry.Duration.Milliseconds()
	}

	if entry.RequestSize > 0 {
		logrusFields["request_size"] = entry.RequestSize
	}

	if entry.ResponseSize > 0 {
		logrusFields["response_size"] = entry.ResponseSize
	}

	if entry.StatusCode > 0 {
		logrusFields["status_code"] = entry.StatusCode
	}

	// 记录日志
	logEntry := l.logger.WithFields(logrusFields)
	switch level {
	case DebugLevel:
		logEntry.Debug(msg)
	case InfoLevel:
		logEntry.Info(msg)
	case WarnLevel:
		logEntry.Warn(msg)
	case ErrorLevel:
		logEntry.Error(msg)
	case FatalLevel:
		logEntry.Fatal(msg)
	case PanicLevel:
		logEntry.Panic(msg)
	}
}

// Debug 记录调试日志
func (l *VoiceHelperLogger) Debug(msg string, fields ...Field) {
	l.log(DebugLevel, DebugLog, msg, fields)
}

// Info 记录信息日志
func (l *VoiceHelperLogger) Info(msg string, fields ...Field) {
	l.log(InfoLevel, SystemLog, msg, fields)
}

// Warn 记录警告日志
func (l *VoiceHelperLogger) Warn(msg string, fields ...Field) {
	l.log(WarnLevel, SystemLog, msg, fields)
}

// Error 记录错误日志
func (l *VoiceHelperLogger) Error(msg string, fields ...Field) {
	l.log(ErrorLevel, ErrorLog, msg, fields)
}

// Fatal 记录致命错误日志
func (l *VoiceHelperLogger) Fatal(msg string, fields ...Field) {
	l.log(FatalLevel, ErrorLog, msg, fields)
}

// Panic 记录恐慌日志
func (l *VoiceHelperLogger) Panic(msg string, fields ...Field) {
	l.log(PanicLevel, ErrorLog, msg, fields)
}

// Startup 记录启动日志
func (l *VoiceHelperLogger) Startup(msg string, fields ...Field) {
	l.log(InfoLevel, StartupLog, msg, fields)
}

// Request 记录请求日志
func (l *VoiceHelperLogger) Request(req *http.Request, fields ...Field) {
	networkInfo := getNetworkInfo(req)
	allFields := append(fields, Field{Key: "network", Value: networkInfo})

	if req.ContentLength > 0 {
		allFields = append(allFields, Field{Key: "request_size", Value: req.ContentLength})
	}

	l.log(InfoLevel, RequestLog, fmt.Sprintf("%s %s", req.Method, req.URL.Path), allFields)
}

// Response 记录响应日志
func (l *VoiceHelperLogger) Response(req *http.Request, statusCode int, duration time.Duration, fields ...Field) {
	networkInfo := getNetworkInfo(req)
	allFields := append(fields,
		Field{Key: "network", Value: networkInfo},
		Field{Key: "status_code", Value: statusCode},
		Field{Key: "duration_ms", Value: duration.Milliseconds()},
	)

	level := InfoLevel
	if statusCode >= 400 {
		level = WarnLevel
	}
	if statusCode >= 500 {
		level = ErrorLevel
	}

	l.log(level, ResponseLog, fmt.Sprintf("%s %s - %d", req.Method, req.URL.Path, statusCode), allFields)
}

// ErrorWithCode 记录带错误码的错误日志
func (l *VoiceHelperLogger) ErrorWithCode(code errors.ErrorCode, msg string, fields ...Field) {
	allFields := append(fields, Field{Key: "error_code", Value: code})
	l.log(ErrorLevel, ErrorLog, msg, allFields)
}

// Performance 记录性能日志
func (l *VoiceHelperLogger) Performance(operation string, duration time.Duration, fields ...Field) {
	allFields := append(fields, Field{Key: "duration_ms", Value: duration.Milliseconds()})
	l.log(InfoLevel, PerformanceLog, fmt.Sprintf("Performance: %s", operation), allFields)
}

// Security 记录安全日志
func (l *VoiceHelperLogger) Security(event string, fields ...Field) {
	l.log(WarnLevel, SecurityLog, fmt.Sprintf("Security Event: %s", event), fields)
}

// Business 记录业务日志
func (l *VoiceHelperLogger) Business(event string, fields ...Field) {
	l.log(InfoLevel, BusinessLog, fmt.Sprintf("Business Event: %s", event), fields)
}

// WithContext 添加上下文
func (l *VoiceHelperLogger) WithContext(ctx context.Context) Logger {
	newLogger := *l
	newLogger.ctx = ctx
	newLogger.fields = make(map[string]interface{})
	for k, v := range l.fields {
		newLogger.fields[k] = v
	}

	// 从上下文中提取信息
	if requestID := ctx.Value("request_id"); requestID != nil {
		newLogger.fields["request_id"] = requestID
	}
	if userID := ctx.Value("user_id"); userID != nil {
		newLogger.fields["user_id"] = userID
	}
	if traceID := ctx.Value("trace_id"); traceID != nil {
		newLogger.fields["trace_id"] = traceID
	}

	return &newLogger
}

// WithFields 添加字段
func (l *VoiceHelperLogger) WithFields(fields ...Field) Logger {
	newLogger := *l
	newLogger.fields = make(map[string]interface{})
	for k, v := range l.fields {
		newLogger.fields[k] = v
	}

	for _, field := range fields {
		newLogger.fields[field.Key] = field.Value
	}

	return &newLogger
}

// WithService 设置服务名
func (l *VoiceHelperLogger) WithService(service string) Logger {
	newLogger := *l
	newLogger.service = service
	return &newLogger
}

// WithModule 设置模块名
func (l *VoiceHelperLogger) WithModule(module string) Logger {
	newLogger := *l
	newLogger.module = module
	return &newLogger
}

// GinLoggerMiddleware Gin日志中间件
func GinLoggerMiddleware(logger Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		// 记录请求日志
		logger.Request(c.Request)

		// 处理请求
		c.Next()

		// 记录响应日志
		duration := time.Since(start)
		logger.Response(c.Request, c.Writer.Status(), duration)

		// 如果有错误，记录错误日志
		if len(c.Errors) > 0 {
			for _, err := range c.Errors {
				logger.Error("Request error", Field{Key: "error", Value: err.Error()})
			}
		}
	}
}

// 便利函数
func F(key string, value interface{}) Field {
	return Field{Key: key, Value: value}
}

// 全局日志器实例
var defaultLogger Logger

// InitDefaultLogger 初始化默认日志器
func InitDefaultLogger(service string) {
	defaultLogger = NewLogger(service)
}

// GetDefaultLogger 获取默认日志器
func GetDefaultLogger() Logger {
	if defaultLogger == nil {
		defaultLogger = NewLogger("voicehelper")
	}
	return defaultLogger
}

// 全局便利函数
func Debug(msg string, fields ...Field) {
	GetDefaultLogger().Debug(msg, fields...)
}

func Info(msg string, fields ...Field) {
	GetDefaultLogger().Info(msg, fields...)
}

func Warn(msg string, fields ...Field) {
	GetDefaultLogger().Warn(msg, fields...)
}

func Error(msg string, fields ...Field) {
	GetDefaultLogger().Error(msg, fields...)
}

func Fatal(msg string, fields ...Field) {
	GetDefaultLogger().Fatal(msg, fields...)
}

func Panic(msg string, fields ...Field) {
	GetDefaultLogger().Panic(msg, fields...)
}

func Startup(msg string, fields ...Field) {
	GetDefaultLogger().Startup(msg, fields...)
}

func ErrorWithCode(code errors.ErrorCode, msg string, fields ...Field) {
	GetDefaultLogger().ErrorWithCode(code, msg, fields...)
}

func Performance(operation string, duration time.Duration, fields ...Field) {
	GetDefaultLogger().Performance(operation, duration, fields...)
}

func Security(event string, fields ...Field) {
	GetDefaultLogger().Security(event, fields...)
}

func Business(event string, fields ...Field) {
	GetDefaultLogger().Business(event, fields...)
}
