package middleware

import (
	"strconv"
	"time"

	"voicehelper/backend/pkg/metrics"

	"github.com/gin-gonic/gin"
)

// MetricsMiddleware 指标收集中间件
func MetricsMiddleware() gin.HandlerFunc {
	collector := metrics.GetMetricsCollector()

	return func(c *gin.Context) {
		start := time.Now()
		method := c.Request.Method
		path := c.FullPath()
		tenantID := c.GetString("tenant_id")

		// 如果没有租户ID，使用默认值
		if tenantID == "" {
			tenantID = "default"
		}

		// 增加进行中的请求计数
		collector.IncHTTPInFlight(method, path)

		// 处理请求
		c.Next()

		// 减少进行中的请求计数
		collector.DecHTTPInFlight(method, path)

		// 记录请求指标
		duration := time.Since(start)
		statusCode := strconv.Itoa(c.Writer.Status())

		collector.IncHTTPRequests(method, path, statusCode, tenantID)
		collector.ObserveHTTPDuration(method, path, tenantID, duration)

		// 记录错误指标
		if c.Writer.Status() >= 400 {
			errorType := "client_error"
			if c.Writer.Status() >= 500 {
				errorType = "server_error"
			}
			collector.IncErrors(errorType, "http", tenantID)
		}
	}
}

// ConversationMetricsMiddleware 会话指标中间件
func ConversationMetricsMiddleware() gin.HandlerFunc {
	collector := metrics.GetMetricsCollector()

	return func(c *gin.Context) {
		tenantID := c.GetString("tenant_id")
		userID := c.GetString("user_id")

		if tenantID == "" {
			tenantID = "default"
		}
		if userID == "" {
			userID = "anonymous"
		}

		// 根据路径和方法确定操作类型
		path := c.FullPath()
		method := c.Request.Method

		var action string
		switch {
		case path == "/api/v2/conversations" && method == "POST":
			action = "create"
		case path == "/api/v2/conversations/:id" && method == "GET":
			action = "get"
		case path == "/api/v2/conversations/:id" && method == "PUT":
			action = "update"
		case path == "/api/v2/conversations/:id" && method == "DELETE":
			action = "delete"
		case path == "/api/v2/conversations" && method == "GET":
			action = "list"
		default:
			action = "other"
		}

		c.Next()

		// 只在成功时记录指标
		if c.Writer.Status() < 400 {
			collector.IncConversations(action, tenantID, userID)
		}
	}
}

// AgentMetricsMiddleware Agent指标中间件
func AgentMetricsMiddleware() gin.HandlerFunc {
	collector := metrics.GetMetricsCollector()

	return func(c *gin.Context) {
		tenantID := c.GetString("tenant_id")
		if tenantID == "" {
			tenantID = "default"
		}

		path := c.FullPath()

		// 检查是否是Agent流式连接
		if path == "/api/v2/agent/stream" {
			collector.IncAgentStreamConnections(tenantID)
			defer collector.DecAgentStreamConnections(tenantID)
		}

		c.Next()
	}
}

// DocumentSearchMetricsMiddleware 文档搜索指标中间件
func DocumentSearchMetricsMiddleware() gin.HandlerFunc {
	collector := metrics.GetMetricsCollector()

	return func(c *gin.Context) {
		tenantID := c.GetString("tenant_id")
		if tenantID == "" {
			tenantID = "default"
		}

		path := c.FullPath()

		// 只对搜索相关的路径记录指标
		if path != "/api/v2/documents/search" {
			c.Next()
			return
		}

		start := time.Now()
		c.Next()
		duration := time.Since(start)

		status := "success"
		if c.Writer.Status() >= 400 {
			status = "error"
		}

		collector.IncDocumentSearches(status, tenantID)
		collector.ObserveDocumentSearchDuration(tenantID, duration)
	}
}

// WebSocketMetricsMiddleware WebSocket指标中间件
func WebSocketMetricsMiddleware(connectionType string) gin.HandlerFunc {
	collector := metrics.GetMetricsCollector()

	return func(c *gin.Context) {
		tenantID := c.GetString("tenant_id")
		if tenantID == "" {
			tenantID = "default"
		}

		// WebSocket连接建立时增加计数
		collector.IncWebSocketConnections(connectionType, tenantID)

		c.Next()

		// WebSocket连接关闭时减少计数
		collector.DecWebSocketConnections(connectionType, tenantID)
	}
}

// VoiceProcessingMetricsMiddleware 语音处理指标中间件
func VoiceProcessingMetricsMiddleware() gin.HandlerFunc {
	collector := metrics.GetMetricsCollector()

	return func(c *gin.Context) {
		tenantID := c.GetString("tenant_id")
		if tenantID == "" {
			tenantID = "default"
		}

		path := c.FullPath()

		// 只对语音相关的路径记录指标
		if path != "/api/v2/voice/stream" && path != "/api/v2/voice/query" {
			c.Next()
			return
		}

		start := time.Now()
		c.Next()
		duration := time.Since(start)

		status := "success"
		if c.Writer.Status() >= 400 {
			status = "error"
		}

		collector.IncVoiceSessions(status, tenantID)
		collector.ObserveVoiceSessionDuration(tenantID, duration)
	}
}

// SystemMetricsCollector 系统指标收集器
type SystemMetricsCollector struct {
	collector *metrics.MetricsCollector
	stopCh    chan struct{}
}

// NewSystemMetricsCollector 创建系统指标收集器
func NewSystemMetricsCollector() *SystemMetricsCollector {
	return &SystemMetricsCollector{
		collector: metrics.GetMetricsCollector(),
		stopCh:    make(chan struct{}),
	}
}

// Start 开始收集系统指标
func (s *SystemMetricsCollector) Start() {
	go s.collectSystemMetrics()
}

// Stop 停止收集系统指标
func (s *SystemMetricsCollector) Stop() {
	close(s.stopCh)
}

// collectSystemMetrics 收集系统指标
func (s *SystemMetricsCollector) collectSystemMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.updateSystemMetrics()
		case <-s.stopCh:
			return
		}
	}
}

// updateSystemMetrics 更新系统指标
func (s *SystemMetricsCollector) updateSystemMetrics() {
	// 这里应该实现实际的系统指标收集
	// 为了简化，使用模拟数据

	// 模拟内存使用
	s.collector.SetMemoryUsage("heap", 1024*1024*100) // 100MB
	s.collector.SetMemoryUsage("stack", 1024*1024*10) // 10MB

	// 模拟CPU使用
	s.collector.SetCPUUsage("user", 25.5)
	s.collector.SetCPUUsage("system", 10.2)

	// 模拟Goroutine数量
	s.collector.SetGoroutinesCount(150)
}
