package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// MetricsCollector 统一的指标收集器
type MetricsCollector struct {
	// HTTP请求指标
	httpRequestsTotal    *prometheus.CounterVec
	httpRequestDuration  *prometheus.HistogramVec
	httpRequestsInFlight *prometheus.GaugeVec

	// 会话管理指标
	conversationsTotal   *prometheus.CounterVec
	conversationsActive  *prometheus.GaugeVec
	conversationDuration *prometheus.HistogramVec
	messagesTotal        *prometheus.CounterVec

	// Agent工具指标
	agentToolExecutions    *prometheus.CounterVec
	agentToolDuration      *prometheus.HistogramVec
	agentStreamConnections *prometheus.GaugeVec

	// 文档搜索指标
	documentSearches       *prometheus.CounterVec
	documentSearchDuration *prometheus.HistogramVec
	documentSearchResults  *prometheus.HistogramVec

	// WebSocket指标
	websocketConnections *prometheus.GaugeVec
	websocketMessages    *prometheus.CounterVec
	websocketErrors      *prometheus.CounterVec

	// 语音处理指标
	voiceSessionsTotal   *prometheus.CounterVec
	voiceSessionDuration *prometheus.HistogramVec
	voiceLatency         *prometheus.HistogramVec

	// 系统资源指标
	memoryUsage     *prometheus.GaugeVec
	cpuUsage        *prometheus.GaugeVec
	goroutinesCount prometheus.Gauge

	// 错误指标
	errorsTotal *prometheus.CounterVec

	once sync.Once
}

var (
	instance *MetricsCollector
	mu       sync.RWMutex
)

// GetMetricsCollector 获取指标收集器单例
func GetMetricsCollector() *MetricsCollector {
	mu.RLock()
	if instance != nil {
		mu.RUnlock()
		return instance
	}
	mu.RUnlock()

	mu.Lock()
	defer mu.Unlock()
	if instance == nil {
		instance = &MetricsCollector{}
		instance.initMetrics()
	}
	return instance
}

// initMetrics 初始化所有指标
func (m *MetricsCollector) initMetrics() {
	m.once.Do(func() {
		// HTTP请求指标
		m.httpRequestsTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_http_requests_total",
				Help: "Total number of HTTP requests",
			},
			[]string{"method", "path", "status_code", "tenant_id"},
		)

		m.httpRequestDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_http_request_duration_seconds",
				Help:    "HTTP request duration in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method", "path", "tenant_id"},
		)

		m.httpRequestsInFlight = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_http_requests_in_flight",
				Help: "Number of HTTP requests currently being processed",
			},
			[]string{"method", "path"},
		)

		// 会话管理指标
		m.conversationsTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_conversations_total",
				Help: "Total number of conversations",
			},
			[]string{"action", "tenant_id", "user_id"},
		)

		m.conversationsActive = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_conversations_active",
				Help: "Number of active conversations",
			},
			[]string{"tenant_id"},
		)

		m.conversationDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_conversation_duration_seconds",
				Help:    "Conversation duration in seconds",
				Buckets: []float64{1, 5, 10, 30, 60, 300, 600, 1800, 3600},
			},
			[]string{"tenant_id"},
		)

		m.messagesTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_messages_total",
				Help: "Total number of messages",
			},
			[]string{"role", "modality", "tenant_id"},
		)

		// Agent工具指标
		m.agentToolExecutions = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_agent_tool_executions_total",
				Help: "Total number of agent tool executions",
			},
			[]string{"tool_name", "status", "tenant_id"},
		)

		m.agentToolDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_agent_tool_duration_seconds",
				Help:    "Agent tool execution duration in seconds",
				Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 30},
			},
			[]string{"tool_name", "tenant_id"},
		)

		m.agentStreamConnections = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_agent_stream_connections",
				Help: "Number of active agent stream connections",
			},
			[]string{"tenant_id"},
		)

		// 文档搜索指标
		m.documentSearches = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_document_searches_total",
				Help: "Total number of document searches",
			},
			[]string{"status", "tenant_id"},
		)

		m.documentSearchDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_document_search_duration_seconds",
				Help:    "Document search duration in seconds",
				Buckets: []float64{0.1, 0.5, 1, 2, 5, 10},
			},
			[]string{"tenant_id"},
		)

		m.documentSearchResults = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_document_search_results",
				Help:    "Number of document search results",
				Buckets: []float64{0, 1, 5, 10, 20, 50, 100},
			},
			[]string{"tenant_id"},
		)

		// WebSocket指标
		m.websocketConnections = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_websocket_connections",
				Help: "Number of active WebSocket connections",
			},
			[]string{"connection_type", "tenant_id"},
		)

		m.websocketMessages = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_websocket_messages_total",
				Help: "Total number of WebSocket messages",
			},
			[]string{"direction", "message_type", "tenant_id"},
		)

		m.websocketErrors = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_websocket_errors_total",
				Help: "Total number of WebSocket errors",
			},
			[]string{"error_type", "tenant_id"},
		)

		// 语音处理指标
		m.voiceSessionsTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_sessions_total",
				Help: "Total number of voice sessions",
			},
			[]string{"status", "tenant_id"},
		)

		m.voiceSessionDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_session_duration_seconds",
				Help:    "Voice session duration in seconds",
				Buckets: []float64{1, 5, 10, 30, 60, 300, 600},
			},
			[]string{"tenant_id"},
		)

		m.voiceLatency = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_latency_seconds",
				Help:    "Voice processing latency in seconds",
				Buckets: []float64{0.1, 0.2, 0.5, 1, 2, 5},
			},
			[]string{"processing_type", "tenant_id"},
		)

		// 系统资源指标
		m.memoryUsage = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_memory_usage_bytes",
				Help: "Memory usage in bytes",
			},
			[]string{"type"},
		)

		m.cpuUsage = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_cpu_usage_percent",
				Help: "CPU usage percentage",
			},
			[]string{"type"},
		)

		m.goroutinesCount = promauto.NewGauge(
			prometheus.GaugeOpts{
				Name: "voicehelper_goroutines_count",
				Help: "Number of goroutines",
			},
		)

		// 错误指标
		m.errorsTotal = promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_errors_total",
				Help: "Total number of errors",
			},
			[]string{"error_type", "component", "tenant_id"},
		)
	})
}

// HTTP请求指标方法
func (m *MetricsCollector) IncHTTPRequests(method, path, statusCode, tenantID string) {
	m.httpRequestsTotal.WithLabelValues(method, path, statusCode, tenantID).Inc()
}

func (m *MetricsCollector) ObserveHTTPDuration(method, path, tenantID string, duration time.Duration) {
	m.httpRequestDuration.WithLabelValues(method, path, tenantID).Observe(duration.Seconds())
}

func (m *MetricsCollector) IncHTTPInFlight(method, path string) {
	m.httpRequestsInFlight.WithLabelValues(method, path).Inc()
}

func (m *MetricsCollector) DecHTTPInFlight(method, path string) {
	m.httpRequestsInFlight.WithLabelValues(method, path).Dec()
}

// 会话管理指标方法
func (m *MetricsCollector) IncConversations(action, tenantID, userID string) {
	m.conversationsTotal.WithLabelValues(action, tenantID, userID).Inc()
}

func (m *MetricsCollector) SetActiveConversations(tenantID string, count float64) {
	m.conversationsActive.WithLabelValues(tenantID).Set(count)
}

func (m *MetricsCollector) ObserveConversationDuration(tenantID string, duration time.Duration) {
	m.conversationDuration.WithLabelValues(tenantID).Observe(duration.Seconds())
}

func (m *MetricsCollector) IncMessages(role, modality, tenantID string) {
	m.messagesTotal.WithLabelValues(role, modality, tenantID).Inc()
}

// Agent工具指标方法
func (m *MetricsCollector) IncAgentToolExecutions(toolName, status, tenantID string) {
	m.agentToolExecutions.WithLabelValues(toolName, status, tenantID).Inc()
}

func (m *MetricsCollector) ObserveAgentToolDuration(toolName, tenantID string, duration time.Duration) {
	m.agentToolDuration.WithLabelValues(toolName, tenantID).Observe(duration.Seconds())
}

func (m *MetricsCollector) IncAgentStreamConnections(tenantID string) {
	m.agentStreamConnections.WithLabelValues(tenantID).Inc()
}

func (m *MetricsCollector) DecAgentStreamConnections(tenantID string) {
	m.agentStreamConnections.WithLabelValues(tenantID).Dec()
}

// 文档搜索指标方法
func (m *MetricsCollector) IncDocumentSearches(status, tenantID string) {
	m.documentSearches.WithLabelValues(status, tenantID).Inc()
}

func (m *MetricsCollector) ObserveDocumentSearchDuration(tenantID string, duration time.Duration) {
	m.documentSearchDuration.WithLabelValues(tenantID).Observe(duration.Seconds())
}

func (m *MetricsCollector) ObserveDocumentSearchResults(tenantID string, count float64) {
	m.documentSearchResults.WithLabelValues(tenantID).Observe(count)
}

// WebSocket指标方法
func (m *MetricsCollector) IncWebSocketConnections(connectionType, tenantID string) {
	m.websocketConnections.WithLabelValues(connectionType, tenantID).Inc()
}

func (m *MetricsCollector) DecWebSocketConnections(connectionType, tenantID string) {
	m.websocketConnections.WithLabelValues(connectionType, tenantID).Dec()
}

func (m *MetricsCollector) IncWebSocketMessages(direction, messageType, tenantID string) {
	m.websocketMessages.WithLabelValues(direction, messageType, tenantID).Inc()
}

func (m *MetricsCollector) IncWebSocketErrors(errorType, tenantID string) {
	m.websocketErrors.WithLabelValues(errorType, tenantID).Inc()
}

// 语音处理指标方法
func (m *MetricsCollector) IncVoiceSessions(status, tenantID string) {
	m.voiceSessionsTotal.WithLabelValues(status, tenantID).Inc()
}

func (m *MetricsCollector) ObserveVoiceSessionDuration(tenantID string, duration time.Duration) {
	m.voiceSessionDuration.WithLabelValues(tenantID).Observe(duration.Seconds())
}

func (m *MetricsCollector) ObserveVoiceLatency(processingType, tenantID string, latency time.Duration) {
	m.voiceLatency.WithLabelValues(processingType, tenantID).Observe(latency.Seconds())
}

// 系统资源指标方法
func (m *MetricsCollector) SetMemoryUsage(memType string, bytes float64) {
	m.memoryUsage.WithLabelValues(memType).Set(bytes)
}

func (m *MetricsCollector) SetCPUUsage(cpuType string, percent float64) {
	m.cpuUsage.WithLabelValues(cpuType).Set(percent)
}

func (m *MetricsCollector) SetGoroutinesCount(count float64) {
	m.goroutinesCount.Set(count)
}

// 错误指标方法
func (m *MetricsCollector) IncErrors(errorType, component, tenantID string) {
	m.errorsTotal.WithLabelValues(errorType, component, tenantID).Inc()
}

// 便捷方法
func IncHTTPRequests(method, path, statusCode, tenantID string) {
	GetMetricsCollector().IncHTTPRequests(method, path, statusCode, tenantID)
}

func ObserveHTTPDuration(method, path, tenantID string, duration time.Duration) {
	GetMetricsCollector().ObserveHTTPDuration(method, path, tenantID, duration)
}

func IncConversations(action, tenantID, userID string) {
	GetMetricsCollector().IncConversations(action, tenantID, userID)
}

func IncAgentToolExecutions(toolName, status, tenantID string) {
	GetMetricsCollector().IncAgentToolExecutions(toolName, status, tenantID)
}

func IncDocumentSearches(status, tenantID string) {
	GetMetricsCollector().IncDocumentSearches(status, tenantID)
}

func IncErrors(errorType, component, tenantID string) {
	GetMetricsCollector().IncErrors(errorType, component, tenantID)
}
