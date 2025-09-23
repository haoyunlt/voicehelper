package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	once            sync.Once
	metricsRegistry *MetricsRegistry
)

// MetricsCollector 指标收集器接口
type MetricsCollector interface {
	RecordHTTPRequest(method, endpoint, statusCode, tenantID string, duration time.Duration, requestSize, responseSize float64)
	RecordVoiceProcessing(processType, model, language string, duration time.Duration)
	RecordASRRequest(model, language, status string, latency time.Duration)
	RecordTTSRequest(voice, language, status string, latency time.Duration)
	RecordLLMRequest(model, provider, status string, latency time.Duration, inputTokens, outputTokens int)
	IncrementWSConnections(connectionType, tenantID string)
	DecrementWSConnections()
	UpdateActiveUsers(count float64)

	// HTTP 指标方法
	IncHTTPInFlight()
	DecHTTPInFlight()
	IncHTTPRequests(method, endpoint, statusCode, tenantID string)
	ObserveHTTPDuration(method, endpoint, tenantID string, duration time.Duration)
	IncErrors(errorType, tenantID string)

	// 业务指标方法
	IncConversations(conversationType, tenantID string)
	IncAgentStreamConnections(tenantID string)
	DecAgentStreamConnections()
	IncDocumentSearches(searchType, tenantID string)
	ObserveDocumentSearchDuration(searchType, tenantID string, duration time.Duration)

	// WebSocket 指标方法
	IncWebSocketConnections(connectionType, tenantID string)
	DecWebSocketConnections()

	// Voice 指标方法
	IncVoiceSessions(sessionType, tenantID string)
	ObserveVoiceSessionDuration(sessionType, tenantID string, duration time.Duration)

	// 系统指标方法
	SetMemoryUsage(usage float64)
	SetCPUUsage(usage float64)
	SetGoroutinesCount(count float64)
}

type MetricsRegistry struct {
	// HTTP请求指标
	HTTPRequestsTotal    *prometheus.CounterVec
	HTTPRequestDuration  *prometheus.HistogramVec
	HTTPRequestSize      *prometheus.HistogramVec
	HTTPResponseSize     *prometheus.HistogramVec
	HTTPRequestsInFlight prometheus.Gauge
	ErrorsTotal          *prometheus.CounterVec

	// WebSocket连接指标
	WSConnectionsActive prometheus.Gauge
	WSConnectionsTotal  *prometheus.CounterVec
	WSMessagesSent      *prometheus.CounterVec
	WSMessagesReceived  *prometheus.CounterVec

	// 语音处理指标
	VoiceSessionsActive     prometheus.Gauge
	VoiceProcessingDuration *prometheus.HistogramVec
	ASRRequestsTotal        *prometheus.CounterVec
	ASRLatency              *prometheus.HistogramVec
	TTSRequestsTotal        *prometheus.CounterVec
	TTSLatency              *prometheus.HistogramVec

	// AI服务指标
	LLMRequestsTotal *prometheus.CounterVec
	LLMTokensUsed    *prometheus.CounterVec
	LLMLatency       *prometheus.HistogramVec
	RAGQueryTotal    *prometheus.CounterVec
	RAGLatency       *prometheus.HistogramVec

	// 系统资源指标
	MemoryUsage     prometheus.Gauge
	CPUUsage        prometheus.Gauge
	DiskUsage       prometheus.Gauge
	GoroutinesCount prometheus.Gauge

	// 业务指标
	ActiveUsers        prometheus.Gauge
	TotalConversations *prometheus.CounterVec
	UserSatisfaction   *prometheus.HistogramVec
}

func GetMetricsRegistry() *MetricsRegistry {
	once.Do(func() {
		metricsRegistry = &MetricsRegistry{
			// HTTP请求指标
			HTTPRequestsTotal: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_http_requests_total",
					Help: "Total number of HTTP requests",
				},
				[]string{"method", "endpoint", "status_code", "tenant_id"},
			),

			HTTPRequestDuration: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_http_request_duration_seconds",
					Help:    "HTTP request duration in seconds",
					Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
				},
				[]string{"method", "endpoint", "tenant_id"},
			),

			HTTPRequestSize: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_http_request_size_bytes",
					Help:    "HTTP request size in bytes",
					Buckets: prometheus.ExponentialBuckets(100, 10, 8),
				},
				[]string{"method", "endpoint"},
			),

			HTTPResponseSize: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_http_response_size_bytes",
					Help:    "HTTP response size in bytes",
					Buckets: prometheus.ExponentialBuckets(100, 10, 8),
				},
				[]string{"method", "endpoint"},
			),

			HTTPRequestsInFlight: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_http_requests_in_flight",
					Help: "Number of HTTP requests currently being processed",
				},
			),

			ErrorsTotal: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_errors_total",
					Help: "Total number of errors",
				},
				[]string{"error_type", "tenant_id"},
			),

			// WebSocket连接指标
			WSConnectionsActive: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_ws_connections_active",
					Help: "Number of active WebSocket connections",
				},
			),

			WSConnectionsTotal: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_ws_connections_total",
					Help: "Total number of WebSocket connections",
				},
				[]string{"type", "tenant_id"},
			),

			WSMessagesSent: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_ws_messages_sent_total",
					Help: "Total number of WebSocket messages sent",
				},
				[]string{"type", "session_id"},
			),

			WSMessagesReceived: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_ws_messages_received_total",
					Help: "Total number of WebSocket messages received",
				},
				[]string{"type", "session_id"},
			),

			// 语音处理指标
			VoiceSessionsActive: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_voice_sessions_active",
					Help: "Number of active voice sessions",
				},
			),

			VoiceProcessingDuration: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_voice_processing_duration_seconds",
					Help:    "Voice processing duration in seconds",
					Buckets: []float64{0.1, 0.25, 0.5, 1, 2, 5, 10, 30},
				},
				[]string{"type", "model", "language"},
			),

			ASRRequestsTotal: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_asr_requests_total",
					Help: "Total number of ASR requests",
				},
				[]string{"model", "language", "status"},
			),

			ASRLatency: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_asr_latency_seconds",
					Help:    "ASR processing latency in seconds",
					Buckets: []float64{0.1, 0.2, 0.5, 1, 2, 5},
				},
				[]string{"model", "language"},
			),

			TTSRequestsTotal: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_tts_requests_total",
					Help: "Total number of TTS requests",
				},
				[]string{"voice", "language", "status"},
			),

			TTSLatency: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_tts_latency_seconds",
					Help:    "TTS processing latency in seconds",
					Buckets: []float64{0.1, 0.2, 0.5, 1, 2, 5},
				},
				[]string{"voice", "language"},
			),

			// AI服务指标
			LLMRequestsTotal: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_llm_requests_total",
					Help: "Total number of LLM requests",
				},
				[]string{"model", "provider", "status"},
			),

			LLMTokensUsed: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_llm_tokens_used_total",
					Help: "Total number of LLM tokens used",
				},
				[]string{"model", "provider", "type"},
			),

			LLMLatency: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_llm_latency_seconds",
					Help:    "LLM processing latency in seconds",
					Buckets: []float64{0.5, 1, 2, 5, 10, 20, 30},
				},
				[]string{"model", "provider"},
			),

			RAGQueryTotal: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_rag_queries_total",
					Help: "Total number of RAG queries",
				},
				[]string{"index", "status"},
			),

			RAGLatency: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_rag_latency_seconds",
					Help:    "RAG query latency in seconds",
					Buckets: []float64{0.01, 0.05, 0.1, 0.25, 0.5, 1, 2},
				},
				[]string{"index", "type"},
			),

			// 系统资源指标
			MemoryUsage: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_memory_usage_bytes",
					Help: "Memory usage in bytes",
				},
			),

			CPUUsage: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_cpu_usage_percent",
					Help: "CPU usage percentage",
				},
			),

			DiskUsage: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_disk_usage_bytes",
					Help: "Disk usage in bytes",
				},
			),

			GoroutinesCount: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_goroutines_count",
					Help: "Number of goroutines",
				},
			),

			// 业务指标
			ActiveUsers: promauto.NewGauge(
				prometheus.GaugeOpts{
					Name: "voicehelper_active_users",
					Help: "Number of active users",
				},
			),

			TotalConversations: promauto.NewCounterVec(
				prometheus.CounterOpts{
					Name: "voicehelper_conversations_total",
					Help: "Total number of conversations",
				},
				[]string{"type", "tenant_id"},
			),

			UserSatisfaction: promauto.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:    "voicehelper_user_satisfaction_score",
					Help:    "User satisfaction score",
					Buckets: []float64{1, 2, 3, 4, 5},
				},
				[]string{"tenant_id"},
			),
		}
	})

	return metricsRegistry
}

// 便捷方法
func RecordHTTPRequest(method, endpoint, statusCode, tenantID string, duration time.Duration, requestSize, responseSize float64) {
	registry := GetMetricsRegistry()

	registry.HTTPRequestsTotal.WithLabelValues(method, endpoint, statusCode, tenantID).Inc()
	registry.HTTPRequestDuration.WithLabelValues(method, endpoint, tenantID).Observe(duration.Seconds())
	registry.HTTPRequestSize.WithLabelValues(method, endpoint).Observe(requestSize)
	registry.HTTPResponseSize.WithLabelValues(method, endpoint).Observe(responseSize)
}

func RecordVoiceProcessing(processType, model, language string, duration time.Duration) {
	registry := GetMetricsRegistry()
	registry.VoiceProcessingDuration.WithLabelValues(processType, model, language).Observe(duration.Seconds())
}

func RecordASRRequest(model, language, status string, latency time.Duration) {
	registry := GetMetricsRegistry()
	registry.ASRRequestsTotal.WithLabelValues(model, language, status).Inc()
	registry.ASRLatency.WithLabelValues(model, language).Observe(latency.Seconds())
}

func RecordTTSRequest(voice, language, status string, latency time.Duration) {
	registry := GetMetricsRegistry()
	registry.TTSRequestsTotal.WithLabelValues(voice, language, status).Inc()
	registry.TTSLatency.WithLabelValues(voice, language).Observe(latency.Seconds())
}

func RecordLLMRequest(model, provider, status string, latency time.Duration, inputTokens, outputTokens int) {
	registry := GetMetricsRegistry()
	registry.LLMRequestsTotal.WithLabelValues(model, provider, status).Inc()
	registry.LLMLatency.WithLabelValues(model, provider).Observe(latency.Seconds())
	registry.LLMTokensUsed.WithLabelValues(model, provider, "input").Add(float64(inputTokens))
	registry.LLMTokensUsed.WithLabelValues(model, provider, "output").Add(float64(outputTokens))
}

func IncrementWSConnections(connectionType, tenantID string) {
	registry := GetMetricsRegistry()
	registry.WSConnectionsActive.Inc()
	registry.WSConnectionsTotal.WithLabelValues(connectionType, tenantID).Inc()
}

func DecrementWSConnections() {
	registry := GetMetricsRegistry()
	registry.WSConnectionsActive.Dec()
}

func UpdateActiveUsers(count float64) {
	registry := GetMetricsRegistry()
	registry.ActiveUsers.Set(count)
}

// GetMetricsCollector 获取指标收集器
func GetMetricsCollector() MetricsCollector {
	return &defaultMetricsCollector{}
}

// defaultMetricsCollector 默认指标收集器实现
type defaultMetricsCollector struct{}

func (c *defaultMetricsCollector) RecordHTTPRequest(method, endpoint, statusCode, tenantID string, duration time.Duration, requestSize, responseSize float64) {
	RecordHTTPRequest(method, endpoint, statusCode, tenantID, duration, requestSize, responseSize)
}

func (c *defaultMetricsCollector) RecordVoiceProcessing(processType, model, language string, duration time.Duration) {
	RecordVoiceProcessing(processType, model, language, duration)
}

func (c *defaultMetricsCollector) RecordASRRequest(model, language, status string, latency time.Duration) {
	RecordASRRequest(model, language, status, latency)
}

func (c *defaultMetricsCollector) RecordTTSRequest(voice, language, status string, latency time.Duration) {
	RecordTTSRequest(voice, language, status, latency)
}

func (c *defaultMetricsCollector) RecordLLMRequest(model, provider, status string, latency time.Duration, inputTokens, outputTokens int) {
	RecordLLMRequest(model, provider, status, latency, inputTokens, outputTokens)
}

func (c *defaultMetricsCollector) IncrementWSConnections(connectionType, tenantID string) {
	IncrementWSConnections(connectionType, tenantID)
}

func (c *defaultMetricsCollector) DecrementWSConnections() {
	DecrementWSConnections()
}

func (c *defaultMetricsCollector) UpdateActiveUsers(count float64) {
	UpdateActiveUsers(count)
}

// HTTP 指标方法实现
func (c *defaultMetricsCollector) IncHTTPInFlight() {
	registry := GetMetricsRegistry()
	registry.HTTPRequestsInFlight.Inc()
}

func (c *defaultMetricsCollector) DecHTTPInFlight() {
	registry := GetMetricsRegistry()
	registry.HTTPRequestsInFlight.Dec()
}

func (c *defaultMetricsCollector) IncHTTPRequests(method, endpoint, statusCode, tenantID string) {
	registry := GetMetricsRegistry()
	registry.HTTPRequestsTotal.WithLabelValues(method, endpoint, statusCode, tenantID).Inc()
}

func (c *defaultMetricsCollector) ObserveHTTPDuration(method, endpoint, tenantID string, duration time.Duration) {
	registry := GetMetricsRegistry()
	registry.HTTPRequestDuration.WithLabelValues(method, endpoint, tenantID).Observe(duration.Seconds())
}

func (c *defaultMetricsCollector) IncErrors(errorType, tenantID string) {
	registry := GetMetricsRegistry()
	registry.ErrorsTotal.WithLabelValues(errorType, tenantID).Inc()
}

// 业务指标方法实现
func (c *defaultMetricsCollector) IncConversations(conversationType, tenantID string) {
	registry := GetMetricsRegistry()
	registry.TotalConversations.WithLabelValues(conversationType, tenantID).Inc()
}

func (c *defaultMetricsCollector) IncAgentStreamConnections(tenantID string) {
	registry := GetMetricsRegistry()
	registry.WSConnectionsActive.Inc()
	registry.WSConnectionsTotal.WithLabelValues("agent_stream", tenantID).Inc()
}

func (c *defaultMetricsCollector) DecAgentStreamConnections() {
	registry := GetMetricsRegistry()
	registry.WSConnectionsActive.Dec()
}

func (c *defaultMetricsCollector) IncDocumentSearches(searchType, tenantID string) {
	registry := GetMetricsRegistry()
	registry.RAGQueryTotal.WithLabelValues(searchType, tenantID, "success").Inc()
}

func (c *defaultMetricsCollector) ObserveDocumentSearchDuration(searchType, tenantID string, duration time.Duration) {
	registry := GetMetricsRegistry()
	registry.RAGLatency.WithLabelValues(searchType, tenantID).Observe(duration.Seconds())
}

// WebSocket 指标方法实现
func (c *defaultMetricsCollector) IncWebSocketConnections(connectionType, tenantID string) {
	registry := GetMetricsRegistry()
	registry.WSConnectionsActive.Inc()
	registry.WSConnectionsTotal.WithLabelValues(connectionType, tenantID).Inc()
}

func (c *defaultMetricsCollector) DecWebSocketConnections() {
	registry := GetMetricsRegistry()
	registry.WSConnectionsActive.Dec()
}

// Voice 指标方法实现
func (c *defaultMetricsCollector) IncVoiceSessions(sessionType, tenantID string) {
	registry := GetMetricsRegistry()
	registry.VoiceSessionsActive.Inc()
}

func (c *defaultMetricsCollector) ObserveVoiceSessionDuration(sessionType, tenantID string, duration time.Duration) {
	registry := GetMetricsRegistry()
	registry.VoiceProcessingDuration.WithLabelValues("session", sessionType, tenantID).Observe(duration.Seconds())
}

// 系统指标方法实现
func (c *defaultMetricsCollector) SetMemoryUsage(usage float64) {
	registry := GetMetricsRegistry()
	registry.MemoryUsage.Set(usage)
}

func (c *defaultMetricsCollector) SetCPUUsage(usage float64) {
	registry := GetMetricsRegistry()
	registry.CPUUsage.Set(usage)
}

func (c *defaultMetricsCollector) SetGoroutinesCount(count float64) {
	registry := GetMetricsRegistry()
	registry.GoroutinesCount.Set(count)
}
