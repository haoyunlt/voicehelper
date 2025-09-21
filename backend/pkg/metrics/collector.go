package metrics

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// MetricsCollector 指标收集器
type MetricsCollector struct {
	// HTTP指标
	httpRequestsTotal    *prometheus.CounterVec
	httpRequestDuration  *prometheus.HistogramVec
	httpRequestsInFlight *prometheus.GaugeVec

	// 业务指标
	conversationsTotal *prometheus.CounterVec
	messagesTotal      *prometheus.CounterVec
	userSessionsActive *prometheus.GaugeVec
	userFeedbackTotal  *prometheus.CounterVec

	// LLM指标
	llmRequestsTotal   *prometheus.CounterVec
	llmTokenUsageTotal *prometheus.CounterVec
	llmRequestDuration *prometheus.HistogramVec
	llmTokenQuotaTotal *prometheus.GaugeVec

	// 缓存指标
	cacheRequestsTotal *prometheus.CounterVec
	cacheHitsTotal     *prometheus.CounterVec
	cacheMemoryUsage   *prometheus.GaugeVec

	// 语音指标
	voiceRequestsTotal      *prometheus.CounterVec
	voiceProcessingDuration *prometheus.HistogramVec
	asrRequestsTotal        *prometheus.CounterVec
	asrErrorsTotal          *prometheus.CounterVec
	ttsRequestsTotal        *prometheus.CounterVec
	ttsErrorsTotal          *prometheus.CounterVec

	// 成本指标
	costUSDTotal *prometheus.CounterVec

	// 系统健康指标
	systemHealthScore *prometheus.GaugeVec
}

// NewMetricsCollector 创建指标收集器
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		// HTTP指标
		httpRequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "http_requests_total",
				Help: "Total number of HTTP requests",
			},
			[]string{"method", "endpoint", "status", "service"},
		),
		httpRequestDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "http_request_duration_seconds",
				Help:    "HTTP request duration in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method", "endpoint", "service"},
		),

		// 业务指标
		conversationsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "conversation_total",
				Help: "Total number of conversations",
			},
			[]string{"tenant_id", "status"},
		),
		messagesTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "messages_total",
				Help: "Total number of messages",
			},
			[]string{"tenant_id", "role", "modality"},
		),
		userSessionsActive: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "user_sessions_active",
				Help: "Number of active user sessions",
			},
			[]string{"tenant_id"},
		),

		// LLM指标
		llmRequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "llm_requests_total",
				Help: "Total LLM requests",
			},
			[]string{"model", "status"},
		),
		llmTokenUsageTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "llm_token_usage_total",
				Help: "Total LLM token usage",
			},
			[]string{"model", "type"},
		),
		llmTokenQuotaTotal: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_token_quota_total",
				Help: "Total LLM token quota",
			},
			[]string{"model"},
		),

		// 缓存指标
		cacheRequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cache_requests_total",
				Help: "Total cache requests",
			},
			[]string{"cache_type", "operation"},
		),
		cacheHitsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cache_hits_total",
				Help: "Total cache hits",
			},
			[]string{"cache_type"},
		),

		// 成本指标
		costUSDTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cost_usd_total",
				Help: "Total cost in USD",
			},
			[]string{"service", "resource_type"},
		),

		// 系统健康指标
		systemHealthScore: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "system_health_score",
				Help: "System health score (0-100)",
			},
			[]string{"service"},
		),
	}
}

// RecordHTTPRequest 记录HTTP请求指标
func (m *MetricsCollector) RecordHTTPRequest(method, endpoint, status, service string, duration time.Duration) {
	m.httpRequestsTotal.WithLabelValues(method, endpoint, status, service).Inc()
	m.httpRequestDuration.WithLabelValues(method, endpoint, service).Observe(duration.Seconds())
}

// RecordConversation 记录对话指标
func (m *MetricsCollector) RecordConversation(tenantID, status string) {
	m.conversationsTotal.WithLabelValues(tenantID, status).Inc()
}

// RecordMessage 记录消息指标
func (m *MetricsCollector) RecordMessage(tenantID, role, modality string) {
	m.messagesTotal.WithLabelValues(tenantID, role, modality).Inc()
}

// SetActiveUserSessions 设置活跃用户会话数
func (m *MetricsCollector) SetActiveUserSessions(tenantID string, count float64) {
	m.userSessionsActive.WithLabelValues(tenantID).Set(count)
}

// RecordLLMRequest 记录LLM请求指标
func (m *MetricsCollector) RecordLLMRequest(model, status string, duration time.Duration, inputTokens, outputTokens int) {
	m.llmRequestsTotal.WithLabelValues(model, status).Inc()
	m.llmTokenUsageTotal.WithLabelValues(model, "input").Add(float64(inputTokens))
	m.llmTokenUsageTotal.WithLabelValues(model, "output").Add(float64(outputTokens))
}

// SetLLMTokenQuota 设置LLM Token配额
func (m *MetricsCollector) SetLLMTokenQuota(model string, quota float64) {
	m.llmTokenQuotaTotal.WithLabelValues(model).Set(quota)
}

// RecordCacheOperation 记录缓存操作
func (m *MetricsCollector) RecordCacheOperation(cacheType, operation string, hit bool) {
	m.cacheRequestsTotal.WithLabelValues(cacheType, operation).Inc()
	if hit {
		m.cacheHitsTotal.WithLabelValues(cacheType).Inc()
	}
}

// RecordCost 记录成本
func (m *MetricsCollector) RecordCost(service, resourceType string, costUSD float64) {
	m.costUSDTotal.WithLabelValues(service, resourceType).Add(costUSD)
}

// SetSystemHealthScore 设置系统健康分数
func (m *MetricsCollector) SetSystemHealthScore(service string, score float64) {
	m.systemHealthScore.WithLabelValues(service).Set(score)
}

// StartHealthScoreCalculator 启动健康分数计算器
func (m *MetricsCollector) StartHealthScoreCalculator(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.calculateAndUpdateHealthScore()
		}
	}
}

// calculateAndUpdateHealthScore 计算并更新健康分数
func (m *MetricsCollector) calculateAndUpdateHealthScore() {
	// 示例：简单的健康分数计算
	baseScore := 100.0
	healthScore := baseScore * 0.95 // 95%健康度
	m.SetSystemHealthScore("chatbot", healthScore)
}
