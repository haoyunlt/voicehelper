package voice

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// VoiceMetricsCollector 语音指标收集器
type VoiceMetricsCollector struct {
	// 会话指标
	sessionsTotal   *prometheus.CounterVec
	sessionsActive  *prometheus.GaugeVec
	sessionDuration *prometheus.HistogramVec
	sessionErrors   *prometheus.CounterVec

	// 音频处理指标
	audioChunksTotal    *prometheus.CounterVec
	audioProcessingTime *prometheus.HistogramVec
	audioQuality        *prometheus.HistogramVec

	// ASR指标
	asrRequestsTotal *prometheus.CounterVec
	asrLatency       *prometheus.HistogramVec
	asrAccuracy      *prometheus.HistogramVec
	asrErrors        *prometheus.CounterVec

	// TTS指标
	ttsRequestsTotal *prometheus.CounterVec
	ttsLatency       *prometheus.HistogramVec
	ttsQuality       *prometheus.HistogramVec
	ttsErrors        *prometheus.CounterVec

	// 连接指标
	connectionsTotal   *prometheus.CounterVec
	connectionDuration *prometheus.HistogramVec
	connectionErrors   *prometheus.CounterVec

	// 资源使用指标
	memoryUsage      *prometheus.GaugeVec
	cpuUsage         *prometheus.GaugeVec
	networkBandwidth *prometheus.GaugeVec

	// 业务指标
	userSessions   *prometheus.GaugeVec
	tenantSessions *prometheus.GaugeVec
	quotaUsage     *prometheus.GaugeVec
}

// NewVoiceMetricsCollector 创建语音指标收集器
func NewVoiceMetricsCollector() *VoiceMetricsCollector {
	return &VoiceMetricsCollector{
		// 会话指标
		sessionsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_sessions_total",
				Help: "Total number of voice sessions",
			},
			[]string{"status", "tenant_id", "user_type"},
		),

		sessionsActive: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_voice_sessions_active",
				Help: "Number of active voice sessions",
			},
			[]string{"tenant_id"},
		),

		sessionDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_session_duration_seconds",
				Help:    "Voice session duration in seconds",
				Buckets: []float64{1, 5, 10, 30, 60, 300, 600, 1800, 3600},
			},
			[]string{"tenant_id", "session_type"},
		),

		sessionErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_session_errors_total",
				Help: "Total number of voice session errors",
			},
			[]string{"error_type", "tenant_id"},
		),

		// 音频处理指标
		audioChunksTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_audio_chunks_total",
				Help: "Total number of audio chunks processed",
			},
			[]string{"direction", "format", "tenant_id"},
		),

		audioProcessingTime: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_audio_processing_seconds",
				Help:    "Audio processing time in seconds",
				Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5},
			},
			[]string{"operation", "tenant_id"},
		),

		audioQuality: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_audio_quality_score",
				Help:    "Audio quality score (0-1)",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"type", "tenant_id"},
		),

		// ASR指标
		asrRequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_asr_requests_total",
				Help: "Total number of ASR requests",
			},
			[]string{"language", "model", "status", "tenant_id"},
		),

		asrLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_asr_latency_seconds",
				Help:    "ASR processing latency in seconds",
				Buckets: []float64{0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10},
			},
			[]string{"language", "model", "tenant_id"},
		),

		asrAccuracy: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_asr_accuracy_score",
				Help:    "ASR accuracy score (0-1)",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"language", "model", "tenant_id"},
		),

		asrErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_asr_errors_total",
				Help: "Total number of ASR errors",
			},
			[]string{"error_type", "language", "tenant_id"},
		),

		// TTS指标
		ttsRequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_tts_requests_total",
				Help: "Total number of TTS requests",
			},
			[]string{"voice", "language", "status", "tenant_id"},
		),

		ttsLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_tts_latency_seconds",
				Help:    "TTS processing latency in seconds",
				Buckets: []float64{0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10},
			},
			[]string{"voice", "language", "tenant_id"},
		),

		ttsQuality: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_tts_quality_score",
				Help:    "TTS quality score (0-1)",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"voice", "language", "tenant_id"},
		),

		ttsErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_tts_errors_total",
				Help: "Total number of TTS errors",
			},
			[]string{"error_type", "voice", "tenant_id"},
		),

		// 连接指标
		connectionsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_connections_total",
				Help: "Total number of voice connections",
			},
			[]string{"type", "status", "tenant_id"},
		),

		connectionDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_voice_connection_duration_seconds",
				Help:    "Voice connection duration in seconds",
				Buckets: []float64{1, 5, 10, 30, 60, 300, 600, 1800, 3600},
			},
			[]string{"type", "tenant_id"},
		),

		connectionErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_voice_connection_errors_total",
				Help: "Total number of voice connection errors",
			},
			[]string{"error_type", "tenant_id"},
		),

		// 资源使用指标
		memoryUsage: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_voice_memory_usage_bytes",
				Help: "Voice service memory usage in bytes",
			},
			[]string{"type"},
		),

		cpuUsage: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_voice_cpu_usage_percent",
				Help: "Voice service CPU usage percentage",
			},
			[]string{"type"},
		),

		networkBandwidth: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_voice_network_bandwidth_bytes_per_second",
				Help: "Voice service network bandwidth in bytes per second",
			},
			[]string{"direction", "type"},
		),

		// 业务指标
		userSessions: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_voice_user_sessions",
				Help: "Number of voice sessions per user",
			},
			[]string{"user_id", "tenant_id"},
		),

		tenantSessions: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_voice_tenant_sessions",
				Help: "Number of voice sessions per tenant",
			},
			[]string{"tenant_id"},
		),

		quotaUsage: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_voice_quota_usage_percent",
				Help: "Voice service quota usage percentage",
			},
			[]string{"quota_type", "user_id", "tenant_id"},
		),
	}
}

// 会话指标方法
func (m *VoiceMetricsCollector) IncSessionsTotal(status, tenantID, userType string) {
	m.sessionsTotal.WithLabelValues(status, tenantID, userType).Inc()
}

func (m *VoiceMetricsCollector) SetSessionsActive(tenantID string, count int) {
	m.sessionsActive.WithLabelValues(tenantID).Set(float64(count))
}

func (m *VoiceMetricsCollector) ObserveSessionDuration(tenantID, sessionType string, duration time.Duration) {
	m.sessionDuration.WithLabelValues(tenantID, sessionType).Observe(duration.Seconds())
}

func (m *VoiceMetricsCollector) IncSessionErrors(errorType, tenantID string) {
	m.sessionErrors.WithLabelValues(errorType, tenantID).Inc()
}

// 音频处理指标方法
func (m *VoiceMetricsCollector) IncAudioChunks(direction, format, tenantID string) {
	m.audioChunksTotal.WithLabelValues(direction, format, tenantID).Inc()
}

func (m *VoiceMetricsCollector) ObserveAudioProcessingTime(operation, tenantID string, duration time.Duration) {
	m.audioProcessingTime.WithLabelValues(operation, tenantID).Observe(duration.Seconds())
}

func (m *VoiceMetricsCollector) ObserveAudioQuality(audioType, tenantID string, quality float64) {
	m.audioQuality.WithLabelValues(audioType, tenantID).Observe(quality)
}

// ASR指标方法
func (m *VoiceMetricsCollector) IncASRRequests(language, model, status, tenantID string) {
	m.asrRequestsTotal.WithLabelValues(language, model, status, tenantID).Inc()
}

func (m *VoiceMetricsCollector) ObserveASRLatency(language, model, tenantID string, duration time.Duration) {
	m.asrLatency.WithLabelValues(language, model, tenantID).Observe(duration.Seconds())
}

func (m *VoiceMetricsCollector) ObserveASRAccuracy(language, model, tenantID string, accuracy float64) {
	m.asrAccuracy.WithLabelValues(language, model, tenantID).Observe(accuracy)
}

func (m *VoiceMetricsCollector) IncASRErrors(errorType, language, tenantID string) {
	m.asrErrors.WithLabelValues(errorType, language, tenantID).Inc()
}

// TTS指标方法
func (m *VoiceMetricsCollector) IncTTSRequests(voice, language, status, tenantID string) {
	m.ttsRequestsTotal.WithLabelValues(voice, language, status, tenantID).Inc()
}

func (m *VoiceMetricsCollector) ObserveTTSLatency(voice, language, tenantID string, duration time.Duration) {
	m.ttsLatency.WithLabelValues(voice, language, tenantID).Observe(duration.Seconds())
}

func (m *VoiceMetricsCollector) ObserveTTSQuality(voice, language, tenantID string, quality float64) {
	m.ttsQuality.WithLabelValues(voice, language, tenantID).Observe(quality)
}

func (m *VoiceMetricsCollector) IncTTSErrors(errorType, voice, tenantID string) {
	m.ttsErrors.WithLabelValues(errorType, voice, tenantID).Inc()
}

// 连接指标方法
func (m *VoiceMetricsCollector) IncConnections(connType, status, tenantID string) {
	m.connectionsTotal.WithLabelValues(connType, status, tenantID).Inc()
}

func (m *VoiceMetricsCollector) ObserveConnectionDuration(connType, tenantID string, duration time.Duration) {
	m.connectionDuration.WithLabelValues(connType, tenantID).Observe(duration.Seconds())
}

func (m *VoiceMetricsCollector) IncConnectionErrors(errorType, tenantID string) {
	m.connectionErrors.WithLabelValues(errorType, tenantID).Inc()
}

// 资源使用指标方法
func (m *VoiceMetricsCollector) SetMemoryUsage(memType string, bytes int64) {
	m.memoryUsage.WithLabelValues(memType).Set(float64(bytes))
}

func (m *VoiceMetricsCollector) SetCPUUsage(cpuType string, percent float64) {
	m.cpuUsage.WithLabelValues(cpuType).Set(percent)
}

func (m *VoiceMetricsCollector) SetNetworkBandwidth(direction, netType string, bytesPerSecond float64) {
	m.networkBandwidth.WithLabelValues(direction, netType).Set(bytesPerSecond)
}

// 业务指标方法
func (m *VoiceMetricsCollector) SetUserSessions(userID, tenantID string, count int) {
	m.userSessions.WithLabelValues(userID, tenantID).Set(float64(count))
}

func (m *VoiceMetricsCollector) SetTenantSessions(tenantID string, count int) {
	m.tenantSessions.WithLabelValues(tenantID).Set(float64(count))
}

func (m *VoiceMetricsCollector) SetQuotaUsage(quotaType, userID, tenantID string, percent float64) {
	m.quotaUsage.WithLabelValues(quotaType, userID, tenantID).Set(percent)
}

// 全局语音指标收集器实例
var (
	globalVoiceMetrics *VoiceMetricsCollector
	voiceMetricsOnce   sync.Once
)

// GetGlobalVoiceMetrics 获取全局语音指标收集器
func GetGlobalVoiceMetrics() *VoiceMetricsCollector {
	voiceMetricsOnce.Do(func() {
		globalVoiceMetrics = NewVoiceMetricsCollector()
	})
	return globalVoiceMetrics
}
