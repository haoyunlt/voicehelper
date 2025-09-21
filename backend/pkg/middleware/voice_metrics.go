package middleware

import (
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// VoiceMetrics 语音相关指标
type VoiceMetrics struct {
	// ASR 指标
	ASRLatency        time.Duration
	ASRAccuracy       float64
	ASRWordErrorRate  float64
	
	// TTS 指标
	TTSLatency        time.Duration
	TTSFirstAudio     time.Duration
	TTSInterruptTime  time.Duration
	
	// 对话指标
	EndToEndLatency   time.Duration
	BargeInSuccessRate float64
	CancelResponseTime time.Duration
	
	// 质量指标
	AudioQuality      float64
	NetworkLatency    time.Duration
}

// VoiceMetricsCollector 语音指标收集器
type VoiceMetricsCollector struct {
	metrics map[string]*VoiceMetrics
}

func NewVoiceMetricsCollector() *VoiceMetricsCollector {
	return &VoiceMetricsCollector{
		metrics: make(map[string]*VoiceMetrics),
	}
}

// VoiceMetricsMiddleware 语音指标中间件
func VoiceMetricsMiddleware(collector *VoiceMetricsCollector) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		// 处理请求
		c.Next()
		
		// 收集指标
		duration := time.Since(start)
		path := c.Request.URL.Path
		method := c.Request.Method
		status := c.Writer.Status()
		
		// 记录语音相关指标
		if isVoiceEndpoint(path) {
			collector.recordMetrics(c, duration, status)
		}
		
		// 记录日志
		logrus.WithFields(logrus.Fields{
			"method":   method,
			"path":     path,
			"status":   status,
			"duration": duration,
			"type":     "voice_request",
		}).Info("Voice request processed")
	}
}

func (c *VoiceMetricsCollector) recordMetrics(ctx *gin.Context, duration time.Duration, status int) {
	sessionID := ctx.GetHeader("X-Session-ID")
	if sessionID == "" {
		sessionID = "unknown"
	}
	
	if _, exists := c.metrics[sessionID]; !exists {
		c.metrics[sessionID] = &VoiceMetrics{}
	}
	
	metrics := c.metrics[sessionID]
	path := ctx.Request.URL.Path
	
	switch {
	case path == "/api/voice/stream":
		// WebSocket 连接指标
		metrics.NetworkLatency = duration
		
	case path == "/api/chat/cancel":
		// 取消请求响应时间
		metrics.CancelResponseTime = duration
		
		// 记录 Barge-in 成功率
		if status == 200 {
			// 成功取消
			logrus.WithFields(logrus.Fields{
				"session_id": sessionID,
				"cancel_time": duration,
				"status": "success",
			}).Info("Barge-in cancel successful")
		}
	}
	
	// 记录端到端延迟
	if endToEndStart := ctx.GetHeader("X-Start-Time"); endToEndStart != "" {
		if startTime, err := strconv.ParseInt(endToEndStart, 10, 64); err == nil {
			metrics.EndToEndLatency = time.Since(time.Unix(0, startTime*int64(time.Millisecond)))
		}
	}
}

// RecordASRMetrics 记录 ASR 指标
func (c *VoiceMetricsCollector) RecordASRMetrics(sessionID string, latency time.Duration, accuracy float64) {
	if _, exists := c.metrics[sessionID]; !exists {
		c.metrics[sessionID] = &VoiceMetrics{}
	}
	
	metrics := c.metrics[sessionID]
	metrics.ASRLatency = latency
	metrics.ASRAccuracy = accuracy
	
	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"asr_latency": latency,
		"asr_accuracy": accuracy,
	}).Info("ASR metrics recorded")
}

// RecordTTSMetrics 记录 TTS 指标
func (c *VoiceMetricsCollector) RecordTTSMetrics(sessionID string, latency, firstAudio time.Duration) {
	if _, exists := c.metrics[sessionID]; !exists {
		c.metrics[sessionID] = &VoiceMetrics{}
	}
	
	metrics := c.metrics[sessionID]
	metrics.TTSLatency = latency
	metrics.TTSFirstAudio = firstAudio
	
	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"tts_latency": latency,
		"tts_first_audio": firstAudio,
	}).Info("TTS metrics recorded")
}

// RecordBargeInMetrics 记录打断指标
func (c *VoiceMetricsCollector) RecordBargeInMetrics(sessionID string, interruptTime time.Duration, success bool) {
	if _, exists := c.metrics[sessionID]; !exists {
		c.metrics[sessionID] = &VoiceMetrics{}
	}
	
	metrics := c.metrics[sessionID]
	metrics.TTSInterruptTime = interruptTime
	
	if success {
		metrics.BargeInSuccessRate = 1.0
	} else {
		metrics.BargeInSuccessRate = 0.0
	}
	
	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"interrupt_time": interruptTime,
		"success": success,
	}).Info("Barge-in metrics recorded")
}

// GetMetrics 获取会话指标
func (c *VoiceMetricsCollector) GetMetrics(sessionID string) *VoiceMetrics {
	if metrics, exists := c.metrics[sessionID]; exists {
		return metrics
	}
	return nil
}

// GetAllMetrics 获取所有指标
func (c *VoiceMetricsCollector) GetAllMetrics() map[string]*VoiceMetrics {
	return c.metrics
}

// CleanupMetrics 清理过期指标
func (c *VoiceMetricsCollector) CleanupMetrics(maxAge time.Duration) {
	// 实现指标清理逻辑
	// 这里可以根据时间戳清理过期的指标数据
}

// ExportPrometheusMetrics 导出 Prometheus 格式指标
func (c *VoiceMetricsCollector) ExportPrometheusMetrics() string {
	// 实现 Prometheus 指标导出
	// 返回符合 Prometheus 格式的指标字符串
	return ""
}

func isVoiceEndpoint(path string) bool {
	voiceEndpoints := []string{
		"/api/voice/stream",
		"/api/chat/cancel",
	}
	
	for _, endpoint := range voiceEndpoints {
		if path == endpoint {
			return true
		}
	}
	return false
}
