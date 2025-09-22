package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// 五段延迟监控指标
var (
	// 延迟直方图 - 按阶段分类
	VoiceLatencyHistogram = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "voice_latency_seconds",
			Help:    "Voice processing latency by stage",
			Buckets: []float64{0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
		},
		[]string{"stage", "session_id", "trace_id"},
	)

	// 端到端延迟
	E2ELatencyHistogram = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "voice_e2e_latency_seconds",
			Help:    "End-to-end voice processing latency",
			Buckets: []float64{0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0},
		},
		[]string{"session_id", "success"},
	)

	// 音频质量指标
	AudioHealthMetrics = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "audio_health_score",
			Help: "Audio quality health metrics",
		},
		[]string{"metric_type", "session_id"},
	)

	// 音频帧统计
	AudioFrameCounter = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "audio_frames_total",
			Help: "Total audio frames processed",
		},
		[]string{"direction", "session_id", "status"},
	)

	// 乱序帧计数
	OutOfOrderFrames = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "audio_ooo_frames_total",
			Help: "Total out-of-order audio frames",
		},
		[]string{"session_id"},
	)

	// 丢失帧计数
	DroppedFrames = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "audio_dropped_frames_total",
			Help: "Total dropped audio frames",
		},
		[]string{"session_id", "reason"},
	)

	// 抖动指标
	JitterHistogram = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "audio_jitter_seconds",
			Help:    "Audio jitter measurements",
			Buckets: []float64{0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2},
		},
		[]string{"session_id"},
	)

	// 打断事件
	BargeInCounter = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "barge_in_total",
			Help: "Total barge-in attempts and successes",
		},
		[]string{"result", "session_id"},
	)

	// 缓冲区健康度
	BufferHealthGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "buffer_health_score",
			Help: "Buffer health score (0-100)",
		},
		[]string{"buffer_type", "session_id"},
	)

	// VAD状态指标
	VADStateGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "vad_state",
			Help: "VAD state (0=silence, 1=speech)",
		},
		[]string{"session_id"},
	)

	// TTS生成指标
	TTSGenerationHistogram = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "tts_generation_seconds",
			Help:    "TTS generation time per sentence",
			Buckets: []float64{0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
		},
		[]string{"provider", "voice", "session_id"},
	)
)

// 延迟阶段常量
const (
	StageCAPTURE = "capture"
	StageASR     = "asr"
	StageLLM     = "llm"
	StageTTS     = "tts"
	StagePLAY    = "play"
	StageNETWORK = "network"
)

// LatencyTracker 延迟追踪器
type LatencyTracker struct {
	sessionID string
	traceID   string
	stages    map[string]time.Time
	mutex     sync.RWMutex
}

// NewLatencyTracker 创建延迟追踪器
func NewLatencyTracker(sessionID, traceID string) *LatencyTracker {
	return &LatencyTracker{
		sessionID: sessionID,
		traceID:   traceID,
		stages:    make(map[string]time.Time),
	}
}

// StartStage 开始阶段计时
func (lt *LatencyTracker) StartStage(stage string) {
	lt.mutex.Lock()
	defer lt.mutex.Unlock()
	lt.stages[stage] = time.Now()
}

// EndStage 结束阶段计时并记录指标
func (lt *LatencyTracker) EndStage(stage string) float64 {
	lt.mutex.Lock()
	defer lt.mutex.Unlock()

	startTime, exists := lt.stages[stage]
	if !exists {
		return 0
	}

	duration := time.Since(startTime).Seconds()
	VoiceLatencyHistogram.WithLabelValues(stage, lt.sessionID, lt.traceID).Observe(duration)

	delete(lt.stages, stage)
	return duration
}

// GetStageDuration 获取阶段持续时间（不记录指标）
func (lt *LatencyTracker) GetStageDuration(stage string) float64 {
	lt.mutex.RLock()
	defer lt.mutex.RUnlock()

	startTime, exists := lt.stages[stage]
	if !exists {
		return 0
	}

	return time.Since(startTime).Seconds()
}

// RecordE2ELatency 记录端到端延迟
func (lt *LatencyTracker) RecordE2ELatency(totalDuration float64, success bool) {
	successStr := "true"
	if !success {
		successStr = "false"
	}
	E2ELatencyHistogram.WithLabelValues(lt.sessionID, successStr).Observe(totalDuration)
}

// AudioQualityTracker 音频质量追踪器
type AudioQualityTracker struct {
	sessionID        string
	totalFrames      int64
	outOfOrderFrames int64
	droppedFrames    int64
	lastSequenceNum  int32
	jitterSum        float64
	jitterCount      int64
	lastFrameTime    time.Time
	mutex            sync.RWMutex
}

// NewAudioQualityTracker 创建音频质量追踪器
func NewAudioQualityTracker(sessionID string) *AudioQualityTracker {
	return &AudioQualityTracker{
		sessionID: sessionID,
	}
}

// RecordFrame 记录音频帧
func (aqt *AudioQualityTracker) RecordFrame(sequenceNum int32, direction string) {
	aqt.mutex.Lock()
	defer aqt.mutex.Unlock()

	aqt.totalFrames++
	now := time.Now()

	// 检查乱序
	if sequenceNum < aqt.lastSequenceNum {
		aqt.outOfOrderFrames++
		OutOfOrderFrames.WithLabelValues(aqt.sessionID).Inc()
	} else if sequenceNum > aqt.lastSequenceNum+1 {
		// 检查丢帧
		dropped := int64(sequenceNum - aqt.lastSequenceNum - 1)
		aqt.droppedFrames += dropped
		DroppedFrames.WithLabelValues(aqt.sessionID, "sequence_gap").Add(float64(dropped))
	}

	aqt.lastSequenceNum = sequenceNum

	// 计算抖动
	if !aqt.lastFrameTime.IsZero() {
		expectedInterval := 20 * time.Millisecond // 20ms per frame
		actualInterval := now.Sub(aqt.lastFrameTime)
		jitter := float64(abs(actualInterval-expectedInterval)) / float64(time.Second)

		aqt.jitterSum += jitter
		aqt.jitterCount++

		JitterHistogram.WithLabelValues(aqt.sessionID).Observe(jitter)
	}

	aqt.lastFrameTime = now

	// 记录帧计数
	AudioFrameCounter.WithLabelValues(direction, aqt.sessionID, "success").Inc()
}

// RecordDroppedFrame 记录丢失帧
func (aqt *AudioQualityTracker) RecordDroppedFrame(reason string) {
	aqt.mutex.Lock()
	defer aqt.mutex.Unlock()

	aqt.droppedFrames++
	DroppedFrames.WithLabelValues(aqt.sessionID, reason).Inc()
}

// GetStats 获取统计信息
func (aqt *AudioQualityTracker) GetStats() map[string]interface{} {
	aqt.mutex.RLock()
	defer aqt.mutex.RUnlock()

	avgJitter := float64(0)
	if aqt.jitterCount > 0 {
		avgJitter = aqt.jitterSum / float64(aqt.jitterCount)
	}

	packetLossRate := float64(0)
	if aqt.totalFrames > 0 {
		packetLossRate = float64(aqt.droppedFrames) / float64(aqt.totalFrames)
	}

	return map[string]interface{}{
		"total_frames":        aqt.totalFrames,
		"out_of_order_frames": aqt.outOfOrderFrames,
		"dropped_frames":      aqt.droppedFrames,
		"avg_jitter_ms":       avgJitter * 1000,
		"packet_loss_rate":    packetLossRate,
	}
}

// UpdateHealthMetrics 更新健康度指标
func (aqt *AudioQualityTracker) UpdateHealthMetrics() {
	stats := aqt.GetStats()

	// 计算健康度分数 (0-100)
	healthScore := 100.0

	// 丢包率影响 (丢包率 > 1% 开始扣分)
	if packetLoss, ok := stats["packet_loss_rate"].(float64); ok {
		if packetLoss > 0.01 {
			healthScore -= (packetLoss - 0.01) * 1000 // 每增加1%丢包率扣10分
		}
	}

	// 抖动影响 (抖动 > 10ms 开始扣分)
	if jitter, ok := stats["avg_jitter_ms"].(float64); ok {
		if jitter > 10 {
			healthScore -= (jitter - 10) * 2 // 每增加1ms抖动扣2分
		}
	}

	// 乱序率影响
	if totalFrames, ok := stats["total_frames"].(int64); ok && totalFrames > 0 {
		if oooFrames, ok := stats["out_of_order_frames"].(int64); ok {
			oooRate := float64(oooFrames) / float64(totalFrames)
			if oooRate > 0.005 { // 乱序率 > 0.5% 开始扣分
				healthScore -= (oooRate - 0.005) * 2000 // 每增加0.1%乱序率扣2分
			}
		}
	}

	// 确保分数在0-100范围内
	if healthScore < 0 {
		healthScore = 0
	}

	AudioHealthMetrics.WithLabelValues("overall", aqt.sessionID).Set(healthScore)
	AudioHealthMetrics.WithLabelValues("packet_loss", aqt.sessionID).Set(stats["packet_loss_rate"].(float64) * 100)
	AudioHealthMetrics.WithLabelValues("jitter", aqt.sessionID).Set(stats["avg_jitter_ms"].(float64))
}

// 便捷函数

// RecordLatency 记录延迟指标
func RecordLatency(stage, sessionID, traceID string, duration float64) {
	VoiceLatencyHistogram.WithLabelValues(stage, sessionID, traceID).Observe(duration)
}

// RecordBargeIn 记录打断事件
func RecordBargeIn(sessionID string, success bool) {
	result := "success"
	if !success {
		result = "failure"
	}
	BargeInCounter.WithLabelValues(result, sessionID).Inc()
}

// RecordBufferHealth 记录缓冲区健康度
func RecordBufferHealth(bufferType, sessionID string, healthScore float64) {
	BufferHealthGauge.WithLabelValues(bufferType, sessionID).Set(healthScore)
}

// RecordVADState 记录VAD状态
func RecordVADState(sessionID string, isSpeech bool) {
	value := 0.0
	if isSpeech {
		value = 1.0
	}
	VADStateGauge.WithLabelValues(sessionID).Set(value)
}

// RecordTTSGeneration 记录TTS生成时间
func RecordTTSGeneration(provider, voice, sessionID string, duration float64) {
	TTSGenerationHistogram.WithLabelValues(provider, voice, sessionID).Observe(duration)
}

// 辅助函数
func abs(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}

// VoiceMetricsCollector 语音指标收集器
type VoiceMetricsCollector struct {
	latencyTrackers map[string]*LatencyTracker
	qualityTrackers map[string]*AudioQualityTracker
	mutex           sync.RWMutex
}

// NewVoiceMetricsCollector 创建语音指标收集器
func NewVoiceMetricsCollector() *VoiceMetricsCollector {
	return &VoiceMetricsCollector{
		latencyTrackers: make(map[string]*LatencyTracker),
		qualityTrackers: make(map[string]*AudioQualityTracker),
	}
}

// GetLatencyTracker 获取延迟追踪器
func (mc *VoiceMetricsCollector) GetLatencyTracker(sessionID, traceID string) *LatencyTracker {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	key := sessionID + ":" + traceID
	if tracker, exists := mc.latencyTrackers[key]; exists {
		return tracker
	}

	tracker := NewLatencyTracker(sessionID, traceID)
	mc.latencyTrackers[key] = tracker
	return tracker
}

// GetQualityTracker 获取质量追踪器
func (mc *VoiceMetricsCollector) GetQualityTracker(sessionID string) *AudioQualityTracker {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if tracker, exists := mc.qualityTrackers[sessionID]; exists {
		return tracker
	}

	tracker := NewAudioQualityTracker(sessionID)
	mc.qualityTrackers[sessionID] = tracker
	return tracker
}

// CleanupSession 清理会话相关的追踪器
func (mc *VoiceMetricsCollector) CleanupSession(sessionID string) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	// 清理延迟追踪器
	for key := range mc.latencyTrackers {
		if len(key) > len(sessionID) && key[:len(sessionID)] == sessionID {
			delete(mc.latencyTrackers, key)
		}
	}

	// 清理质量追踪器
	delete(mc.qualityTrackers, sessionID)
}

// 全局指标收集器实例
var GlobalVoiceMetricsCollector = NewVoiceMetricsCollector()
