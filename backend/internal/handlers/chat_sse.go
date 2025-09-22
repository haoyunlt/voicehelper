package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"voicehelper/backend/pkg/types"
)

// SSE Prometheus指标
var (
	sseActiveStreams = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "sse_active_streams",
		Help: "Number of active SSE streams",
	})

	sseEventssent = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "sse_events_sent_total",
		Help: "Total SSE events sent",
	}, []string{"event_type", "session_id"})

	sseErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "sse_errors_total",
		Help: "Total SSE errors",
	}, []string{"error_type", "session_id"})

	sseStreamDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "sse_stream_duration_seconds",
		Help:    "SSE stream duration",
		Buckets: []float64{1, 5, 10, 30, 60, 300, 600},
	}, []string{"session_id"})
)

// SSEStream SSE流管理
type SSEStream struct {
	ID           string
	Writer       http.ResponseWriter
	Flusher      http.Flusher
	Context      context.Context
	Cancel       context.CancelFunc
	EventQueue   chan *types.EventEnvelope
	LastActivity time.Time
	StartTime    time.Time
	EventsSent   int64
	BytesSent    int64
	mutex        sync.RWMutex
}

// ChatSSEHandler SSE聊天处理器
type ChatSSEHandler struct {
	streams    map[string]*SSEStream
	streamsMux sync.RWMutex
	eventBus   EventBus
	config     SSEConfig
}

// SSEConfig SSE配置
type SSEConfig struct {
	MaxStreams        int
	EventQueueSize    int
	KeepAliveInterval time.Duration
	StreamTimeout     time.Duration
	MaxEventSize      int
}

// NewChatSSEHandler 创建SSE处理器
func NewChatSSEHandler(eventBus EventBus, config SSEConfig) *ChatSSEHandler {
	if config.EventQueueSize == 0 {
		config.EventQueueSize = 100
	}
	if config.KeepAliveInterval == 0 {
		config.KeepAliveInterval = 30 * time.Second
	}
	if config.StreamTimeout == 0 {
		config.StreamTimeout = 10 * time.Minute
	}
	if config.MaxEventSize == 0 {
		config.MaxEventSize = 64 * 1024 // 64KB
	}

	handler := &ChatSSEHandler{
		streams:  make(map[string]*SSEStream),
		eventBus: eventBus,
		config:   config,
	}

	// 启动清理协程
	go handler.cleanupRoutine()

	return handler
}

// HandleStream 处理SSE流请求
func (h *ChatSSEHandler) HandleStream(w http.ResponseWriter, r *http.Request) {
	// 检查连接数限制
	if h.config.MaxStreams > 0 && len(h.streams) >= h.config.MaxStreams {
		http.Error(w, "Too many active streams", http.StatusTooManyRequests)
		return
	}

	// 设置SSE头部
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	// 创建流会话
	streamID := h.generateStreamID()
	ctx, cancel := context.WithTimeout(r.Context(), h.config.StreamTimeout)

	stream := &SSEStream{
		ID:           streamID,
		Writer:       w,
		Flusher:      flusher,
		Context:      ctx,
		Cancel:       cancel,
		EventQueue:   make(chan *types.EventEnvelope, h.config.EventQueueSize),
		LastActivity: time.Now(),
		StartTime:    time.Now(),
	}

	h.addStream(stream)
	sseActiveStreams.Inc()

	// 发送连接确认
	h.sendConnectedEvent(stream)

	// 启动协程
	go h.handleStreamEvents(stream)
	go h.keepAliveManager(stream)

	log.Printf("New SSE stream established: %s", streamID)

	// 等待流结束
	<-stream.Context.Done()

	// 清理
	h.removeStream(streamID)
	sseActiveStreams.Dec()

	// 记录流持续时间
	duration := time.Since(stream.StartTime).Seconds()
	sseStreamDuration.WithLabelValues(streamID).Observe(duration)

	log.Printf("SSE stream closed: %s, duration: %.2fs, events: %d",
		streamID, duration, stream.EventsSent)
}

// handleStreamEvents 处理流事件
func (h *ChatSSEHandler) handleStreamEvents(stream *SSEStream) {
	defer stream.Cancel()

	for {
		select {
		case <-stream.Context.Done():
			return

		case event := <-stream.EventQueue:
			if err := h.sendEvent(stream, event); err != nil {
				log.Printf("Failed to send event to stream %s: %v", stream.ID, err)
				sseErrors.WithLabelValues("send_error", stream.ID).Inc()
				return
			}

			stream.mutex.Lock()
			stream.EventsSent++
			stream.LastActivity = time.Now()
			stream.mutex.Unlock()

			sseEventssent.WithLabelValues(event.Type, stream.ID).Inc()
		}
	}
}

// keepAliveManager 保活管理
func (h *ChatSSEHandler) keepAliveManager(stream *SSEStream) {
	ticker := time.NewTicker(h.config.KeepAliveInterval)
	defer ticker.Stop()

	for {
		select {
		case <-stream.Context.Done():
			return

		case <-ticker.C:
			// 发送保活事件
			keepAliveEvent := types.NewEventEnvelope("keep_alive", map[string]interface{}{
				"timestamp": time.Now().UnixMilli(),
			}, stream.ID, h.generateTraceID())

			select {
			case stream.EventQueue <- keepAliveEvent:
			default:
				// 队列满，跳过保活
				log.Printf("Stream %s: keep-alive queue full", stream.ID)
			}
		}
	}
}

// SendToStream 发送事件到指定流
func (h *ChatSSEHandler) SendToStream(streamID string, event *types.EventEnvelope) error {
	h.streamsMux.RLock()
	stream, exists := h.streams[streamID]
	h.streamsMux.RUnlock()

	if !exists {
		return fmt.Errorf("stream not found: %s", streamID)
	}

	select {
	case stream.EventQueue <- event:
		return nil
	default:
		sseErrors.WithLabelValues("queue_full", streamID).Inc()
		return fmt.Errorf("event queue full for stream: %s", streamID)
	}
}

// BroadcastEvent 广播事件到所有流
func (h *ChatSSEHandler) BroadcastEvent(event *types.EventEnvelope) {
	h.streamsMux.RLock()
	streams := make([]*SSEStream, 0, len(h.streams))
	for _, stream := range h.streams {
		streams = append(streams, stream)
	}
	h.streamsMux.RUnlock()

	for _, stream := range streams {
		select {
		case stream.EventQueue <- event:
		default:
			log.Printf("Failed to broadcast to stream %s: queue full", stream.ID)
			sseErrors.WithLabelValues("broadcast_queue_full", stream.ID).Inc()
		}
	}
}

// sendEvent 发送单个事件
func (h *ChatSSEHandler) sendEvent(stream *SSEStream, event *types.EventEnvelope) error {
	// 序列化事件数据
	data, err := json.Marshal(event.Data)
	if err != nil {
		return fmt.Errorf("failed to marshal event data: %v", err)
	}

	// 检查事件大小
	if len(data) > h.config.MaxEventSize {
		return fmt.Errorf("event too large: %d bytes", len(data))
	}

	// 构造SSE格式
	var sseData string
	if event.Type != "" {
		sseData += fmt.Sprintf("event: %s\n", event.Type)
	}
	if event.Meta.TraceID != "" {
		sseData += fmt.Sprintf("id: %s\n", event.Meta.TraceID)
	}
	sseData += fmt.Sprintf("data: %s\n\n", string(data))

	// 发送数据
	if _, err := fmt.Fprint(stream.Writer, sseData); err != nil {
		return fmt.Errorf("failed to write SSE data: %v", err)
	}

	stream.Flusher.Flush()

	stream.mutex.Lock()
	stream.BytesSent += int64(len(sseData))
	stream.mutex.Unlock()

	return nil
}

// sendConnectedEvent 发送连接确认事件
func (h *ChatSSEHandler) sendConnectedEvent(stream *SSEStream) {
	event := types.NewEventEnvelope("connected", map[string]interface{}{
		"stream_id": stream.ID,
		"timestamp": time.Now().UnixMilli(),
		"config": map[string]interface{}{
			"keep_alive_interval": h.config.KeepAliveInterval.Milliseconds(),
			"max_event_size":      h.config.MaxEventSize,
		},
	}, stream.ID, h.generateTraceID())

	select {
	case stream.EventQueue <- event:
	default:
		log.Printf("Failed to send connected event to stream: %s", stream.ID)
	}
}

// ProcessChatRequest 处理聊天请求
func (h *ChatSSEHandler) ProcessChatRequest(w http.ResponseWriter, r *http.Request) {
	// 解析请求
	var chatRequest struct {
		Message     string                 `json:"message"`
		SessionID   string                 `json:"session_id"`
		StreamID    string                 `json:"stream_id"`
		Context     map[string]interface{} `json:"context"`
		Temperature float32                `json:"temperature"`
		MaxTokens   int                    `json:"max_tokens"`
	}

	if err := json.NewDecoder(r.Body).Decode(&chatRequest); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// 验证必填字段
	if chatRequest.Message == "" {
		http.Error(w, "Message is required", http.StatusBadRequest)
		return
	}

	if chatRequest.StreamID == "" {
		http.Error(w, "Stream ID is required", http.StatusBadRequest)
		return
	}

	// 检查流是否存在
	h.streamsMux.RLock()
	_, exists := h.streams[chatRequest.StreamID]
	h.streamsMux.RUnlock()

	if !exists {
		http.Error(w, "Stream not found", http.StatusNotFound)
		return
	}

	// 创建聊天事件
	chatEvent := types.NewEventEnvelope("chat_request", map[string]interface{}{
		"message":     chatRequest.Message,
		"session_id":  chatRequest.SessionID,
		"stream_id":   chatRequest.StreamID,
		"context":     chatRequest.Context,
		"temperature": chatRequest.Temperature,
		"max_tokens":  chatRequest.MaxTokens,
		"timestamp":   time.Now().UnixMilli(),
	}, chatRequest.SessionID, h.generateTraceID())

	// 发布到事件总线
	if err := h.eventBus.Publish("chat.request", chatEvent); err != nil {
		log.Printf("Failed to publish chat request: %v", err)
		http.Error(w, "Failed to process request", http.StatusInternalServerError)
		return
	}

	// 返回成功响应
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "accepted",
		"trace_id": chatEvent.Meta.TraceID,
	})
}

// 辅助方法

func (h *ChatSSEHandler) addStream(stream *SSEStream) {
	h.streamsMux.Lock()
	h.streams[stream.ID] = stream
	h.streamsMux.Unlock()
}

func (h *ChatSSEHandler) removeStream(streamID string) {
	h.streamsMux.Lock()
	delete(h.streams, streamID)
	h.streamsMux.Unlock()
}

func (h *ChatSSEHandler) generateStreamID() string {
	return fmt.Sprintf("sse_%d_%d", time.Now().UnixNano(), len(h.streams))
}

func (h *ChatSSEHandler) generateTraceID() string {
	return fmt.Sprintf("trace_%d", time.Now().UnixNano())
}

// cleanupRoutine 清理过期流
func (h *ChatSSEHandler) cleanupRoutine() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		now := time.Now()
		var expiredStreams []string

		h.streamsMux.RLock()
		for streamID, stream := range h.streams {
			stream.mutex.RLock()
			if now.Sub(stream.LastActivity) > h.config.StreamTimeout {
				expiredStreams = append(expiredStreams, streamID)
			}
			stream.mutex.RUnlock()
		}
		h.streamsMux.RUnlock()

		// 清理过期流
		for _, streamID := range expiredStreams {
			h.streamsMux.RLock()
			if stream, exists := h.streams[streamID]; exists {
				stream.Cancel()
				log.Printf("Cleaned up expired stream: %s", streamID)
			}
			h.streamsMux.RUnlock()
		}
	}
}

// GetStreamStats 获取流统计信息
func (h *ChatSSEHandler) GetStreamStats(streamID string) map[string]interface{} {
	h.streamsMux.RLock()
	stream, exists := h.streams[streamID]
	h.streamsMux.RUnlock()

	if !exists {
		return nil
	}

	stream.mutex.RLock()
	defer stream.mutex.RUnlock()

	return map[string]interface{}{
		"stream_id":     stream.ID,
		"start_time":    stream.StartTime,
		"last_activity": stream.LastActivity,
		"events_sent":   stream.EventsSent,
		"bytes_sent":    stream.BytesSent,
		"queue_size":    len(stream.EventQueue),
		"duration":      time.Since(stream.StartTime).Seconds(),
	}
}

// GetAllStreamsStats 获取所有流的统计信息
func (h *ChatSSEHandler) GetAllStreamsStats() map[string]interface{} {
	h.streamsMux.RLock()
	defer h.streamsMux.RUnlock()

	stats := map[string]interface{}{
		"total_streams": len(h.streams),
		"streams":       make([]map[string]interface{}, 0, len(h.streams)),
	}

	for _, stream := range h.streams {
		streamStats := h.GetStreamStats(stream.ID)
		if streamStats != nil {
			stats["streams"] = append(stats["streams"].([]map[string]interface{}), streamStats)
		}
	}

	return stats
}
