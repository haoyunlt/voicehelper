package handlers

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"voicehelper/backend/pkg/types"
)

// Prometheus指标
var (
		Name: "ws_active_connections",
		Help: "Number of active WebSocket connections",
	})

		Name: "audio_frames_received_total",
		Help: "Total audio frames received",
	}, []string{"session_id"})

		Name: "audio_frames_sent_total",
		Help: "Total audio frames sent",
	}, []string{"session_id"})

		Name:    "ws_message_latency_seconds",
		Help:    "WebSocket message processing latency",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
	}, []string{"message_type", "session_id"})

		Name: "ws_errors_total",
		Help: "Total WebSocket errors",
	}, []string{"error_type", "session_id"})

		Name: "heartbeats_sent_total",
		Help: "Total heartbeats sent",
	}, []string{"session_id"})

		Name: "throttle_events_total",
		Help: "Total throttle events",
	}, []string{"reason", "session_id"})
)

// WebSocket升级器配置
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // 生产环境需要严格检查
	},
	ReadBufferSize:    4096,
	WriteBufferSize:   4096,
	EnableCompression: false, // 音频数据不压缩
}

// Session 会话管理
type Session struct {
	ID           string
	Conn         *websocket.Conn
	SendQueue    chan []byte
	Context      context.Context
	Cancel       context.CancelFunc
	LastActivity time.Time
	SequenceNum  int32

	// 背压控制
	QueueSize     int32
	ThrottleLimit int32
	SendInterval  time.Duration

	// 心跳管理
	HeartbeatInterval time.Duration
	HeartbeatTimeout  time.Duration
	LastHeartbeat     time.Time
	MissedHeartbeats  int32

	// 音频帧管理
	ExpectedSequence int32
	OutOfOrderFrames int64
	DroppedFrames    int64

	// 统计信息
	BytesReceived  int64
	BytesSent      int64
	FramesReceived int64
	FramesSent     int64

	mutex sync.RWMutex
}

// VoiceWSHandler WebSocket语音处理器
type VoiceWSHandler struct {
	sessions    map[string]*Session
	sessionsMux sync.RWMutex
	eventBus    EventBus
	config      WSConfig
}

// WSConfig WebSocket配置
type WSConfig struct {
	MaxConnections    int
	SendQueueSize     int
	HeartbeatInterval time.Duration
	HeartbeatTimeout  time.Duration
	ThrottleLimit     int32
	MaxFrameSize      int
	EnableCompression bool
}

// NewVoiceWSHandler 创建WebSocket处理器
func NewVoiceWSHandler(eventBus EventBus, config WSConfig) *VoiceWSHandler {
	if config.SendQueueSize == 0 {
		config.SendQueueSize = 100
	}
	if config.HeartbeatInterval == 0 {
		config.HeartbeatInterval = 30 * time.Second
	}
	if config.HeartbeatTimeout == 0 {
		config.HeartbeatTimeout = 60 * time.Second
	}
	if config.ThrottleLimit == 0 {
		config.ThrottleLimit = 50
	}
	if config.MaxFrameSize == 0 {
		config.MaxFrameSize = 8192
	}

	handler := &VoiceWSHandler{
		sessions: make(map[string]*Session),
		eventBus: eventBus,
		config:   config,
	}

	// 启动清理协程
	go handler.cleanupRoutine()

	return handler
}

// HandleConnection 处理WebSocket连接
func (h *VoiceWSHandler) HandleConnection(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}

	sessionID := h.generateSessionID()
	ctx, cancel := context.WithCancel(context.Background())

	session := &Session{
		ID:                sessionID,
		Conn:              conn,
		SendQueue:         make(chan []byte, h.config.SendQueueSize),
		Context:           ctx,
		Cancel:            cancel,
		LastActivity:      time.Now(),
		SequenceNum:       0,
		QueueSize:         0,
		ThrottleLimit:     h.config.ThrottleLimit,
		SendInterval:      10 * time.Millisecond,
		HeartbeatInterval: h.config.HeartbeatInterval,
		HeartbeatTimeout:  h.config.HeartbeatTimeout,
		LastHeartbeat:     time.Now(),
		ExpectedSequence:  0,
	}

	h.addSession(session)
	wsActiveConnections.Inc()

	// 发送连接确认
	h.sendConnectedEvent(session)

	// 启动协程
	go h.handleIncoming(session)
	go h.handleOutgoing(session)
	go h.heartbeatManager(session)

	log.Printf("New WebSocket connection established: %s", sessionID)
}

// handleIncoming 处理接收消息
func (h *VoiceWSHandler) handleIncoming(session *Session) {
	defer func() {
		h.removeSession(session.ID)
		session.Cancel()
		session.Conn.Close()
		wsActiveConnections.Dec()
		log.Printf("WebSocket connection closed: %s", session.ID)
	}()

	session.Conn.SetReadLimit(int64(h.config.MaxFrameSize))
	session.Conn.SetReadDeadline(time.Now().Add(h.config.HeartbeatTimeout))
	session.Conn.SetPongHandler(func(string) error {
		session.mutex.Lock()
		session.LastHeartbeat = time.Now()
		session.MissedHeartbeats = 0
		session.mutex.Unlock()
		session.Conn.SetReadDeadline(time.Now().Add(h.config.HeartbeatTimeout))
		return nil
	})

	for {
		select {
		case <-session.Context.Done():
			return
		default:
			messageType, message, err := session.Conn.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("WebSocket error: %v", err)
					wsErrors.WithLabelValues("read_error", session.ID).Inc()
				}
				return
			}

			startTime := time.Now()
			session.LastActivity = startTime

			switch messageType {
			case websocket.BinaryMessage:
				h.handleBinaryMessage(session, message)
			case websocket.TextMessage:
				h.handleTextMessage(session, message)
			}

			// 记录处理延迟
			latency := time.Since(startTime).Seconds()
			wsMessageLatency.WithLabelValues("incoming", session.ID).Observe(latency)

			session.mutex.Lock()
			session.BytesReceived += int64(len(message))
			session.mutex.Unlock()
		}
	}
}

// handleBinaryMessage 处理二进制消息(音频帧)
func (h *VoiceWSHandler) handleBinaryMessage(session *Session, data []byte) {
	if len(data) < 20 { // 最小帧头大小
		wsErrors.WithLabelValues("invalid_frame", session.ID).Inc()
		return
	}

	// 解析音频帧头部
	header := h.parseAudioHeader(data[:20])
	audioData := data[20:]

	// 序列号检查
	session.mutex.Lock()
	if header.SequenceNum != session.ExpectedSequence {
		if header.SequenceNum < session.ExpectedSequence {
			// 重复帧，忽略
			session.mutex.Unlock()
			return
		} else {
			// 乱序或丢失帧
			session.OutOfOrderFrames++
			session.DroppedFrames += int64(header.SequenceNum - session.ExpectedSequence)
		}
	}
	session.ExpectedSequence = header.SequenceNum + 1
	session.FramesReceived++
	session.mutex.Unlock()

	// 背压检查
	if session.QueueSize > session.ThrottleLimit {
		h.sendThrottleEvent(session, "queue_full")
		return
	}

	// 创建音频帧事件
	frameData := &types.AudioFrameData{
		Audio:       audioData,
		Timestamp:   header.Timestamp,
		SequenceNum: header.SequenceNum,
		SampleRate:  header.SampleRate,
		Channels:    header.Channels,
		FrameSize:   header.FrameSize,
	}

	event := types.NewEventEnvelope(types.EventTypeAudioFrame, frameData, session.ID, h.generateTraceID())

	// 发布到事件总线
	if err := h.eventBus.Publish("audio.frame", event); err != nil {
		log.Printf("Failed to publish audio frame event: %v", err)
		wsErrors.WithLabelValues("publish_error", session.ID).Inc()
	}

	audioFramesReceived.WithLabelValues(session.ID).Inc()
}

// handleTextMessage 处理文本消息(控制命令)
func (h *VoiceWSHandler) handleTextMessage(session *Session, data []byte) {
	event, err := types.EventFromJSON(data)
	if err != nil {
		log.Printf("Failed to parse text message: %v", err)
		wsErrors.WithLabelValues("parse_error", session.ID).Inc()
		return
	}

	// 处理控制命令
	switch event.Type {
	case types.EventTypeHeartbeat:
		h.handleHeartbeat(session, event)
	case types.EventTypeCancel:
		h.handleCancel(session, event)
	default:
		// 转发到事件总线
		if err := h.eventBus.Publish("control."+event.Type, event); err != nil {
			log.Printf("Failed to publish control event: %v", err)
			wsErrors.WithLabelValues("publish_error", session.ID).Inc()
		}
	}
}

// handleOutgoing 处理发送消息
func (h *VoiceWSHandler) handleOutgoing(session *Session) {
	ticker := time.NewTicker(session.SendInterval)
	defer ticker.Stop()

	for {
		select {
		case <-session.Context.Done():
			return
		case message := <-session.SendQueue:
			if err := session.Conn.WriteMessage(websocket.BinaryMessage, message); err != nil {
				log.Printf("Failed to send message: %v", err)
				wsErrors.WithLabelValues("send_error", session.ID).Inc()
				return
			}

			session.mutex.Lock()
			session.BytesSent += int64(len(message))
			session.QueueSize--
			session.mutex.Unlock()

		case <-ticker.C:
			// 动态调整发送间隔
			session.mutex.Lock()
			queueSize := session.QueueSize
			session.mutex.Unlock()

			if queueSize > session.ThrottleLimit/2 {
				// 队列积压，减少发送间隔
				session.SendInterval = 5 * time.Millisecond
			} else {
				// 正常发送间隔
				session.SendInterval = 10 * time.Millisecond
			}
			ticker.Reset(session.SendInterval)
		}
	}
}

// heartbeatManager 心跳管理
func (h *VoiceWSHandler) heartbeatManager(session *Session) {
	ticker := time.NewTicker(session.HeartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-session.Context.Done():
			return
		case <-ticker.C:
			session.mutex.Lock()
			timeSinceLastHeartbeat := time.Since(session.LastHeartbeat)
			session.mutex.Unlock()

			if timeSinceLastHeartbeat > session.HeartbeatTimeout {
				session.mutex.Lock()
				session.MissedHeartbeats++
				missedCount := session.MissedHeartbeats
				session.mutex.Unlock()

				if missedCount >= 3 {
					log.Printf("Session %s: too many missed heartbeats, closing connection", session.ID)
					session.Cancel()
					return
				}
			}

			// 发送ping
			if err := session.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				log.Printf("Failed to send ping: %v", err)
				return
			}

			heartbeatsSent.WithLabelValues(session.ID).Inc()
		}
	}
}

// SendToSession 发送消息到指定会话
func (h *VoiceWSHandler) SendToSession(sessionID string, event *types.EventEnvelope) error {
	h.sessionsMux.RLock()
	session, exists := h.sessions[sessionID]
	h.sessionsMux.RUnlock()

	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	data, err := event.ToJSON()
	if err != nil {
		return fmt.Errorf("failed to serialize event: %v", err)
	}

	select {
	case session.SendQueue <- data:
		session.mutex.Lock()
		session.QueueSize++
		session.mutex.Unlock()
		return nil
	default:
		// 队列满，丢弃消息
		wsErrors.WithLabelValues("queue_full", sessionID).Inc()
		return fmt.Errorf("send queue full for session: %s", sessionID)
	}
}

// SendAudioFrame 发送音频帧
func (h *VoiceWSHandler) SendAudioFrame(sessionID string, frameData *types.TTSChunkData) error {
	h.sessionsMux.RLock()
	session, exists := h.sessions[sessionID]
	h.sessionsMux.RUnlock()

	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	// 构造二进制帧
	header := types.AudioHeader{
		SessionID:   sessionID,
		SequenceNum: session.SequenceNum,
		SampleRate:  16000,
		Channels:    1,
		FrameSize:   int16(len(frameData.Audio)),
		Timestamp:   frameData.Timestamp,
		Format:      frameData.Format,
	}

	headerBytes := h.serializeAudioHeader(header)
	frameBytes := append(headerBytes, frameData.Audio...)

	select {
	case session.SendQueue <- frameBytes:
		session.mutex.Lock()
		session.QueueSize++
		session.SequenceNum++
		session.FramesSent++
		session.mutex.Unlock()

		audioFramesSent.WithLabelValues(sessionID).Inc()
		return nil
	default:
		wsErrors.WithLabelValues("queue_full", sessionID).Inc()
		return fmt.Errorf("send queue full for session: %s", sessionID)
	}
}

// 辅助方法

func (h *VoiceWSHandler) parseAudioHeader(data []byte) types.AudioHeader {
	return types.AudioHeader{
		SequenceNum: int32(binary.LittleEndian.Uint32(data[0:4])),
		SampleRate:  int32(binary.LittleEndian.Uint32(data[4:8])),
		Channels:    int8(data[8]),
		FrameSize:   int16(binary.LittleEndian.Uint16(data[9:11])),
		Timestamp:   int64(binary.LittleEndian.Uint64(data[12:20])),
	}
}

func (h *VoiceWSHandler) serializeAudioHeader(header types.AudioHeader) []byte {
	data := make([]byte, 20)
	binary.LittleEndian.PutUint32(data[0:4], uint32(header.SequenceNum))
	binary.LittleEndian.PutUint32(data[4:8], uint32(header.SampleRate))
	data[8] = byte(header.Channels)
	binary.LittleEndian.PutUint16(data[9:11], uint16(header.FrameSize))
	binary.LittleEndian.PutUint64(data[12:20], uint64(header.Timestamp))
	return data
}

func (h *VoiceWSHandler) sendConnectedEvent(session *Session) {
	event := types.NewEventEnvelope(types.EventTypeConnected, map[string]interface{}{
		"session_id": session.ID,
		"timestamp":  time.Now().UnixMilli(),
	}, session.ID, h.generateTraceID())

	if data, err := event.ToJSON(); err == nil {
		select {
		case session.SendQueue <- data:
			session.mutex.Lock()
			session.QueueSize++
			session.mutex.Unlock()
		default:
			log.Printf("Failed to send connected event to session: %s", session.ID)
		}
	}
}

func (h *VoiceWSHandler) sendThrottleEvent(session *Session, reason string) {
	throttleData := &types.ThrottleData{
		Reason:       reason,
		RetryAfterMs: 100,
		CurrentRate:  session.QueueSize,
		MaxRate:      session.ThrottleLimit,
	}

	event := types.NewEventEnvelope(types.EventTypeThrottle, throttleData, session.ID, h.generateTraceID())

	if data, err := event.ToJSON(); err == nil {
		select {
		case session.SendQueue <- data:
			session.mutex.Lock()
			session.QueueSize++
			session.mutex.Unlock()
		default:
			// 无法发送限流事件，直接关闭连接
			session.Cancel()
		}
	}

	throttleEvents.WithLabelValues(reason, session.ID).Inc()
}

func (h *VoiceWSHandler) handleHeartbeat(session *Session, event *types.EventEnvelope) {
	// 更新心跳时间
	session.mutex.Lock()
	session.LastHeartbeat = time.Now()
	session.MissedHeartbeats = 0
	session.mutex.Unlock()

	// 回复心跳
	response := types.NewEventEnvelope(types.EventTypeHeartbeat, &types.HeartbeatData{
		Timestamp: time.Now().UnixMilli(),
		Sequence:  session.SequenceNum,
	}, session.ID, event.Meta.TraceID)

	if data, err := response.ToJSON(); err == nil {
		select {
		case session.SendQueue <- data:
			session.mutex.Lock()
			session.QueueSize++
			session.mutex.Unlock()
		default:
			log.Printf("Failed to send heartbeat response to session: %s", session.ID)
		}
	}
}

func (h *VoiceWSHandler) handleCancel(session *Session, event *types.EventEnvelope) {
	// 转发取消事件到事件总线
	if err := h.eventBus.Publish("control.cancel", event); err != nil {
		log.Printf("Failed to publish cancel event: %v", err)
	}

	log.Printf("Cancel event received from session: %s", session.ID)
}

func (h *VoiceWSHandler) addSession(session *Session) {
	h.sessionsMux.Lock()
	h.sessions[session.ID] = session
	h.sessionsMux.Unlock()
}

func (h *VoiceWSHandler) removeSession(sessionID string) {
	h.sessionsMux.Lock()
	delete(h.sessions, sessionID)
	h.sessionsMux.Unlock()
}

func (h *VoiceWSHandler) generateSessionID() string {
	return fmt.Sprintf("ws_%d_%d", time.Now().UnixNano(), len(h.sessions))
}

func (h *VoiceWSHandler) generateTraceID() string {
	return fmt.Sprintf("trace_%d", time.Now().UnixNano())
}

// cleanupRoutine 清理过期会话
func (h *VoiceWSHandler) cleanupRoutine() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		now := time.Now()
		var expiredSessions []string

		h.sessionsMux.RLock()
		for sessionID, session := range h.sessions {
			session.mutex.RLock()
			if now.Sub(session.LastActivity) > 5*time.Minute {
				expiredSessions = append(expiredSessions, sessionID)
			}
			session.mutex.RUnlock()
		}
		h.sessionsMux.RUnlock()

		// 清理过期会话
		for _, sessionID := range expiredSessions {
			h.sessionsMux.RLock()
			if session, exists := h.sessions[sessionID]; exists {
				session.Cancel()
				log.Printf("Cleaned up expired session: %s", sessionID)
			}
			h.sessionsMux.RUnlock()
		}
	}
}

// GetSessionStats 获取会话统计信息
func (h *VoiceWSHandler) GetSessionStats(sessionID string) map[string]interface{} {
	h.sessionsMux.RLock()
	session, exists := h.sessions[sessionID]
	h.sessionsMux.RUnlock()

	if !exists {
		return nil
	}

	session.mutex.RLock()
	defer session.mutex.RUnlock()

	return map[string]interface{}{
		"session_id":          session.ID,
		"bytes_received":      session.BytesReceived,
		"bytes_sent":          session.BytesSent,
		"frames_received":     session.FramesReceived,
		"frames_sent":         session.FramesSent,
		"queue_size":          session.QueueSize,
		"out_of_order_frames": session.OutOfOrderFrames,
		"dropped_frames":      session.DroppedFrames,
		"last_activity":       session.LastActivity,
		"missed_heartbeats":   session.MissedHeartbeats,
	}
}
