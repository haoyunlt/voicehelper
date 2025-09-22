package handlers

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/pion/webrtc/v3"

	"voicehelper/backend/pkg/types"
)

// WebRTC Prometheus指标
var (
		Name: "webrtc_connections_active",
		Help: "Number of active WebRTC connections",
	})

		Name: "webrtc_connections_total",
		Help: "Total WebRTC connections",
	}, []string{"status"})

		Name: "webrtc_datachannel_messages_total",
		Help: "Total WebRTC data channel messages",
	}, []string{"direction", "session_id"})

		Name:    "webrtc_connection_duration_seconds",
		Help:    "WebRTC connection duration",
		Buckets: []float64{1, 5, 10, 30, 60, 300, 600, 1800},
	}, []string{"session_id"})

		Name:    "webrtc_signaling_latency_seconds",
		Help:    "WebRTC signaling latency",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
	}, []string{"message_type"})
)

// SignalingMessage 信令消息
type SignalingMessage struct {
	Type      string                 `json:"type"`
	SessionID string                 `json:"session_id"`
	Data      map[string]interface{} `json:"data"`
	Timestamp int64                  `json:"timestamp"`
}

// RTCSession WebRTC会话
type RTCSession struct {
	ID             string
	PeerConnection *webrtc.PeerConnection
	DataChannel    *webrtc.DataChannel
	SignalingConn  *websocket.Conn
	Context        context.Context
	Cancel         context.CancelFunc
	CreatedAt      time.Time
	LastActivity   time.Time

	// 状态
	ConnectionState webrtc.PeerConnectionState
	ICEState        webrtc.ICEConnectionState

	// 统计
	BytesSent        int64
	BytesReceived    int64
	MessagesSent     int64
	MessagesReceived int64

	// 音频处理
	AudioHandler func([]byte) error

	mutex sync.RWMutex
}

// WebRTCSignalingHandler WebRTC信令处理器
type WebRTCSignalingHandler struct {
	sessions    map[string]*RTCSession
	sessionsMux sync.RWMutex
	api         *webrtc.API
	config      WebRTCConfig
	eventBus    EventBus
}

// WebRTCConfig WebRTC配置
type WebRTCConfig struct {
	ICEServers        []webrtc.ICEServer
	DataChannelConfig *webrtc.DataChannelInit
	ConnectionTimeout time.Duration
	KeepAliveInterval time.Duration
	MaxSessions       int
	AudioCodec        string
	EnableTrickleICE  bool
}

// NewWebRTCSignalingHandler 创建WebRTC信令处理器
func NewWebRTCSignalingHandler(config WebRTCConfig, eventBus EventBus) *WebRTCSignalingHandler {
	// 默认配置
	if len(config.ICEServers) == 0 {
		config.ICEServers = []webrtc.ICEServer{
			{URLs: []string{"stun:stun.l.google.com:19302"}},
		}
	}

	if config.DataChannelConfig == nil {
		ordered := false
		maxRetransmits := uint16(0)
		config.DataChannelConfig = &webrtc.DataChannelInit{
			Ordered:        &ordered,
			MaxRetransmits: &maxRetransmits,
		}
	}

	if config.ConnectionTimeout == 0 {
		config.ConnectionTimeout = 30 * time.Second
	}

	if config.KeepAliveInterval == 0 {
		config.KeepAliveInterval = 30 * time.Second
	}

	if config.MaxSessions == 0 {
		config.MaxSessions = 1000
	}

	// 创建WebRTC API
	mediaEngine := &webrtc.MediaEngine{}
	if err := mediaEngine.RegisterDefaultCodecs(); err != nil {
		log.Fatalf("Failed to register WebRTC codecs: %v", err)
	}

	api := webrtc.NewAPI(webrtc.WithMediaEngine(mediaEngine))

	handler := &WebRTCSignalingHandler{
		sessions: make(map[string]*RTCSession),
		api:      api,
		config:   config,
		eventBus: eventBus,
	}

	// 启动清理协程
	go handler.cleanupRoutine()

	return handler
}

// HandleSignaling 处理信令连接
func (h *WebRTCSignalingHandler) HandleSignaling(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebRTC signaling upgrade failed: %v", err)
		return
	}

	sessionID := h.generateSessionID()
	log.Printf("New WebRTC signaling connection: %s", sessionID)

	// 处理信令消息
	go h.handleSignalingConnection(conn, sessionID)
}

// handleSignalingConnection 处理信令连接
func (h *WebRTCSignalingHandler) handleSignalingConnection(conn *websocket.Conn, sessionID string) {
	defer func() {
		conn.Close()
		h.removeSession(sessionID)
		log.Printf("WebRTC signaling connection closed: %s", sessionID)
	}()

	// 设置读取超时
	conn.SetReadDeadline(time.Now().Add(h.config.ConnectionTimeout))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(h.config.ConnectionTimeout))
		return nil
	})

	for {
		var msg SignalingMessage
		if err := conn.ReadJSON(&msg); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebRTC signaling error: %v", err)
			}
			break
		}

		msg.SessionID = sessionID
		msg.Timestamp = time.Now().UnixMilli()

		// 处理信令消息
		if err := h.handleSignalingMessage(conn, &msg); err != nil {
			log.Printf("Failed to handle signaling message: %v", err)

			// 发送错误响应
			errorMsg := SignalingMessage{
				Type:      "error",
				SessionID: sessionID,
				Data: map[string]interface{}{
					"error": err.Error(),
				},
				Timestamp: time.Now().UnixMilli(),
			}
			conn.WriteJSON(errorMsg)
		}
	}
}

// handleSignalingMessage 处理信令消息
func (h *WebRTCSignalingHandler) handleSignalingMessage(conn *websocket.Conn, msg *SignalingMessage) error {
	startTime := time.Now()
	defer func() {
		latency := time.Since(startTime).Seconds()
		webrtcSignalingLatency.WithLabelValues(msg.Type).Observe(latency)
	}()

	switch msg.Type {
	case "offer":
		return h.handleOffer(conn, msg)
	case "answer":
		return h.handleAnswer(conn, msg)
	case "ice-candidate":
		return h.handleICECandidate(conn, msg)
	case "ping":
		return h.handlePing(conn, msg)
	default:
		return fmt.Errorf("unknown signaling message type: %s", msg.Type)
	}
}

// handleOffer 处理Offer
func (h *WebRTCSignalingHandler) handleOffer(conn *websocket.Conn, msg *SignalingMessage) error {
	// 检查会话数限制
	if len(h.sessions) >= h.config.MaxSessions {
		return fmt.Errorf("maximum sessions reached")
	}

	// 解析SDP Offer
	offerSDP, ok := msg.Data["sdp"].(string)
	if !ok {
		return fmt.Errorf("invalid offer SDP")
	}

	offer := webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  offerSDP,
	}

	// 创建WebRTC会话
	session, err := h.createSession(msg.SessionID, conn)
	if err != nil {
		return fmt.Errorf("failed to create session: %v", err)
	}

	// 设置远端描述
	if err := session.PeerConnection.SetRemoteDescription(offer); err != nil {
		return fmt.Errorf("failed to set remote description: %v", err)
	}

	// 创建Answer
	answer, err := session.PeerConnection.CreateAnswer(nil)
	if err != nil {
		return fmt.Errorf("failed to create answer: %v", err)
	}

	// 设置本地描述
	if err := session.PeerConnection.SetLocalDescription(answer); err != nil {
		return fmt.Errorf("failed to set local description: %v", err)
	}

	// 发送Answer
	answerMsg := SignalingMessage{
		Type:      "answer",
		SessionID: msg.SessionID,
		Data: map[string]interface{}{
			"sdp": answer.SDP,
		},
		Timestamp: time.Now().UnixMilli(),
	}

	return conn.WriteJSON(answerMsg)
}

// handleAnswer 处理Answer
func (h *WebRTCSignalingHandler) handleAnswer(conn *websocket.Conn, msg *SignalingMessage) error {
	session := h.getSession(msg.SessionID)
	if session == nil {
		return fmt.Errorf("session not found: %s", msg.SessionID)
	}

	// 解析SDP Answer
	answerSDP, ok := msg.Data["sdp"].(string)
	if !ok {
		return fmt.Errorf("invalid answer SDP")
	}

	answer := webrtc.SessionDescription{
		Type: webrtc.SDPTypeAnswer,
		SDP:  answerSDP,
	}

	// 设置远端描述
	return session.PeerConnection.SetRemoteDescription(answer)
}

// handleICECandidate 处理ICE候选
func (h *WebRTCSignalingHandler) handleICECandidate(conn *websocket.Conn, msg *SignalingMessage) error {
	session := h.getSession(msg.SessionID)
	if session == nil {
		return fmt.Errorf("session not found: %s", msg.SessionID)
	}

	candidateData, ok := msg.Data["candidate"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid ICE candidate")
	}

	candidate := webrtc.ICECandidateInit{
		Candidate:     candidateData["candidate"].(string),
		SDPMid:        stringPtr(candidateData["sdpMid"].(string)),
		SDPMLineIndex: uint16Ptr(uint16(candidateData["sdpMLineIndex"].(float64))),
	}

	return session.PeerConnection.AddICECandidate(candidate)
}

// handlePing 处理Ping
func (h *WebRTCSignalingHandler) handlePing(conn *websocket.Conn, msg *SignalingMessage) error {
	pongMsg := SignalingMessage{
		Type:      "pong",
		SessionID: msg.SessionID,
		Data: map[string]interface{}{
			"timestamp": msg.Timestamp,
		},
		Timestamp: time.Now().UnixMilli(),
	}

	return conn.WriteJSON(pongMsg)
}

// createSession 创建WebRTC会话
func (h *WebRTCSignalingHandler) createSession(sessionID string, conn *websocket.Conn) (*RTCSession, error) {
	// 创建PeerConnection
	config := webrtc.Configuration{
		ICEServers: h.config.ICEServers,
	}

	peerConnection, err := h.api.NewPeerConnection(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create peer connection: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	session := &RTCSession{
		ID:              sessionID,
		PeerConnection:  peerConnection,
		SignalingConn:   conn,
		Context:         ctx,
		Cancel:          cancel,
		CreatedAt:       time.Now(),
		LastActivity:    time.Now(),
		ConnectionState: webrtc.PeerConnectionStateNew,
		ICEState:        webrtc.ICEConnectionStateNew,
	}

	// 设置事件处理器
	h.setupPeerConnectionHandlers(session)

	// 创建数据通道
	if err := h.createDataChannel(session); err != nil {
		peerConnection.Close()
		return nil, fmt.Errorf("failed to create data channel: %v", err)
	}

	// 添加到会话管理
	h.addSession(session)

	webrtcConnections.Inc()
	webrtcConnectionsTotal.WithLabelValues("created").Inc()

	return session, nil
}

// setupPeerConnectionHandlers 设置PeerConnection事件处理器
func (h *WebRTCSignalingHandler) setupPeerConnectionHandlers(session *RTCSession) {
	// 连接状态变化
	session.PeerConnection.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		session.mutex.Lock()
		session.ConnectionState = state
		session.mutex.Unlock()

		log.Printf("Session %s connection state: %s", session.ID, state)

		switch state {
		case webrtc.PeerConnectionStateConnected:
			webrtcConnectionsTotal.WithLabelValues("connected").Inc()

			// 发布连接事件
			event := types.NewEventEnvelope("webrtc_connected", map[string]interface{}{
				"session_id": session.ID,
				"timestamp":  time.Now().UnixMilli(),
			}, session.ID, h.generateTraceID())

			h.eventBus.Publish("webrtc.connected", event)

		case webrtc.PeerConnectionStateFailed:
		case webrtc.PeerConnectionStateClosed:
			webrtcConnectionsTotal.WithLabelValues("disconnected").Inc()
			h.removeSession(session.ID)
		}
	})

	// ICE连接状态变化
	session.PeerConnection.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		session.mutex.Lock()
		session.ICEState = state
		session.mutex.Unlock()

		log.Printf("Session %s ICE state: %s", session.ID, state)
	})

	// ICE候选
	session.PeerConnection.OnICECandidate(func(candidate *webrtc.ICECandidate) {
		if candidate == nil {
			return
		}

		// 发送ICE候选到客户端
		candidateMsg := SignalingMessage{
			Type:      "ice-candidate",
			SessionID: session.ID,
			Data: map[string]interface{}{
				"candidate": map[string]interface{}{
					"candidate":     candidate.String(),
					"sdpMid":        nil, // WebRTC v3 API change
					"sdpMLineIndex": 0,   // WebRTC v3 API change
				},
			},
			Timestamp: time.Now().UnixMilli(),
		}

		if err := session.SignalingConn.WriteJSON(candidateMsg); err != nil {
			log.Printf("Failed to send ICE candidate: %v", err)
		}
	})
}

// createDataChannel 创建数据通道
func (h *WebRTCSignalingHandler) createDataChannel(session *RTCSession) error {
	dataChannel, err := session.PeerConnection.CreateDataChannel("audio", h.config.DataChannelConfig)
	if err != nil {
		return err
	}

	session.DataChannel = dataChannel

	// 设置数据通道事件处理器
	dataChannel.OnOpen(func() {
		log.Printf("Session %s data channel opened", session.ID)
	})

	dataChannel.OnClose(func() {
		log.Printf("Session %s data channel closed", session.ID)
	})

	dataChannel.OnMessage(func(msg webrtc.DataChannelMessage) {
		session.mutex.Lock()
		session.BytesReceived += int64(len(msg.Data))
		session.MessagesReceived++
		session.LastActivity = time.Now()
		session.mutex.Unlock()

		webrtcDataChannelMessages.WithLabelValues("received", session.ID).Inc()

		// 处理音频数据
		if session.AudioHandler != nil {
			if err := session.AudioHandler(msg.Data); err != nil {
				log.Printf("Audio handler error: %v", err)
			}
		}

		// 发布音频帧事件
		event := types.NewEventEnvelope("webrtc_audio_frame", map[string]interface{}{
			"session_id": session.ID,
			"data_size":  len(msg.Data),
			"timestamp":  time.Now().UnixMilli(),
		}, session.ID, h.generateTraceID())

		h.eventBus.Publish("webrtc.audio_frame", event)
	})

	dataChannel.OnError(func(err error) {
		log.Printf("Session %s data channel error: %v", session.ID, err)
	})

	return nil
}

// SendAudioFrame 发送音频帧
func (h *WebRTCSignalingHandler) SendAudioFrame(sessionID string, audioData []byte) error {
	session := h.getSession(sessionID)
	if session == nil {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	if session.DataChannel == nil || session.DataChannel.ReadyState() != webrtc.DataChannelStateOpen {
		return fmt.Errorf("data channel not ready for session: %s", sessionID)
	}

	if err := session.DataChannel.Send(audioData); err != nil {
		return fmt.Errorf("failed to send audio data: %v", err)
	}

	session.mutex.Lock()
	session.BytesSent += int64(len(audioData))
	session.MessagesSent++
	session.LastActivity = time.Now()
	session.mutex.Unlock()

	webrtcDataChannelMessages.WithLabelValues("sent", sessionID).Inc()

	return nil
}

// 会话管理方法

func (h *WebRTCSignalingHandler) addSession(session *RTCSession) {
	h.sessionsMux.Lock()
	h.sessions[session.ID] = session
	h.sessionsMux.Unlock()
}

func (h *WebRTCSignalingHandler) getSession(sessionID string) *RTCSession {
	h.sessionsMux.RLock()
	defer h.sessionsMux.RUnlock()
	return h.sessions[sessionID]
}

func (h *WebRTCSignalingHandler) removeSession(sessionID string) {
	h.sessionsMux.Lock()
	defer h.sessionsMux.Unlock()

	if session, exists := h.sessions[sessionID]; exists {
		// 记录连接持续时间
		duration := time.Since(session.CreatedAt).Seconds()
		webrtcConnectionDuration.WithLabelValues(sessionID).Observe(duration)

		// 清理资源
		session.Cancel()
		if session.DataChannel != nil {
			session.DataChannel.Close()
		}
		if session.PeerConnection != nil {
			session.PeerConnection.Close()
		}

		delete(h.sessions, sessionID)
		webrtcConnections.Dec()

		log.Printf("Removed WebRTC session: %s, duration: %.2fs", sessionID, duration)
	}
}

// cleanupRoutine 清理过期会话
func (h *WebRTCSignalingHandler) cleanupRoutine() {
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
			h.removeSession(sessionID)
		}
	}
}

// GetSessionStats 获取会话统计信息
func (h *WebRTCSignalingHandler) GetSessionStats(sessionID string) map[string]interface{} {
	session := h.getSession(sessionID)
	if session == nil {
		return nil
	}

	session.mutex.RLock()
	defer session.mutex.RUnlock()

	return map[string]interface{}{
		"session_id":        session.ID,
		"created_at":        session.CreatedAt,
		"last_activity":     session.LastActivity,
		"connection_state":  session.ConnectionState.String(),
		"ice_state":         session.ICEState.String(),
		"bytes_sent":        session.BytesSent,
		"bytes_received":    session.BytesReceived,
		"messages_sent":     session.MessagesSent,
		"messages_received": session.MessagesReceived,
		"duration_seconds":  time.Since(session.CreatedAt).Seconds(),
	}
}

// GetAllStats 获取所有会话统计信息
func (h *WebRTCSignalingHandler) GetAllStats() map[string]interface{} {
	h.sessionsMux.RLock()
	defer h.sessionsMux.RUnlock()

	stats := map[string]interface{}{
		"total_sessions": len(h.sessions),
		"sessions":       make([]map[string]interface{}, 0, len(h.sessions)),
	}

	for _, session := range h.sessions {
		sessionStats := h.GetSessionStats(session.ID)
		if sessionStats != nil {
			stats["sessions"] = append(stats["sessions"].([]map[string]interface{}), sessionStats)
		}
	}

	return stats
}

// 辅助函数

func (h *WebRTCSignalingHandler) generateSessionID() string {
	return fmt.Sprintf("webrtc_%d_%d", time.Now().UnixNano(), len(h.sessions))
}

func (h *WebRTCSignalingHandler) generateTraceID() string {
	return fmt.Sprintf("trace_%d", time.Now().UnixNano())
}

func stringPtr(s string) *string {
	return &s
}

func uint16Ptr(u uint16) *uint16 {
	return &u
}
