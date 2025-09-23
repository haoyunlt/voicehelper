package handlers

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"

	"voicehelper/backend/common/logger"
)

type V2VoiceHandler struct {
	BaseHandler
	upgrader       websocket.Upgrader
	algoServiceURL string
	activeSessions map[string]*VoiceSession
	sessionsMutex  sync.RWMutex
}

// VoiceSession 语音会话
type VoiceSession struct {
	SessionID    string
	ClientConn   *websocket.Conn
	AlgoConn     *websocket.Conn
	Config       map[string]interface{}
	StartTime    time.Time
	LastActivity time.Time
	Context      context.Context
	Cancel       context.CancelFunc
	Mutex        sync.RWMutex
}

type VoiceMessage struct {
	Type      string                 `json:"type"`
	SessionID string                 `json:"session_id,omitempty"`
	Data      string                 `json:"data,omitempty"`
	Config    map[string]interface{} `json:"config,omitempty"`
}

// WSWriter WebSocket写入器
type WSWriter struct {
	conn   *websocket.Conn
	mutex  sync.Mutex
	closed bool
}

// NewWSWriter 创建WebSocket写入器
func NewWSWriter(conn *websocket.Conn) *WSWriter {
	return &WSWriter{
		conn: conn,
	}
}

// WriteEvent 写入事件
func (w *WSWriter) WriteEvent(event string, payload interface{}) error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.closed {
		return fmt.Errorf("writer closed")
	}

	message := map[string]interface{}{
		"event": event,
		"data":  payload,
	}

	return w.conn.WriteJSON(message)
}

// WriteError 写入错误
func (w *WSWriter) WriteError(code, message string) error {
	return w.WriteEvent("error", ErrorInfo{Code: code, Message: message})
}

// Close 关闭写入器
func (w *WSWriter) Close() error {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.closed {
		return nil
	}

	w.closed = true
	return w.conn.Close()
}

// ReadJSON 读取JSON消息
func (w *WSWriter) ReadJSON(v interface{}) error {
	return w.conn.ReadJSON(v)
}

// IsClosed 检查是否已关闭
func (w *WSWriter) IsClosed() bool {
	w.mutex.Lock()
	defer w.mutex.Unlock()
	return w.closed
}

func NewV2VoiceHandler(algoServiceURL string) *V2VoiceHandler {
	return &V2VoiceHandler{
		BaseHandler: BaseHandler{
			AlgoServiceURL: algoServiceURL,
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // 允许所有来源，生产环境应该限制
			},
		},
		algoServiceURL: algoServiceURL,
		activeSessions: make(map[string]*VoiceSession),
	}
}

func (h *V2VoiceHandler) HandleWebSocket(c *gin.Context) {
	// 升级到WebSocket连接
	conn, err := h.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logrus.WithError(err).Error("WebSocket升级失败")
		return
	}
	defer conn.Close()

	// 提取追踪信息
	traceID, tenantID := h.extractTraceInfo(c)

	// 创建WebSocket写入器
	wsWriter := NewWSWriter(conn)
	streamHandler := h.createStreamHandler(wsWriter, traceID, tenantID)

	logrus.WithFields(logrus.Fields{
		"trace_id":  traceID,
		"tenant_id": tenantID,
	}).Info("WebSocket语音连接建立")

	// 发送连接确认
	streamHandler.WriteEnvelope("connected", map[string]interface{}{
		"status": "connected",
	})

	// 处理消息循环
	h.handleMessageLoop(streamHandler, wsWriter)
}

func (h *V2VoiceHandler) handleMessageLoop(handler *BaseStreamHandler, wsWriter *WSWriter) {
	sessionID := ""

	for {
		if wsWriter.IsClosed() {
			break
		}

		var msg VoiceMessage
		err := wsWriter.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				logrus.WithError(err).Error("WebSocket读取错误")
			}
			break
		}

		logrus.WithFields(logrus.Fields{
			"trace_id":   handler.TraceID,
			"session_id": msg.SessionID,
			"type":       msg.Type,
		}).Debug("收到WebSocket消息")

		switch msg.Type {
		case "start":
			sessionID = msg.SessionID
			h.handleVoiceStart(handler, msg, wsWriter.conn)

		case "audio":
			if sessionID == "" {
				handler.WriteErrorEnvelope("no_session", "No active voice session")
				continue
			}
			h.handleVoiceAudio(handler, msg)

		case "stop":
			h.handleVoiceStop(handler, msg)
			sessionID = ""

		default:
			handler.WriteErrorEnvelope("unknown_message_type", "Unknown message type: "+msg.Type)
		}
	}

	logrus.WithFields(logrus.Fields{
		"trace_id":   handler.TraceID,
		"session_id": sessionID,
	}).Info("WebSocket语音连接关闭")
}

func (h *V2VoiceHandler) handleVoiceStart(handler *BaseStreamHandler, msg VoiceMessage, clientConn *websocket.Conn) {
	if msg.SessionID == "" {
		handler.WriteErrorEnvelope("missing_session_id", "Session ID is required")
		return
	}

	// 验证配置
	config := msg.Config
	if config == nil {
		config = map[string]interface{}{
			"sample_rate": 16000,
			"channels":    1,
			"language":    "zh-CN",
		}
	}

	logrus.WithFields(logrus.Fields{
		"trace_id":   handler.TraceID,
		"session_id": msg.SessionID,
		"config":     config,
	}).Info("开始语音会话")

	// 建立到算法服务的WebSocket连接
	algoConn, err := h.connectToAlgoService(msg.SessionID, config)
	if err != nil {
		logger.Error("连接算法服务失败", logger.Field{Key: "error", Value: err.Error()})
		handler.WriteErrorEnvelope("algo_connection_failed", fmt.Sprintf("Failed to connect to algorithm service: %v", err))
		return
	}

	// 创建语音会话
	ctx, cancel := context.WithCancel(context.Background())
	session := &VoiceSession{
		SessionID:    msg.SessionID,
		ClientConn:   clientConn,
		AlgoConn:     algoConn,
		Config:       config,
		StartTime:    time.Now(),
		LastActivity: time.Now(),
		Context:      ctx,
		Cancel:       cancel,
	}

	// 保存会话
	h.sessionsMutex.Lock()
	h.activeSessions[msg.SessionID] = session
	h.sessionsMutex.Unlock()

	// 启动算法服务消息转发
	go h.forwardAlgoMessages(session, handler)

	// 发送会话开始确认
	handler.WriteEnvelope("session_started", map[string]interface{}{
		"session_id": msg.SessionID,
		"config":     config,
		"status":     "started",
	})
}

func (h *V2VoiceHandler) handleVoiceAudio(handler *BaseStreamHandler, msg VoiceMessage) {
	if msg.Data == "" {
		handler.WriteErrorEnvelope("empty_audio_data", "Audio data is empty")
		return
	}

	// 获取会话
	h.sessionsMutex.RLock()
	session, exists := h.activeSessions[msg.SessionID]
	h.sessionsMutex.RUnlock()

	if !exists {
		handler.WriteErrorEnvelope("session_not_found", "Voice session not found")
		return
	}

	// 更新会话活动时间
	session.Mutex.Lock()
	session.LastActivity = time.Now()
	session.Mutex.Unlock()

	logrus.WithFields(logrus.Fields{
		"trace_id":   handler.TraceID,
		"session_id": msg.SessionID,
		"data_size":  len(msg.Data),
	}).Debug("处理音频数据")

	// 转发音频数据到算法服务
	if err := h.forwardAudioToAlgo(session, msg); err != nil {
		logger.Error("转发音频数据失败", logger.Field{Key: "error", Value: err.Error()})
		handler.WriteErrorEnvelope("audio_forward_failed", fmt.Sprintf("Failed to forward audio: %v", err))
		return
	}
}

func (h *V2VoiceHandler) handleVoiceStop(handler *BaseStreamHandler, msg VoiceMessage) {
	logrus.WithFields(logrus.Fields{
		"trace_id":   handler.TraceID,
		"session_id": msg.SessionID,
	}).Info("停止语音会话")

	// 获取并删除会话
	h.sessionsMutex.Lock()
	session, exists := h.activeSessions[msg.SessionID]
	if exists {
		delete(h.activeSessions, msg.SessionID)
	}
	h.sessionsMutex.Unlock()

	if exists {
		// 关闭算法服务连接
		h.closeVoiceSession(session)
	}

	// 发送会话停止确认
	handler.WriteEnvelope("session_stopped", map[string]interface{}{
		"session_id": msg.SessionID,
		"status":     "stopped",
	})
}

// connectToAlgoService 连接到算法服务
func (h *V2VoiceHandler) connectToAlgoService(sessionID string, config map[string]interface{}) (*websocket.Conn, error) {
	// 构建算法服务WebSocket URL
	u, err := url.Parse(h.algoServiceURL)
	if err != nil {
		return nil, fmt.Errorf("invalid algo service URL: %v", err)
	}

	// 设置WebSocket协议
	if u.Scheme == "http" {
		u.Scheme = "ws"
	} else if u.Scheme == "https" {
		u.Scheme = "wss"
	}
	u.Path = "/api/v1/voice/stream"

	// 添加查询参数
	q := u.Query()
	q.Set("session_id", sessionID)
	u.RawQuery = q.Encode()

	// 建立WebSocket连接
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial(u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to algo service: %v", err)
	}

	// 发送初始化消息
	initMsg := map[string]interface{}{
		"type":       "voice_start",
		"session_id": sessionID,
		"config":     config,
	}

	if err := conn.WriteJSON(initMsg); err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to send init message: %v", err)
	}

	return conn, nil
}

// forwardAudioToAlgo 转发音频数据到算法服务
func (h *V2VoiceHandler) forwardAudioToAlgo(session *VoiceSession, msg VoiceMessage) error {
	session.Mutex.RLock()
	algoConn := session.AlgoConn
	session.Mutex.RUnlock()

	if algoConn == nil {
		return fmt.Errorf("algorithm service connection is nil")
	}

	// 构建转发消息
	forwardMsg := map[string]interface{}{
		"type":       "voice_audio",
		"session_id": msg.SessionID,
		"data":       msg.Data,
		"timestamp":  time.Now().Unix(),
	}

	return algoConn.WriteJSON(forwardMsg)
}

// forwardAlgoMessages 转发算法服务消息到客户端
func (h *V2VoiceHandler) forwardAlgoMessages(session *VoiceSession, handler *BaseStreamHandler) {
	defer func() {
		if r := recover(); r != nil {
			logrus.WithFields(logrus.Fields{
				"session_id": session.SessionID,
				"panic":      r,
			}).Error("Panic in forwardAlgoMessages")
		}
	}()

	for {
		select {
		case <-session.Context.Done():
			return
		default:
			session.Mutex.RLock()
			algoConn := session.AlgoConn
			session.Mutex.RUnlock()

			if algoConn == nil {
				return
			}

			// 设置读取超时
			algoConn.SetReadDeadline(time.Now().Add(30 * time.Second))

			var message map[string]interface{}
			err := algoConn.ReadJSON(&message)
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					logrus.WithFields(logrus.Fields{
						"session_id": session.SessionID,
						"error":      err,
					}).Error("Algorithm service connection error")
				}
				return
			}

			// 转发消息到客户端
			if msgType, ok := message["type"].(string); ok {
				switch msgType {
				case "asr_partial", "asr_final", "tts_audio", "error":
					handler.WriteEnvelope(msgType, message)
				default:
					logrus.WithFields(logrus.Fields{
						"session_id": session.SessionID,
						"msg_type":   msgType,
					}).Debug("Unknown message type from algo service")
				}
			}
		}
	}
}

// closeVoiceSession 关闭语音会话
func (h *V2VoiceHandler) closeVoiceSession(session *VoiceSession) {
	// 取消上下文
	if session.Cancel != nil {
		session.Cancel()
	}

	// 关闭算法服务连接
	session.Mutex.Lock()
	if session.AlgoConn != nil {
		// 发送停止消息
		stopMsg := map[string]interface{}{
			"type":       "voice_stop",
			"session_id": session.SessionID,
		}
		session.AlgoConn.WriteJSON(stopMsg)
		session.AlgoConn.Close()
		session.AlgoConn = nil
	}
	session.Mutex.Unlock()

	logrus.WithFields(logrus.Fields{
		"session_id": session.SessionID,
		"duration":   time.Since(session.StartTime),
	}).Info("Voice session closed")
}

// cleanupInactiveSessions 清理不活跃的会话
func (h *V2VoiceHandler) cleanupInactiveSessions() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			h.sessionsMutex.Lock()
			now := time.Now()
			for sessionID, session := range h.activeSessions {
				session.Mutex.RLock()
				inactive := now.Sub(session.LastActivity) > 5*time.Minute
				session.Mutex.RUnlock()

				if inactive {
					logrus.WithField("session_id", sessionID).Info("Cleaning up inactive voice session")
					h.closeVoiceSession(session)
					delete(h.activeSessions, sessionID)
				}
			}
			h.sessionsMutex.Unlock()
		}
	}
}

// StartCleanupRoutine 启动清理例程
func (h *V2VoiceHandler) StartCleanupRoutine() {
	go h.cleanupInactiveSessions()
}
