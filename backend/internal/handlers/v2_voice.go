package handlers

import (
	"fmt"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

type V2VoiceHandler struct {
	BaseHandler
	upgrader websocket.Upgrader
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
			h.handleVoiceStart(handler, msg)

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

func (h *V2VoiceHandler) handleVoiceStart(handler *BaseStreamHandler, msg VoiceMessage) {
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

	// TODO: 建立到算法服务的WebSocket连接
	// 这里应该创建到算法服务的WebSocket连接并转发消息

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

	// TODO: 转发音频数据到算法服务
	// 这里应该将音频数据转发到算法服务的WebSocket连接

	logrus.WithFields(logrus.Fields{
		"trace_id":   handler.TraceID,
		"session_id": msg.SessionID,
		"data_size":  len(msg.Data),
	}).Debug("处理音频数据")

	// 模拟ASR结果（实际应该从算法服务获取）
	handler.WriteEnvelope("asr_partial", map[string]interface{}{
		"text":       "识别中...",
		"confidence": 0.8,
	})
}

func (h *V2VoiceHandler) handleVoiceStop(handler *BaseStreamHandler, msg VoiceMessage) {
	logrus.WithFields(logrus.Fields{
		"trace_id":   handler.TraceID,
		"session_id": msg.SessionID,
	}).Info("停止语音会话")

	// TODO: 关闭到算法服务的WebSocket连接

	// 发送会话停止确认
	handler.WriteEnvelope("session_stopped", map[string]interface{}{
		"session_id": msg.SessionID,
		"status":     "stopped",
	})
}
