package handler

import (
	"context"
	"net/http"
	"sync"
	"time"

	"chatbot/internal/service"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // 允许跨域，生产环境需要更严格的检查
	},
}

// VoiceMessage WebSocket 消息结构
type VoiceMessage struct {
	Type           string              `json:"type"`
	Seq            int                 `json:"seq,omitempty"`
	ConversationID string              `json:"conversation_id,omitempty"`
	Codec          string              `json:"codec,omitempty"`
	SampleRate     int                 `json:"sample_rate,omitempty"`
	Chunk          string              `json:"chunk,omitempty"`
	Text           string              `json:"text,omitempty"`
	PCM            string              `json:"pcm,omitempty"`
	Items          []service.Reference `json:"items,omitempty"`
	Error          string              `json:"error,omitempty"`
}

// VoiceSession 语音会话
type VoiceSession struct {
	ID             string
	ConversationID string
	Conn           *websocket.Conn
	Send           chan VoiceMessage
	Cancel         context.CancelFunc
	LastActivity   time.Time
	mu             sync.RWMutex
}

// VoiceHub 语音会话管理器
type VoiceHub struct {
	sessions map[string]*VoiceSession
	mu       sync.RWMutex
}

func NewVoiceHub() *VoiceHub {
	hub := &VoiceHub{
		sessions: make(map[string]*VoiceSession),
	}

	// 启动清理协程
	go hub.cleanup()

	return hub
}

func (h *VoiceHub) cleanup() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		h.mu.Lock()
		now := time.Now()
		for id, session := range h.sessions {
			session.mu.RLock()
			if now.Sub(session.LastActivity) > 5*time.Minute {
				session.mu.RUnlock()
				logrus.Infof("Cleaning up inactive voice session: %s", id)
				session.Cancel()
				delete(h.sessions, id)
			} else {
				session.mu.RUnlock()
			}
		}
		h.mu.Unlock()
	}
}

func (h *VoiceHub) AddSession(session *VoiceSession) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.sessions[session.ID] = session
}

func (h *VoiceHub) RemoveSession(sessionID string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if session, exists := h.sessions[sessionID]; exists {
		session.Cancel()
		delete(h.sessions, sessionID)
	}
}

func (h *VoiceHub) GetSession(sessionID string) (*VoiceSession, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	session, exists := h.sessions[sessionID]
	return session, exists
}

var voiceHub = NewVoiceHub()

// VoiceStream WebSocket 语音流处理
func (h *Handlers) VoiceStream(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logrus.WithError(err).Error("Failed to upgrade WebSocket connection")
		return
	}
	defer conn.Close()

	sessionID := c.GetHeader("X-Session-ID")
	if sessionID == "" {
		sessionID = generateSessionID()
	}

	ctx, cancel := context.WithCancel(c.Request.Context())
	session := &VoiceSession{
		ID:           sessionID,
		Conn:         conn,
		Send:         make(chan VoiceMessage, 256),
		Cancel:       cancel,
		LastActivity: time.Now(),
	}

	voiceHub.AddSession(session)
	defer voiceHub.RemoveSession(sessionID)

	// 启动发送协程
	go h.voiceWriter(session)

	// 处理接收消息
	h.voiceReader(ctx, session)
}

func (h *Handlers) voiceReader(ctx context.Context, session *VoiceSession) {
	defer close(session.Send)

	for {
		select {
		case <-ctx.Done():
			return
		default:
			var msg VoiceMessage
			err := session.Conn.ReadJSON(&msg)
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					logrus.WithError(err).Error("WebSocket read error")
				}
				return
			}

			session.mu.Lock()
			session.LastActivity = time.Now()
			session.mu.Unlock()

			// 处理消息
			h.handleVoiceMessage(ctx, session, msg)
		}
	}
}

func (h *Handlers) voiceWriter(session *VoiceSession) {
	ticker := time.NewTicker(54 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case message, ok := <-session.Send:
			session.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				session.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			if err := session.Conn.WriteJSON(message); err != nil {
				logrus.WithError(err).Error("WebSocket write error")
				return
			}

		case <-ticker.C:
			session.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := session.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (h *Handlers) handleVoiceMessage(ctx context.Context, session *VoiceSession, msg VoiceMessage) {
	switch msg.Type {
	case "start":
		session.ConversationID = msg.ConversationID
		logrus.Infof("Voice session started: %s, conversation: %s", session.ID, msg.ConversationID)

	case "audio":
		// 转发音频数据到算法服务
		h.forwardAudioToAlgo(ctx, session, msg)

	case "stop":
		logrus.Infof("Voice session stopped: %s", session.ID)
		session.Send <- VoiceMessage{Type: "done"}

	default:
		logrus.Warnf("Unknown voice message type: %s", msg.Type)
	}
}

func (h *Handlers) forwardAudioToAlgo(ctx context.Context, session *VoiceSession, msg VoiceMessage) {
	// 构建算法服务请求
	algoReq := service.VoiceQueryRequest{
		ConversationID: session.ConversationID,
		AudioChunk:     msg.Chunk,
		Seq:            msg.Seq,
		Codec:          "opus",
		SampleRate:     16000,
	}

	// 调用算法服务流式处理
	responseCh, err := h.services.AlgoService.VoiceQuery(ctx, &algoReq)
	if err != nil {
		logrus.WithError(err).Error("Failed to call algo voice service")
		session.Send <- VoiceMessage{
			Type:  "error",
			Error: "语音处理失败",
		}
		return
	}

	// 转发响应
	go func() {
		for response := range responseCh {
			var voiceMsg VoiceMessage

			switch response.Type {
			case "asr_partial":
				voiceMsg = VoiceMessage{
					Type: "asr_partial",
					Seq:  response.Seq,
					Text: response.Text,
				}
			case "asr_final":
				voiceMsg = VoiceMessage{
					Type: "asr_final",
					Seq:  response.Seq,
					Text: response.Text,
				}
			case "llm_delta":
				voiceMsg = VoiceMessage{
					Type: "llm_delta",
					Text: response.Text,
				}
			case "tts_chunk":
				voiceMsg = VoiceMessage{
					Type: "tts_chunk",
					Seq:  response.Seq,
					PCM:  response.PCM,
				}
			case "refs":
				voiceMsg = VoiceMessage{
					Type:  "refs",
					Items: response.Refs,
				}
			case "done":
				voiceMsg = VoiceMessage{Type: "done"}
			case "error":
				voiceMsg = VoiceMessage{
					Type:  "error",
					Error: response.Error,
				}
			}

			select {
			case session.Send <- voiceMsg:
			case <-ctx.Done():
				return
			}
		}
	}()
}

// CancelChat 取消对话请求
func (h *Handlers) CancelChat(c *gin.Context) {
	requestID := c.GetHeader("X-Request-ID")
	if requestID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Request ID is required"})
		return
	}

	// 调用算法服务取消请求
	err := h.services.AlgoService.CancelRequest(c.Request.Context(), requestID)
	if err != nil {
		logrus.WithError(err).Error("Failed to cancel request")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to cancel request"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"status": "cancelled"})
}

func generateSessionID() string {
	return "voice_" + time.Now().Format("20060102150405") + "_" + randomString(8)
}

func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}
