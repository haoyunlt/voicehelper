package handlers

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

// WebRTCSignalingHandler WebRTC信令处理器
type WebRTCSignalingHandler struct {
	BaseHandler
	upgrader websocket.Upgrader
	rooms    map[string]*Room
	roomsMux sync.RWMutex
}

// Room WebRTC房间
type Room struct {
	ID      string
	Clients map[string]*Client
	mutex   sync.RWMutex
}

// Client WebRTC客户端
type Client struct {
	ID         string
	Conn       *websocket.Conn
	Room       *Room
	Send       chan []byte
	LastActive time.Time
}

// SignalingMessage 信令消息
type SignalingMessage struct {
	Type      string                 `json:"type"`
	Data      interface{}            `json:"data,omitempty"`
	From      string                 `json:"from,omitempty"`
	To        string                 `json:"to,omitempty"`
	RoomID    string                 `json:"room_id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// ICECandidate ICE候选者
type ICECandidate struct {
	Candidate     string `json:"candidate"`
	SDPMid        string `json:"sdpMid"`
	SDPMLineIndex int    `json:"sdpMLineIndex"`
}

// SessionDescription 会话描述
type SessionDescription struct {
	Type string `json:"type"` // "offer" or "answer"
	SDP  string `json:"sdp"`
}

// NewWebRTCSignalingHandler 创建WebRTC信令处理器
func NewWebRTCSignalingHandler(algoServiceURL string) *WebRTCSignalingHandler {
	handler := &WebRTCSignalingHandler{
		BaseHandler: BaseHandler{
			AlgoServiceURL: algoServiceURL,
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
			ReadBufferSize:  1024 * 2,
			WriteBufferSize: 1024 * 2,
		},
		rooms: make(map[string]*Room),
	}

	// 启动清理协程
	go handler.cleanupRoutine()

	return handler
}

// HandleWebRTCSignaling 处理WebRTC信令WebSocket连接
func (h *WebRTCSignalingHandler) HandleWebRTCSignaling(c *gin.Context) {
	conn, err := h.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logrus.WithError(err).Error("Failed to upgrade to WebSocket for WebRTC signaling")
		return
	}
	defer conn.Close()

	// WebSocket 连接建立

	// 提取客户端信息
	traceID, tenantID := h.extractTraceInfo(c)
	clientID := c.Query("client_id")
	if clientID == "" {
		clientID = h.generateTraceID()
	}

	roomID := c.Query("room_id")
	if roomID == "" {
		roomID = "default"
	}

	logrus.WithFields(logrus.Fields{
		"trace_id":  traceID,
		"tenant_id": tenantID,
		"client_id": clientID,
		"room_id":   roomID,
	}).Info("WebRTC signaling connection established")

	// 创建客户端
	client := &Client{
		ID:         clientID,
		Conn:       conn,
		Send:       make(chan []byte, 256),
		LastActive: time.Now(),
	}

	// 加入房间
	room := h.getOrCreateRoom(roomID)
	room.addClient(client)
	client.Room = room

	// 发送连接确认
	welcomeMsg := SignalingMessage{
		Type:      "connected",
		RoomID:    roomID,
		From:      "server",
		To:        clientID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"client_id": clientID,
			"room_id":   roomID,
			"clients":   room.getClientIDs(),
		},
	}

	if err := conn.WriteJSON(welcomeMsg); err != nil {
		logrus.WithError(err).Error("Failed to send welcome message")
		return
	}

	// 通知房间内其他客户端
	room.broadcast(SignalingMessage{
		Type:      "client_joined",
		RoomID:    roomID,
		From:      "server",
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"client_id": clientID,
			"clients":   room.getClientIDs(),
		},
	}, clientID)

	// 启动消息处理协程
	go h.handleClientWrite(client)
	h.handleClientRead(client)

	// 清理
	room.removeClient(clientID)
	close(client.Send)
}

// handleClientRead 处理客户端读取
func (h *WebRTCSignalingHandler) handleClientRead(client *Client) {
	defer client.Conn.Close()

	client.Conn.SetReadLimit(512)
	client.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	client.Conn.SetPongHandler(func(string) error {
		client.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		client.LastActive = time.Now()
		return nil
	})

	for {
		var msg SignalingMessage
		err := client.Conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				logrus.WithError(err).Error("WebRTC signaling read error")
			}
			break
		}

		client.LastActive = time.Now()
		client.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))

		// 设置发送者
		msg.From = client.ID
		msg.RoomID = client.Room.ID
		msg.Timestamp = time.Now().Unix()

		// 处理信令消息
		h.handleSignalingMessage(client, msg)
	}
}

// handleClientWrite 处理客户端写入
func (h *WebRTCSignalingHandler) handleClientWrite(client *Client) {
	ticker := time.NewTicker(54 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case message, ok := <-client.Send:
			client.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				client.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			if err := client.Conn.WriteMessage(websocket.TextMessage, message); err != nil {
				logrus.WithError(err).Error("Failed to write WebRTC signaling message")
				return
			}

		case <-ticker.C:
			client.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := client.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// handleSignalingMessage 处理信令消息
func (h *WebRTCSignalingHandler) handleSignalingMessage(client *Client, msg SignalingMessage) {
	switch msg.Type {
	case "offer", "answer":
		h.handleSessionDescription(client, msg)
	case "ice_candidate":
		h.handleICECandidate(client, msg)
	case "join_room":
		h.handleJoinRoom(client, msg)
	case "leave_room":
		h.handleLeaveRoom(client, msg)
	case "ping":
		h.sendToClient(client, SignalingMessage{
			Type:      "pong",
			From:      "server",
			To:        client.ID,
			RoomID:    client.Room.ID,
			Timestamp: time.Now().Unix(),
		})
	default:
		logrus.WithField("message_type", msg.Type).Warn("Unknown WebRTC signaling message type")
		h.sendToClient(client, SignalingMessage{
			Type:      "error",
			From:      "server",
			To:        client.ID,
			RoomID:    client.Room.ID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "unknown_message_type",
				"message": "Unknown message type: " + msg.Type,
			},
		})
	}
}

// handleSessionDescription 处理会话描述（offer/answer）
func (h *WebRTCSignalingHandler) handleSessionDescription(client *Client, msg SignalingMessage) {
	if msg.To == "" {
		// 广播给房间内其他客户端
		client.Room.broadcast(msg, client.ID)
	} else {
		// 发送给指定客户端
		client.Room.sendToClient(msg.To, msg)
	}
}

// handleICECandidate 处理ICE候选者
func (h *WebRTCSignalingHandler) handleICECandidate(client *Client, msg SignalingMessage) {
	if msg.To == "" {
		// 广播给房间内其他客户端
		client.Room.broadcast(msg, client.ID)
	} else {
		// 发送给指定客户端
		client.Room.sendToClient(msg.To, msg)
	}
}

// handleJoinRoom 处理加入房间
func (h *WebRTCSignalingHandler) handleJoinRoom(client *Client, msg SignalingMessage) {
	roomID, ok := msg.Data.(string)
	if !ok {
		h.sendToClient(client, SignalingMessage{
			Type:      "error",
			From:      "server",
			To:        client.ID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "invalid_room_id",
				"message": "Invalid room ID",
			},
		})
		return
	}

	// 离开当前房间
	client.Room.removeClient(client.ID)

	// 加入新房间
	newRoom := h.getOrCreateRoom(roomID)
	newRoom.addClient(client)
	client.Room = newRoom

	// 发送确认
	h.sendToClient(client, SignalingMessage{
		Type:      "room_joined",
		From:      "server",
		To:        client.ID,
		RoomID:    roomID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"room_id": roomID,
			"clients": newRoom.getClientIDs(),
		},
	})

	// 通知房间内其他客户端
	newRoom.broadcast(SignalingMessage{
		Type:      "client_joined",
		From:      "server",
		RoomID:    roomID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"client_id": client.ID,
			"clients":   newRoom.getClientIDs(),
		},
	}, client.ID)
}

// handleLeaveRoom 处理离开房间
func (h *WebRTCSignalingHandler) handleLeaveRoom(client *Client, msg SignalingMessage) {
	roomID := client.Room.ID
	client.Room.removeClient(client.ID)

	// 发送确认
	h.sendToClient(client, SignalingMessage{
		Type:      "room_left",
		From:      "server",
		To:        client.ID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"room_id": roomID,
		},
	})
}

// sendToClient 发送消息给客户端
func (h *WebRTCSignalingHandler) sendToClient(client *Client, msg SignalingMessage) {
	data, err := json.Marshal(msg)
	if err != nil {
		logrus.WithError(err).Error("Failed to marshal signaling message")
		return
	}

	select {
	case client.Send <- data:
	default:
		close(client.Send)
		client.Room.removeClient(client.ID)
	}
}

// getOrCreateRoom 获取或创建房间
func (h *WebRTCSignalingHandler) getOrCreateRoom(roomID string) *Room {
	h.roomsMux.Lock()
	defer h.roomsMux.Unlock()

	room, exists := h.rooms[roomID]
	if !exists {
		room = &Room{
			ID:      roomID,
			Clients: make(map[string]*Client),
		}
		h.rooms[roomID] = room
		logrus.WithField("room_id", roomID).Info("Created new WebRTC room")
	}
	return room
}

// cleanupRoutine 清理协程
func (h *WebRTCSignalingHandler) cleanupRoutine() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		h.roomsMux.Lock()
		for roomID, room := range h.rooms {
			room.mutex.Lock()
			if len(room.Clients) == 0 {
				delete(h.rooms, roomID)
				logrus.WithField("room_id", roomID).Info("Cleaned up empty WebRTC room")
			} else {
				// 清理不活跃的客户端
				for clientID, client := range room.Clients {
					if time.Since(client.LastActive) > 10*time.Minute {
						delete(room.Clients, clientID)
						close(client.Send)
						logrus.WithFields(logrus.Fields{
							"room_id":   roomID,
							"client_id": clientID,
						}).Info("Cleaned up inactive WebRTC client")
					}
				}
			}
			room.mutex.Unlock()
		}
		h.roomsMux.Unlock()
	}
}

// Room methods

// addClient 添加客户端到房间
func (r *Room) addClient(client *Client) {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	r.Clients[client.ID] = client
}

// removeClient 从房间移除客户端
func (r *Room) removeClient(clientID string) {
	r.mutex.Lock()
	defer r.mutex.Unlock()

	if client, exists := r.Clients[clientID]; exists {
		delete(r.Clients, clientID)

		// 通知其他客户端
		r.broadcastUnsafe(SignalingMessage{
			Type:      "client_left",
			From:      "server",
			RoomID:    r.ID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"client_id": clientID,
				"clients":   r.getClientIDsUnsafe(),
			},
		}, clientID)

		logrus.WithFields(logrus.Fields{
			"room_id":   r.ID,
			"client_id": clientID,
		}).Info("Client left WebRTC room")

		// 关闭客户端连接
		client.Conn.Close()
	}
}

// broadcast 广播消息给房间内所有客户端（除了排除的客户端）
func (r *Room) broadcast(msg SignalingMessage, excludeClientID string) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	r.broadcastUnsafe(msg, excludeClientID)
}

// broadcastUnsafe 不安全的广播（需要调用者持有锁）
func (r *Room) broadcastUnsafe(msg SignalingMessage, excludeClientID string) {
	data, err := json.Marshal(msg)
	if err != nil {
		logrus.WithError(err).Error("Failed to marshal broadcast message")
		return
	}

	for clientID, client := range r.Clients {
		if clientID != excludeClientID {
			select {
			case client.Send <- data:
			default:
				close(client.Send)
				delete(r.Clients, clientID)
			}
		}
	}
}

// sendToClient 发送消息给指定客户端
func (r *Room) sendToClient(clientID string, msg SignalingMessage) {
	r.mutex.RLock()
	client, exists := r.Clients[clientID]
	r.mutex.RUnlock()

	if !exists {
		return
	}

	data, err := json.Marshal(msg)
	if err != nil {
		logrus.WithError(err).Error("Failed to marshal message to client")
		return
	}

	select {
	case client.Send <- data:
	default:
		close(client.Send)
		r.removeClient(clientID)
	}
}

// getClientIDs 获取房间内所有客户端ID
func (r *Room) getClientIDs() []string {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	return r.getClientIDsUnsafe()
}

// getClientIDsUnsafe 不安全的获取客户端ID（需要调用者持有锁）
func (r *Room) getClientIDsUnsafe() []string {
	ids := make([]string, 0, len(r.Clients))
	for id := range r.Clients {
		ids = append(ids, id)
	}
	return ids
}
