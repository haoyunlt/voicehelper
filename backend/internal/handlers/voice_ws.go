package handlers

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

// VoiceWSHandler WebSocket语音处理器
type VoiceWSHandler struct {
	BaseHandler
	upgrader websocket.Upgrader
}

// VoiceWSMessage 语音WebSocket消息结构
type VoiceWSMessage struct {
	Type      string                 `json:"type"`
	Data      interface{}            `json:"data,omitempty"`
	SessionID string                 `json:"session_id,omitempty"`
	Timestamp int64                  `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// AudioData 音频数据结构
type AudioData struct {
	Format     string `json:"format"`      // "pcm16", "opus", "mp3"
	SampleRate int    `json:"sample_rate"` // 采样率
	Channels   int    `json:"channels"`    // 声道数
	Data       []byte `json:"data"`        // 音频数据
}

// VoiceConfig 语音配置
type VoiceConfig struct {
	Language    string `json:"language"`     // 语言代码
	Model       string `json:"model"`        // 模型名称
	EnableVAD   bool   `json:"enable_vad"`   // 是否启用VAD
	EnableNoise bool   `json:"enable_noise"` // 是否启用降噪
}

// NewVoiceWSHandler 创建语音WebSocket处理器
func NewVoiceWSHandler(algoServiceURL string) *VoiceWSHandler {
	return &VoiceWSHandler{
		BaseHandler: BaseHandler{
			AlgoServiceURL: algoServiceURL,
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// 在生产环境中应该检查Origin
				return true
			},
			ReadBufferSize:  1024 * 4, // 4KB读缓冲
			WriteBufferSize: 1024 * 4, // 4KB写缓冲
		},
	}
}

// HandleVoiceWebSocket 处理语音WebSocket连接
func (h *VoiceWSHandler) HandleVoiceWebSocket(c *gin.Context) {
	// 升级到WebSocket连接
	conn, err := h.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logrus.WithError(err).Error("Failed to upgrade to WebSocket")
		return
	}
	defer conn.Close()

	// 提取会话信息
	traceID, tenantID := h.extractTraceInfo(c)
	sessionID := c.Query("session_id")
	if sessionID == "" {
		sessionID = h.generateTraceID()
	}

	// 记录WebSocket连接指标
	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"tenant_id":  tenantID,
		"trace_id":   traceID,
	}).Info("WebSocket voice connection established")

	logrus.WithFields(logrus.Fields{
		"trace_id":   traceID,
		"tenant_id":  tenantID,
		"session_id": sessionID,
	}).Info("Voice WebSocket connection established")

	// 发送连接确认
	welcomeMsg := VoiceWSMessage{
		Type:      "connection_established",
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"status":     "connected",
			"session_id": sessionID,
			"trace_id":   traceID,
		},
	}

	if err := conn.WriteJSON(welcomeMsg); err != nil {
		logrus.WithError(err).Error("Failed to send welcome message")
		return
	}

	// 处理消息循环
	h.handleMessageLoop(conn, sessionID, traceID, tenantID)
}

// handleMessageLoop 处理消息循环
func (h *VoiceWSHandler) handleMessageLoop(conn *websocket.Conn, sessionID, traceID, tenantID string) {
	// 设置读取超时
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	// 启动心跳
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	go func() {
		for range ticker.C {
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}()

	for {
		var msg VoiceWSMessage
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				logrus.WithError(err).Error("WebSocket read error")
				logrus.WithError(err).Error("WebSocket read error")
			}
			break
		}

		logrus.WithField("message_type", msg.Type).Debug("Received WebSocket message")

		// 重置读取超时
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))

		// 处理不同类型的消息
		response := h.handleVoiceMessage(msg, sessionID, traceID, tenantID)
		if response != nil {
			if err := conn.WriteJSON(response); err != nil {
				logrus.WithError(err).Error("Failed to send response")
				logrus.WithError(err).Error("Failed to send WebSocket response")
				break
			}
			logrus.WithField("response_type", response.Type).Debug("Sent WebSocket response")
		}
	}
}

// handleVoiceMessage 处理语音消息
func (h *VoiceWSHandler) handleVoiceMessage(msg VoiceWSMessage, sessionID, traceID, tenantID string) *VoiceWSMessage {
	switch msg.Type {
	case "audio_data":
		return h.handleAudioData(msg, sessionID, traceID, tenantID)
	case "start_recording":
		return h.handleStartRecording(msg, sessionID, traceID, tenantID)
	case "stop_recording":
		return h.handleStopRecording(msg, sessionID, traceID, tenantID)
	case "config_update":
		return h.handleConfigUpdate(msg, sessionID, traceID, tenantID)
	case "ping":
		return &VoiceWSMessage{
			Type:      "pong",
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
		}
	default:
		logrus.WithField("message_type", msg.Type).Warn("Unknown message type")
		return &VoiceWSMessage{
			Type:      "error",
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "unknown_message_type",
				"message": "Unknown message type: " + msg.Type,
			},
		}
	}
}

// handleAudioData 处理音频数据
func (h *VoiceWSHandler) handleAudioData(msg VoiceWSMessage, sessionID, traceID, tenantID string) *VoiceWSMessage {
	// 解析音频数据
	audioDataBytes, err := json.Marshal(msg.Data)
	if err != nil {
		logrus.WithError(err).Error("Failed to marshal audio data")
		return &VoiceWSMessage{
			Type:      "error",
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "invalid_audio_data",
				"message": "Failed to parse audio data",
			},
		}
	}

	var audioData AudioData
	if err := json.Unmarshal(audioDataBytes, &audioData); err != nil {
		logrus.WithError(err).Error("Failed to unmarshal audio data")
		return &VoiceWSMessage{
			Type:      "error",
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "invalid_audio_format",
				"message": "Invalid audio data format",
			},
		}
	}

	// 记录音频处理指标
	logrus.WithFields(logrus.Fields{
		"session_id":   sessionID,
		"audio_bytes":  len(audioData.Data),
		"audio_format": audioData.Format,
		"sample_rate":  audioData.SampleRate,
	}).Info("Processing audio data")

	// 发送到算法服务进行处理
	processedResult, err := h.sendToAlgoService(audioData, sessionID, tenantID)
	if err != nil {
		logrus.WithError(err).Error("Failed to process audio with algo service")
		return &VoiceWSMessage{
			Type:      "error",
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "processing_failed",
				"message": "音频处理失败: " + err.Error(),
			},
		}
	}

	// 返回处理结果
	return &VoiceWSMessage{
		Type:      "audio_processed",
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
		Data:      processedResult,
	}
}

// handleStartRecording 处理开始录音
func (h *VoiceWSHandler) handleStartRecording(msg VoiceWSMessage, sessionID, traceID, tenantID string) *VoiceWSMessage {
	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"trace_id":   traceID,
	}).Info("Starting voice recording")

	return &VoiceWSMessage{
		Type:      "recording_started",
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"status": "recording",
		},
	}
}

// handleStopRecording 处理停止录音
func (h *VoiceWSHandler) handleStopRecording(msg VoiceWSMessage, sessionID, traceID, tenantID string) *VoiceWSMessage {
	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"trace_id":   traceID,
	}).Info("Stopping voice recording")

	return &VoiceWSMessage{
		Type:      "recording_stopped",
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"status": "stopped",
		},
	}
}

// handleConfigUpdate 处理配置更新
func (h *VoiceWSHandler) handleConfigUpdate(msg VoiceWSMessage, sessionID, traceID, tenantID string) *VoiceWSMessage {
	configBytes, err := json.Marshal(msg.Data)
	if err != nil {
		return &VoiceWSMessage{
			Type:      "error",
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "invalid_config",
				"message": "Failed to parse configuration",
			},
		}
	}

	var config VoiceConfig
	if err := json.Unmarshal(configBytes, &config); err != nil {
		return &VoiceWSMessage{
			Type:      "error",
			SessionID: sessionID,
			Timestamp: time.Now().Unix(),
			Data: map[string]interface{}{
				"error":   "invalid_config_format",
				"message": "Invalid configuration format",
			},
		}
	}

	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"config":     config,
	}).Info("Voice configuration updated")

	return &VoiceWSMessage{
		Type:      "config_updated",
		SessionID: sessionID,
		Timestamp: time.Now().Unix(),
		Data: map[string]interface{}{
			"status": "updated",
			"config": config,
		},
	}
}

// sendToAlgoService 发送音频数据到算法服务进行处理
func (h *VoiceWSHandler) sendToAlgoService(audioData AudioData, sessionID, tenantID string) (map[string]interface{}, error) {
	// 记录处理开始时间
	startTime := time.Now()

	// 这里应该实现实际的HTTP请求到算法服务
	// 为了演示，返回模拟的处理结果
	result := map[string]interface{}{
		"status":          "processed",
		"bytes_count":     len(audioData.Data),
		"format":          audioData.Format,
		"processing_time": 0.15,          // 模拟处理时间
		"transcript":      "这是语音识别的结果文本", // 模拟ASR结果
		"confidence":      0.95,
		"voice_activity": map[string]interface{}{
			"detected": true,
			"duration": 2.5,
		},
		"audio_quality": map[string]interface{}{
			"snr":    15.2,
			"volume": 0.8,
		},
	}

	// 记录处理指标
	processingTime := time.Since(startTime)
	logrus.WithFields(logrus.Fields{
		"session_id":      sessionID,
		"processing_time": processingTime.Milliseconds(),
		"audio_bytes":     len(audioData.Data),
	}).Info("Audio processing completed")

	return result, nil
}

// establishAlgoServiceConnection 建立到算法服务的WebSocket连接
func (h *VoiceWSHandler) establishAlgoServiceConnection(sessionID, tenantID string) (*websocket.Conn, error) {
	// 构建算法服务WebSocket URL
	algoWSURL := h.AlgoServiceURL + "/voice/stream"

	// 设置请求头
	headers := http.Header{}
	headers.Set("X-Session-ID", sessionID)
	headers.Set("X-Tenant-ID", tenantID)
	headers.Set("X-Trace-ID", generateTraceID())

	// 建立WebSocket连接
	conn, _, err := websocket.DefaultDialer.Dial(algoWSURL, headers)
	if err != nil {
		return nil, err
	}

	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"tenant_id":  tenantID,
		"algo_url":   algoWSURL,
	}).Info("Established connection to algo service")

	return conn, nil
}

// forwardToAlgoService 转发音频数据到算法服务
func (h *VoiceWSHandler) forwardToAlgoService(conn *websocket.Conn, audioData AudioData, sessionID string) error {
	message := map[string]interface{}{
		"type":       "audio_data",
		"session_id": sessionID,
		"data":       audioData,
		"timestamp":  time.Now().Unix(),
	}

	return conn.WriteJSON(message)
}

// closeAlgoServiceConnection 关闭到算法服务的WebSocket连接
func (h *VoiceWSHandler) closeAlgoServiceConnection(conn *websocket.Conn, sessionID string) {
	if conn != nil {
		// 发送关闭消息
		closeMessage := map[string]interface{}{
			"type":       "close",
			"session_id": sessionID,
			"timestamp":  time.Now().Unix(),
		}

		conn.WriteJSON(closeMessage)
		conn.Close()

		logrus.WithField("session_id", sessionID).Info("Closed connection to algo service")
	}
}

// generateTraceID 生成追踪ID
func generateTraceID() string {
	return "trace_" + time.Now().Format("20060102150405") + "_" + generateRandomString(8)
}

// generateRandomString 生成随机字符串
func generateRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	result := make([]byte, length)
	for i := range result {
		result[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(result)
}
