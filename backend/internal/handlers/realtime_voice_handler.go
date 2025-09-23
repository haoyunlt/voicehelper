package handlers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

type RealtimeVoiceHandler struct {
	upgrader    websocket.Upgrader
	sessions    map[string]*RealtimeVoiceSession
	sessionsMux sync.RWMutex
	asrService  ASRService
	ttsService  TTSService
	metrics     MetricsCollector
}

type RealtimeVoiceSession struct {
	ID           string              `json:"id"`
	UserID       string              `json:"user_id"`
	TenantID     string              `json:"tenant_id"`
	Connection   *websocket.Conn     `json:"-"`
	Context      context.Context     `json:"-"`
	Cancel       context.CancelFunc  `json:"-"`
	State        VoiceSessionState   `json:"state"`
	Config       RealtimeVoiceConfig `json:"config"`
	Stats        VoiceSessionStats   `json:"stats"`
	LastActivity time.Time           `json:"last_activity"`
	CreatedAt    time.Time           `json:"created_at"`
}

type VoiceSessionState struct {
	IsRecording    bool   `json:"is_recording"`
	IsSpeaking     bool   `json:"is_speaking"`
	CurrentText    string `json:"current_text"`
	AudioBuffer    []byte `json:"-"`
	LastTranscript string `json:"last_transcript"`
	Language       string `json:"language"`
}

type RealtimeVoiceConfig struct {
	ASRModel        string `json:"asr_model"`
	TTSVoice        string `json:"tts_voice"`
	Language        string `json:"language"`
	SampleRate      int    `json:"sample_rate"`
	Channels        int    `json:"channels"`
	VADSensitivity  int    `json:"vad_sensitivity"`
	AutoPunctuation bool   `json:"auto_punctuation"`
	RealTimeResults bool   `json:"real_time_results"`
}

type VoiceSessionStats struct {
	TotalAudioDuration float64   `json:"total_audio_duration"`
	TranscriptionCount int       `json:"transcription_count"`
	AverageLatency     float64   `json:"average_latency"`
	ErrorCount         int       `json:"error_count"`
	StartTime          time.Time `json:"start_time"`
	LastUpdateTime     time.Time `json:"last_update_time"`
}

type RealtimeVoiceMessage struct {
	Type      string                 `json:"type"`
	SessionID string                 `json:"session_id,omitempty"`
	Data      interface{}            `json:"data,omitempty"`
	Timestamp int64                  `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type AudioChunk struct {
	Data       string `json:"data"`        // base64编码的音频数据
	Format     string `json:"format"`      // 音频格式: pcm16, opus, mp3
	SampleRate int    `json:"sample_rate"` // 采样率
	Channels   int    `json:"channels"`    // 声道数
	Duration   int    `json:"duration"`    // 持续时间(ms)
	IsFinal    bool   `json:"is_final"`    // 是否为最后一块
	Sequence   int    `json:"sequence"`    // 序列号
}

type TranscriptionResult struct {
	Text         string  `json:"text"`
	Confidence   float64 `json:"confidence"`
	IsFinal      bool    `json:"is_final"`
	Language     string  `json:"language"`
	Duration     int     `json:"duration"`
	WordCount    int     `json:"word_count"`
	ProcessingMs float64 `json:"processing_ms"`
}

func NewRealtimeVoiceHandler(
	asrService ASRService,
	ttsService TTSService,
	metrics MetricsCollector,
) *RealtimeVoiceHandler {
	return &RealtimeVoiceHandler{
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// 生产环境需要严格的Origin检查
				return true
			},
			ReadBufferSize:   8192,
			WriteBufferSize:  8192,
			HandshakeTimeout: 10 * time.Second,
		},
		sessions:   make(map[string]*RealtimeVoiceSession),
		asrService: asrService,
		ttsService: ttsService,
		metrics:    metrics,
	}
}

func (h *RealtimeVoiceHandler) HandleWebSocket(c *gin.Context) {
	// 升级到WebSocket连接
	conn, err := h.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket升级失败: %v", err)
		h.metrics.RecordError("websocket_upgrade_failed", err)
		return
	}
	defer conn.Close()

	// 创建语音会话
	session := h.createVoiceSession(conn, c)

	// 记录连接指标
	h.metrics.IncrementWSConnections("voice", session.TenantID)
	defer h.metrics.DecrementWSConnections()

	// 发送会话创建确认
	h.sendSessionCreated(session)

	// 启动会话管理协程
	go h.manageSession(session)

	// 处理消息循环
	h.handleMessageLoop(session)

	// 清理会话
	h.cleanupSession(session)
}

func (h *RealtimeVoiceHandler) createVoiceSession(conn *websocket.Conn, c *gin.Context) *RealtimeVoiceSession {
	sessionID := uuid.New().String()
	ctx, cancel := context.WithCancel(context.Background())

	session := &RealtimeVoiceSession{
		ID:         sessionID,
		UserID:     c.GetString("user_id"),
		TenantID:   c.GetString("tenant_id"),
		Connection: conn,
		Context:    ctx,
		Cancel:     cancel,
		State: VoiceSessionState{
			IsRecording:    false,
			IsSpeaking:     false,
			CurrentText:    "",
			AudioBuffer:    make([]byte, 0),
			LastTranscript: "",
			Language:       "zh-CN",
		},
		Config: RealtimeVoiceConfig{
			ASRModel:        "whisper-base",
			TTSVoice:        "zh-CN-XiaoxiaoNeural",
			Language:        "zh-CN",
			SampleRate:      16000,
			Channels:        1,
			VADSensitivity:  2,
			AutoPunctuation: true,
			RealTimeResults: true,
		},
		Stats: VoiceSessionStats{
			StartTime:      time.Now(),
			LastUpdateTime: time.Now(),
		},
		LastActivity: time.Now(),
		CreatedAt:    time.Now(),
	}

	h.sessionsMux.Lock()
	h.sessions[sessionID] = session
	h.sessionsMux.Unlock()

	return session
}

func (h *RealtimeVoiceHandler) handleMessageLoop(session *RealtimeVoiceSession) {
	defer session.Cancel()

	for {
		select {
		case <-session.Context.Done():
			return
		default:
			// 设置读取超时
			session.Connection.SetReadDeadline(time.Now().Add(60 * time.Second))

			var msg RealtimeVoiceMessage
			err := session.Connection.ReadJSON(&msg)
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("WebSocket读取错误: %v", err)
					h.metrics.RecordError("websocket_read_error", err)
				}
				return
			}

			// 更新活动时间
			session.LastActivity = time.Now()
			session.Stats.LastUpdateTime = time.Now()

			// 记录消息指标
			h.metrics.RecordWSMessage("received", msg.Type)

			// 处理消息
			h.handleMessage(session, msg)
		}
	}
}

func (h *RealtimeVoiceHandler) handleMessage(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	switch msg.Type {
	case "start_recording":
		h.handleStartRecording(session, msg)
	case "audio_chunk":
		h.handleAudioChunk(session, msg)
	case "stop_recording":
		h.handleStopRecording(session, msg)
	case "configure":
		h.handleConfigure(session, msg)
	case "get_stats":
		h.handleGetStats(session, msg)
	case "ping":
		h.handlePing(session, msg)
	default:
		h.sendError(session, "unknown_message_type", fmt.Sprintf("未知消息类型: %s", msg.Type))
	}
}

func (h *RealtimeVoiceHandler) handleStartRecording(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	if session.State.IsRecording {
		h.sendError(session, "already_recording", "会话已在录音中")
		return
	}

	session.State.IsRecording = true
	session.State.AudioBuffer = session.State.AudioBuffer[:0] // 清空缓冲区
	session.Stats.StartTime = time.Now()

	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "recording_started",
		SessionID: session.ID,
		Data: map[string]interface{}{
			"session_id": session.ID,
			"config":     session.Config,
			"status":     "recording",
		},
		Timestamp: time.Now().UnixMilli(),
	})

	log.Printf("会话 %s 开始录音", session.ID)
}

func (h *RealtimeVoiceHandler) handleAudioChunk(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	if !session.State.IsRecording {
		h.sendError(session, "not_recording", "会话未在录音状态")
		return
	}

	// 解析音频数据
	audioDataMap, ok := msg.Data.(map[string]interface{})
	if !ok {
		h.sendError(session, "invalid_audio_data", "无效的音频数据格式")
		return
	}

	var audioChunk AudioChunk
	audioDataBytes, _ := json.Marshal(audioDataMap)
	if err := json.Unmarshal(audioDataBytes, &audioChunk); err != nil {
		h.sendError(session, "parse_error", fmt.Sprintf("音频数据解析失败: %v", err))
		return
	}

	// 异步处理音频数据
	go h.processAudioChunk(session, audioChunk)
}

func (h *RealtimeVoiceHandler) processAudioChunk(session *RealtimeVoiceSession, audioChunk AudioChunk) {
	startTime := time.Now()

	// 解码base64音频数据
	audioData, err := base64.StdEncoding.DecodeString(audioChunk.Data)
	if err != nil {
		h.sendError(session, "decode_error", fmt.Sprintf("音频数据解码失败: %v", err))
		return
	}

	// 添加到音频缓冲区
	session.State.AudioBuffer = append(session.State.AudioBuffer, audioData...)

	// 调用ASR服务
	asrResult, err := h.asrService.ProcessAudio(ASRRequest{
		AudioData:  audioData,
		Format:     audioChunk.Format,
		SampleRate: audioChunk.SampleRate,
		Channels:   audioChunk.Channels,
		Language:   session.Config.Language,
		IsFinal:    audioChunk.IsFinal,
		SessionID:  session.ID,
	})

	if err != nil {
		log.Printf("ASR处理失败: %v", err)
		h.metrics.RecordError("asr_processing_failed", err)
		session.Stats.ErrorCount++
		return
	}

	// 更新会话状态
	if asrResult.Text != "" {
		session.State.CurrentText = asrResult.Text
		if asrResult.IsFinal {
			session.State.LastTranscript = asrResult.Text
			session.Stats.TranscriptionCount++
		}
	}

	// 计算处理延迟
	processingTime := time.Since(startTime)
	session.Stats.AverageLatency = (session.Stats.AverageLatency + processingTime.Seconds()) / 2

	// 发送ASR结果
	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "transcription",
		SessionID: session.ID,
		Data: TranscriptionResult{
			Text:         asrResult.Text,
			Confidence:   asrResult.Confidence,
			IsFinal:      asrResult.IsFinal,
			Language:     asrResult.Language,
			Duration:     audioChunk.Duration,
			WordCount:    len(strings.Fields(asrResult.Text)),
			ProcessingMs: processingTime.Seconds() * 1000,
		},
		Timestamp: time.Now().UnixMilli(),
	})

	// 记录指标
	h.metrics.RecordASRRequest(
		session.Config.ASRModel,
		session.Config.Language,
		"success",
		processingTime,
	)
}

func (h *RealtimeVoiceHandler) handleStopRecording(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	if !session.State.IsRecording {
		h.sendError(session, "not_recording", "会话未在录音状态")
		return
	}

	session.State.IsRecording = false
	session.Stats.TotalAudioDuration = time.Since(session.Stats.StartTime).Seconds()

	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "recording_stopped",
		SessionID: session.ID,
		Data: map[string]interface{}{
			"session_id":     session.ID,
			"status":         "stopped",
			"final_text":     session.State.LastTranscript,
			"total_duration": session.Stats.TotalAudioDuration,
			"stats":          session.Stats,
		},
		Timestamp: time.Now().UnixMilli(),
	})

	log.Printf("会话 %s 停止录音，总时长: %.2fs", session.ID, session.Stats.TotalAudioDuration)
}

func (h *RealtimeVoiceHandler) handleConfigure(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	configData, ok := msg.Data.(map[string]interface{})
	if !ok {
		h.sendError(session, "invalid_config", "无效的配置数据")
		return
	}

	// 更新配置
	if asrModel, exists := configData["asr_model"]; exists {
		if model, ok := asrModel.(string); ok {
			session.Config.ASRModel = model
		}
	}

	if ttsVoice, exists := configData["tts_voice"]; exists {
		if voice, ok := ttsVoice.(string); ok {
			session.Config.TTSVoice = voice
		}
	}

	if language, exists := configData["language"]; exists {
		if lang, ok := language.(string); ok {
			session.Config.Language = lang
			session.State.Language = lang
		}
	}

	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "configured",
		SessionID: session.ID,
		Data: map[string]interface{}{
			"config": session.Config,
		},
		Timestamp: time.Now().UnixMilli(),
	})
}

func (h *RealtimeVoiceHandler) handleGetStats(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "stats",
		SessionID: session.ID,
		Data:      session.Stats,
		Timestamp: time.Now().UnixMilli(),
	})
}

func (h *RealtimeVoiceHandler) handlePing(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "pong",
		SessionID: session.ID,
		Data: map[string]interface{}{
			"server_time": time.Now().UnixMilli(),
		},
		Timestamp: time.Now().UnixMilli(),
	})
}

func (h *RealtimeVoiceHandler) sendMessage(session *RealtimeVoiceSession, msg RealtimeVoiceMessage) {
	session.Connection.SetWriteDeadline(time.Now().Add(10 * time.Second))
	if err := session.Connection.WriteJSON(msg); err != nil {
		log.Printf("发送消息失败: %v", err)
		h.metrics.RecordError("websocket_write_error", err)
		session.Cancel()
		return
	}

	h.metrics.RecordWSMessage("sent", msg.Type)
}

func (h *RealtimeVoiceHandler) sendError(session *RealtimeVoiceSession, code, message string) {
	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "error",
		SessionID: session.ID,
		Data: map[string]interface{}{
			"code":    code,
			"message": message,
		},
		Timestamp: time.Now().UnixMilli(),
	})
}

func (h *RealtimeVoiceHandler) sendSessionCreated(session *RealtimeVoiceSession) {
	h.sendMessage(session, RealtimeVoiceMessage{
		Type:      "session_created",
		SessionID: session.ID,
		Data: map[string]interface{}{
			"session_id": session.ID,
			"config":     session.Config,
			"status":     "ready",
			"server_info": map[string]interface{}{
				"version":      "1.0.0",
				"capabilities": []string{"asr", "tts", "real_time", "vad"},
			},
		},
		Timestamp: time.Now().UnixMilli(),
	})
}

func (h *RealtimeVoiceHandler) manageSession(session *RealtimeVoiceSession) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-session.Context.Done():
			return
		case <-ticker.C:
			// 检查会话超时
			if time.Since(session.LastActivity) > 10*time.Minute {
				log.Printf("会话 %s 超时，自动清理", session.ID)
				session.Cancel()
				return
			}

			// 发送心跳
			h.sendMessage(session, RealtimeVoiceMessage{
				Type:      "heartbeat",
				SessionID: session.ID,
				Data: map[string]interface{}{
					"server_time":      time.Now().UnixMilli(),
					"session_duration": time.Since(session.CreatedAt).Seconds(),
				},
				Timestamp: time.Now().UnixMilli(),
			})
		}
	}
}

func (h *RealtimeVoiceHandler) cleanupSession(session *RealtimeVoiceSession) {
	h.sessionsMux.Lock()
	delete(h.sessions, session.ID)
	h.sessionsMux.Unlock()

	log.Printf("会话 %s 已清理，持续时间: %.2fs",
		session.ID,
		time.Since(session.CreatedAt).Seconds())
}

// ASR和TTS服务接口定义
type ASRService interface {
	ProcessAudio(request ASRRequest) (*ASRResponse, error)
}

type TTSService interface {
	Synthesize(request TTSRequest) (*TTSResponse, error)
}

type ASRRequest struct {
	AudioData  []byte
	Format     string
	SampleRate int
	Channels   int
	Language   string
	IsFinal    bool
	SessionID  string
}

type ASRResponse struct {
	Text       string
	Confidence float64
	IsFinal    bool
	Language   string
}

type TTSRequest struct {
	Text      string
	Voice     string
	Language  string
	Format    string
	SessionID string
}

type TTSResponse struct {
	AudioData []byte
	Format    string
	Duration  float64
}

type MetricsCollector interface {
	IncrementWSConnections(connType, tenantID string)
	DecrementWSConnections()
	RecordWSMessage(direction, messageType string)
	RecordASRRequest(model, language, status string, latency time.Duration)
	RecordError(errorType string, err error)
}
