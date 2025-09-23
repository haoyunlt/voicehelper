package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/signal"
	"time"

	"github.com/gorilla/websocket"
)

// VoiceMessage 语音消息结构
type VoiceMessage struct {
	Type      string                 `json:"type"`
	SessionID string                 `json:"session_id,omitempty"`
	Data      string                 `json:"data,omitempty"`
	Config    map[string]interface{} `json:"config,omitempty"`
	Timestamp int64                  `json:"timestamp,omitempty"`
}

// VoiceTestClient 语音测试客户端
type VoiceTestClient struct {
	serverURL string
	sessionID string
	conn      *websocket.Conn
	done      chan struct{}
}

// NewVoiceTestClient 创建语音测试客户端
func NewVoiceTestClient(serverURL, sessionID string) *VoiceTestClient {
	return &VoiceTestClient{
		serverURL: serverURL,
		sessionID: sessionID,
		done:      make(chan struct{}),
	}
}

// Connect 连接到语音服务
func (c *VoiceTestClient) Connect() error {
	u, err := url.Parse(c.serverURL)
	if err != nil {
		return fmt.Errorf("invalid server URL: %v", err)
	}

	// 设置WebSocket协议
	if u.Scheme == "http" {
		u.Scheme = "ws"
	} else if u.Scheme == "https" {
		u.Scheme = "wss"
	}
	u.Path = "/api/v2/voice/stream"

	log.Printf("Connecting to %s", u.String())

	// 建立WebSocket连接
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial(u.String(), nil)
	if err != nil {
		return fmt.Errorf("failed to connect: %v", err)
	}

	c.conn = conn
	log.Printf("Connected to voice service")

	// 启动消息接收goroutine
	go c.readMessages()

	return nil
}

// StartVoiceSession 开始语音会话
func (c *VoiceTestClient) StartVoiceSession() error {
	config := map[string]interface{}{
		"sample_rate": 16000,
		"channels":    1,
		"language":    "zh-CN",
		"format":      "pcm",
	}

	message := VoiceMessage{
		Type:      "start",
		SessionID: c.sessionID,
		Config:    config,
		Timestamp: time.Now().Unix(),
	}

	return c.sendMessage(message)
}

// SendAudioData 发送音频数据
func (c *VoiceTestClient) SendAudioData(audioData []byte) error {
	// 将音频数据编码为base64
	encodedData := base64.StdEncoding.EncodeToString(audioData)

	message := VoiceMessage{
		Type:      "audio",
		SessionID: c.sessionID,
		Data:      encodedData,
		Timestamp: time.Now().Unix(),
	}

	return c.sendMessage(message)
}

// StopVoiceSession 停止语音会话
func (c *VoiceTestClient) StopVoiceSession() error {
	message := VoiceMessage{
		Type:      "stop",
		SessionID: c.sessionID,
		Timestamp: time.Now().Unix(),
	}

	return c.sendMessage(message)
}

// sendMessage 发送消息
func (c *VoiceTestClient) sendMessage(message VoiceMessage) error {
	if c.conn == nil {
		return fmt.Errorf("not connected")
	}

	log.Printf("Sending message: %s", message.Type)
	return c.conn.WriteJSON(message)
}

// readMessages 读取消息
func (c *VoiceTestClient) readMessages() {
	defer close(c.done)

	for {
		var message map[string]interface{}
		err := c.conn.ReadJSON(&message)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			return
		}

		c.handleMessage(message)
	}
}

// handleMessage 处理接收到的消息
func (c *VoiceTestClient) handleMessage(message map[string]interface{}) {
	msgType, ok := message["type"].(string)
	if !ok {
		msgType = message["event"].(string) // 兼容旧格式
	}

	log.Printf("Received message: %s", msgType)

	switch msgType {
	case "connected":
		log.Printf("✅ Connected to voice service")

	case "session_started":
		log.Printf("✅ Voice session started: %s", c.sessionID)

	case "asr_partial":
		if text, ok := message["text"].(string); ok {
			log.Printf("🎤 ASR Partial: %s", text)
		}

	case "asr_final":
		if text, ok := message["text"].(string); ok {
			confidence := message["confidence"]
			log.Printf("🎤 ASR Final: %s (confidence: %v)", text, confidence)
		}

	case "agent_start":
		log.Printf("🤖 Agent processing started")

	case "agent_response":
		if text, ok := message["text"].(string); ok {
			isFinal := message["is_final"]
			log.Printf("🤖 Agent Response: %s (final: %v)", text, isFinal)
		}

	case "tts_audio":
		if audioData, ok := message["audio_data"].(string); ok {
			format := message["format"]
			log.Printf("🔊 TTS Audio received: %d bytes (%s)", len(audioData), format)
		}

	case "session_stopped":
		log.Printf("✅ Voice session stopped: %s", c.sessionID)

	case "error":
		if errorMsg, ok := message["error"].(string); ok {
			log.Printf("❌ Error: %s", errorMsg)
		}

	default:
		log.Printf("📨 Unknown message type: %s", msgType)
		if data, err := json.MarshalIndent(message, "", "  "); err == nil {
			log.Printf("Message data: %s", string(data))
		}
	}
}

// Close 关闭连接
func (c *VoiceTestClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// Wait 等待连接关闭
func (c *VoiceTestClient) Wait() {
	<-c.done
}

// simulateAudioData 模拟音频数据
func simulateAudioData() []byte {
	// 生成模拟的PCM音频数据（16kHz, 16bit, mono）
	// 这里只是生成一些随机数据作为示例
	data := make([]byte, 1600) // 100ms的音频数据
	for i := range data {
		data[i] = byte(i % 256)
	}
	return data
}

func main() {
	var (
		serverURL = flag.String("server", "http://localhost:8080", "Voice server URL")
		sessionID = flag.String("session", fmt.Sprintf("test-session-%d", time.Now().Unix()), "Session ID")
		duration  = flag.Int("duration", 10, "Test duration in seconds")
		audioFile = flag.String("audio", "", "Audio file to send (optional)")
	)
	flag.Parse()

	log.Printf("Starting voice test client")
	log.Printf("Server: %s", *serverURL)
	log.Printf("Session: %s", *sessionID)
	log.Printf("Duration: %d seconds", *duration)

	// 创建客户端
	client := NewVoiceTestClient(*serverURL, *sessionID)

	// 设置信号处理
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)

	// 连接到服务器
	if err := client.Connect(); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer client.Close()

	// 开始语音会话
	if err := client.StartVoiceSession(); err != nil {
		log.Fatalf("Failed to start voice session: %v", err)
	}

	// 发送音频数据
	go func() {
		ticker := time.NewTicker(100 * time.Millisecond) // 每100ms发送一次
		defer ticker.Stop()

		var audioData []byte
		if *audioFile != "" {
			// 读取音频文件
			data, err := os.ReadFile(*audioFile)
			if err != nil {
				log.Printf("Failed to read audio file: %v", err)
				audioData = simulateAudioData()
			} else {
				audioData = data
			}
		} else {
			audioData = simulateAudioData()
		}

		timeout := time.After(time.Duration(*duration) * time.Second)

		for {
			select {
			case <-ticker.C:
				if err := client.SendAudioData(audioData); err != nil {
					log.Printf("Failed to send audio data: %v", err)
					return
				}

			case <-timeout:
				log.Printf("Test duration completed")
				return

			case <-interrupt:
				log.Printf("Interrupted")
				return
			}
		}
	}()

	// 等待中断或完成
	select {
	case <-interrupt:
		log.Printf("Interrupt received, stopping...")

	case <-time.After(time.Duration(*duration+2) * time.Second):
		log.Printf("Test completed")
	}

	// 停止语音会话
	if err := client.StopVoiceSession(); err != nil {
		log.Printf("Failed to stop voice session: %v", err)
	}

	// 等待一段时间以接收最后的消息
	time.Sleep(2 * time.Second)

	log.Printf("Voice test client finished")
}
