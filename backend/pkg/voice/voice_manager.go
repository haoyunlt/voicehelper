package voice

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

// VoiceConfig 语音配置
type VoiceConfig struct {
	SampleRate int    `json:"sample_rate"`
	Channels   int    `json:"channels"`
	Language   string `json:"language"`
	Format     string `json:"format"`
}

// VoiceSession 语音会话
type VoiceSession struct {
	ID           string                 `json:"id"`
	UserID       string                 `json:"user_id"`
	TenantID     string                 `json:"tenant_id"`
	Config       VoiceConfig            `json:"config"`
	StartTime    time.Time              `json:"start_time"`
	LastActivity time.Time              `json:"last_activity"`
	Status       string                 `json:"status"` // active, paused, stopped
	Metadata     map[string]interface{} `json:"metadata"`

	// 内部状态
	ClientConn *websocket.Conn    `json:"-"`
	AlgoConn   *websocket.Conn    `json:"-"`
	Context    context.Context    `json:"-"`
	Cancel     context.CancelFunc `json:"-"`
	Mutex      sync.RWMutex       `json:"-"`
}

// VoiceMessage 语音消息
type VoiceMessage struct {
	Type      string                 `json:"type"`
	SessionID string                 `json:"session_id,omitempty"`
	Data      string                 `json:"data,omitempty"`
	Config    map[string]interface{} `json:"config,omitempty"`
	Timestamp int64                  `json:"timestamp,omitempty"`
}

// VoiceManager 语音管理器
type VoiceManager struct {
	sessions      map[string]*VoiceSession
	sessionsMutex sync.RWMutex

	// 配置
	algoServiceURL string
	maxSessions    int
	sessionTimeout time.Duration

	// 统计
	totalSessions  int64
	activeSessions int64
	processedAudio int64

	// 上下文
	ctx    context.Context
	cancel context.CancelFunc
}

// NewVoiceManager 创建语音管理器
func NewVoiceManager(algoServiceURL string) *VoiceManager {
	ctx, cancel := context.WithCancel(context.Background())

	vm := &VoiceManager{
		sessions:       make(map[string]*VoiceSession),
		algoServiceURL: algoServiceURL,
		maxSessions:    100,
		sessionTimeout: 10 * time.Minute,
		ctx:            ctx,
		cancel:         cancel,
	}

	// 启动清理例程
	go vm.cleanupRoutine()

	return vm
}

// CreateSession 创建语音会话
func (vm *VoiceManager) CreateSession(sessionID, userID, tenantID string, config VoiceConfig) (*VoiceSession, error) {
	vm.sessionsMutex.Lock()
	defer vm.sessionsMutex.Unlock()

	// 检查会话是否已存在
	if _, exists := vm.sessions[sessionID]; exists {
		return nil, fmt.Errorf("session already exists: %s", sessionID)
	}

	// 检查会话数量限制
	if len(vm.sessions) >= vm.maxSessions {
		return nil, fmt.Errorf("maximum sessions reached: %d", vm.maxSessions)
	}

	// 创建会话上下文
	ctx, cancel := context.WithCancel(vm.ctx)

	session := &VoiceSession{
		ID:           sessionID,
		UserID:       userID,
		TenantID:     tenantID,
		Config:       config,
		StartTime:    time.Now(),
		LastActivity: time.Now(),
		Status:       "active",
		Metadata:     make(map[string]interface{}),
		Context:      ctx,
		Cancel:       cancel,
	}

	vm.sessions[sessionID] = session
	vm.totalSessions++
	vm.activeSessions++

	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"user_id":    userID,
		"tenant_id":  tenantID,
		"config":     config,
	}).Info("Voice session created")

	return session, nil
}

// GetSession 获取语音会话
func (vm *VoiceManager) GetSession(sessionID string) (*VoiceSession, bool) {
	vm.sessionsMutex.RLock()
	defer vm.sessionsMutex.RUnlock()

	session, exists := vm.sessions[sessionID]
	return session, exists
}

// UpdateSessionActivity 更新会话活动时间
func (vm *VoiceManager) UpdateSessionActivity(sessionID string) {
	vm.sessionsMutex.RLock()
	session, exists := vm.sessions[sessionID]
	vm.sessionsMutex.RUnlock()

	if exists {
		session.Mutex.Lock()
		session.LastActivity = time.Now()
		session.Mutex.Unlock()
	}
}

// CloseSession 关闭语音会话
func (vm *VoiceManager) CloseSession(sessionID string) error {
	vm.sessionsMutex.Lock()
	session, exists := vm.sessions[sessionID]
	if exists {
		delete(vm.sessions, sessionID)
		vm.activeSessions--
	}
	vm.sessionsMutex.Unlock()

	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	// 关闭会话
	vm.closeSessionInternal(session)

	logrus.WithFields(logrus.Fields{
		"session_id": sessionID,
		"duration":   time.Since(session.StartTime),
	}).Info("Voice session closed")

	return nil
}

// closeSessionInternal 内部关闭会话
func (vm *VoiceManager) closeSessionInternal(session *VoiceSession) {
	// 取消上下文
	if session.Cancel != nil {
		session.Cancel()
	}

	// 关闭连接
	session.Mutex.Lock()
	if session.AlgoConn != nil {
		session.AlgoConn.Close()
		session.AlgoConn = nil
	}
	if session.ClientConn != nil {
		session.ClientConn.Close()
		session.ClientConn = nil
	}
	session.Status = "stopped"
	session.Mutex.Unlock()
}

// ProcessAudioData 处理音频数据
func (vm *VoiceManager) ProcessAudioData(sessionID string, audioData []byte) error {
	session, exists := vm.GetSession(sessionID)
	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	// 更新活动时间
	vm.UpdateSessionActivity(sessionID)

	// 增加处理计数
	vm.processedAudio++

	// 转发到算法服务
	session.Mutex.RLock()
	algoConn := session.AlgoConn
	session.Mutex.RUnlock()

	if algoConn == nil {
		return fmt.Errorf("algorithm service connection not available")
	}

	// 构建消息
	message := VoiceMessage{
		Type:      "voice_audio",
		SessionID: sessionID,
		Data:      string(audioData), // 这里应该是base64编码
		Timestamp: time.Now().Unix(),
	}

	return algoConn.WriteJSON(message)
}

// GetStats 获取统计信息
func (vm *VoiceManager) GetStats() map[string]interface{} {
	vm.sessionsMutex.RLock()
	defer vm.sessionsMutex.RUnlock()

	return map[string]interface{}{
		"total_sessions":   vm.totalSessions,
		"active_sessions":  vm.activeSessions,
		"processed_audio":  vm.processedAudio,
		"current_sessions": len(vm.sessions),
		"max_sessions":     vm.maxSessions,
		"session_timeout":  vm.sessionTimeout.String(),
	}
}

// ListSessions 列出所有会话
func (vm *VoiceManager) ListSessions() []*VoiceSession {
	vm.sessionsMutex.RLock()
	defer vm.sessionsMutex.RUnlock()

	sessions := make([]*VoiceSession, 0, len(vm.sessions))
	for _, session := range vm.sessions {
		sessions = append(sessions, session)
	}

	return sessions
}

// cleanupRoutine 清理例程
func (vm *VoiceManager) cleanupRoutine() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			vm.cleanupInactiveSessions()
		case <-vm.ctx.Done():
			return
		}
	}
}

// cleanupInactiveSessions 清理不活跃的会话
func (vm *VoiceManager) cleanupInactiveSessions() {
	vm.sessionsMutex.Lock()
	defer vm.sessionsMutex.Unlock()

	now := time.Now()
	toDelete := make([]string, 0)

	for sessionID, session := range vm.sessions {
		session.Mutex.RLock()
		inactive := now.Sub(session.LastActivity) > vm.sessionTimeout
		session.Mutex.RUnlock()

		if inactive {
			toDelete = append(toDelete, sessionID)
		}
	}

	// 删除不活跃的会话
	for _, sessionID := range toDelete {
		session := vm.sessions[sessionID]
		delete(vm.sessions, sessionID)
		vm.activeSessions--

		// 异步关闭会话
		go vm.closeSessionInternal(session)

		logrus.WithField("session_id", sessionID).Info("Cleaned up inactive voice session")
	}
}

// Shutdown 关闭语音管理器
func (vm *VoiceManager) Shutdown() {
	logrus.Info("Shutting down voice manager")

	// 取消上下文
	vm.cancel()

	// 关闭所有会话
	vm.sessionsMutex.Lock()
	for sessionID, session := range vm.sessions {
		vm.closeSessionInternal(session)
		delete(vm.sessions, sessionID)
	}
	vm.activeSessions = 0
	vm.sessionsMutex.Unlock()

	logrus.Info("Voice manager shutdown completed")
}

// 全局语音管理器实例
var (
	globalVoiceManager *VoiceManager
	voiceManagerOnce   sync.Once
)

// GetGlobalVoiceManager 获取全局语音管理器
func GetGlobalVoiceManager() *VoiceManager {
	voiceManagerOnce.Do(func() {
		globalVoiceManager = NewVoiceManager("http://localhost:8000")
	})
	return globalVoiceManager
}

// InitializeVoiceManager 初始化语音管理器
func InitializeVoiceManager(algoServiceURL string) *VoiceManager {
	globalVoiceManager = NewVoiceManager(algoServiceURL)
	return globalVoiceManager
}
