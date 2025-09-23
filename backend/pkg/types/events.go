package types

import (
	"encoding/json"
	"time"
)

// EventType 事件类型常量
const (
	EventTypeConnected    = "connected"
	EventTypeDisconnected = "disconnected"
	EventTypeHeartbeat    = "heartbeat"
	EventTypeThrottle     = "throttle"
	EventTypeCancel       = "cancel"
	EventTypeAudioFrame   = "audio_frame"
	EventTypeTTSChunk     = "tts_chunk"
	EventTypeASRPartial   = "asr_partial"
	EventTypeASRFinal     = "asr_final"
	EventTypeLLMDelta     = "llm_delta"
	EventTypeLLMDone      = "llm_done"
	EventTypeReferences   = "references"
	EventTypeError        = "error"
	EventTypeAgentPlan    = "agent_plan"
	EventTypeAgentStep    = "agent_step"
	EventTypeToolResult   = "tool_result"
	EventTypeAgentSummary = "agent_summary"
)

// EventMeta 事件元数据
type EventMeta struct {
	TraceID   string    `json:"trace_id"`
	SessionID string    `json:"session_id"`
	Timestamp time.Time `json:"timestamp"`
	Version   string    `json:"version"`
}

// EventEnvelope 事件信封
type EventEnvelope struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
	Meta EventMeta   `json:"meta"`
}

// NewEventEnvelope 创建事件信封
func NewEventEnvelope(eventType string, data interface{}, sessionID, traceID string) *EventEnvelope {
	return &EventEnvelope{
		Type: eventType,
		Data: data,
		Meta: EventMeta{
			TraceID:   traceID,
			SessionID: sessionID,
			Timestamp: time.Now(),
			Version:   "1.0",
		},
	}
}

// ToJSON 序列化为JSON
func (e *EventEnvelope) ToJSON() ([]byte, error) {
	return json.Marshal(e)
}

// EventFromJSON 从JSON反序列化
func EventFromJSON(data []byte) (*EventEnvelope, error) {
	var event EventEnvelope
	err := json.Unmarshal(data, &event)
	return &event, err
}

// 具体事件数据结构

// AudioFrameData 音频帧数据
type AudioFrameData struct {
	Audio       []byte `json:"audio"`
	Timestamp   int64  `json:"timestamp"`
	SequenceNum int32  `json:"sequence_num"`
	SampleRate  int32  `json:"sample_rate"`
	Channels    int8   `json:"channels"`
	FrameSize   int16  `json:"frame_size"`
}

// TTSChunkData TTS音频块数据
type TTSChunkData struct {
	Audio     []byte `json:"audio"`
	Timestamp int64  `json:"timestamp"`
	Format    string `json:"format"`
	IsLast    bool   `json:"is_last"`
}

// ASRData ASR识别数据
type ASRData struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	IsFinal    bool    `json:"is_final"`
	Language   string  `json:"language"`
}

// LLMDeltaData LLM流式响应数据
type LLMDeltaData struct {
	Content string `json:"content"`
	Role    string `json:"role"`
	Delta   string `json:"delta"`
}

// ThrottleData 限流数据
type ThrottleData struct {
	Reason       string `json:"reason"`
	RetryAfterMs int    `json:"retry_after_ms"`
	CurrentRate  int32  `json:"current_rate"`
	MaxRate      int32  `json:"max_rate"`
}

// HeartbeatData 心跳数据
type HeartbeatData struct {
	Timestamp int64 `json:"timestamp"`
	Sequence  int32 `json:"sequence"`
}

// ErrorData 错误数据
type ErrorData struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details"`
}

// ReferenceData 引用数据
type ReferenceData struct {
	ID       string                 `json:"id"`
	Source   string                 `json:"source"`
	Content  string                 `json:"content"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata"`
}

// AgentPlanData Agent规划数据
type AgentPlanData struct {
	Steps       []string `json:"steps"`
	Reasoning   string   `json:"reasoning"`
	Tools       []string `json:"tools"`
	EstimatedMs int      `json:"estimated_ms"`
}

// AgentStepData Agent步骤数据
type AgentStepData struct {
	StepIndex   int    `json:"step_index"`
	Description string `json:"description"`
	Status      string `json:"status"`
	StartTime   int64  `json:"start_time"`
	EndTime     int64  `json:"end_time"`
}

// ToolResultData 工具结果数据
type ToolResultData struct {
	ToolName string      `json:"tool_name"`
	Input    interface{} `json:"input"`
	Output   interface{} `json:"output"`
	Success  bool        `json:"success"`
	Error    string      `json:"error,omitempty"`
	Duration int64       `json:"duration_ms"`
}

// AudioHeader 音频帧头部
type AudioHeader struct {
	SessionID   string `json:"session_id"`
	SequenceNum int32  `json:"sequence_num"`
	SampleRate  int32  `json:"sample_rate"`
	Channels    int8   `json:"channels"`
	FrameSize   int16  `json:"frame_size"`
	Timestamp   int64  `json:"timestamp"`
	Format      string `json:"format"`
}
