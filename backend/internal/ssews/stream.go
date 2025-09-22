package ssews

import (
	"time"
)

// 统一消息信封
type Envelope struct {
	Meta  map[string]interface{} `json:"meta,omitempty"`
	Data  interface{}            `json:"data,omitempty"`
	Error *ErrorInfo             `json:"error,omitempty"`
}

type ErrorInfo struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// 流式写入器接口
type StreamWriter interface {
	WriteEvent(event string, payload interface{}) error
	WriteError(code, message string) error
	Close() error
}

// 基础流处理器
type BaseStreamHandler struct {
	Writer   StreamWriter
	TraceID  string
	TenantID string
}

func (h *BaseStreamHandler) WriteEnvelope(event string, data interface{}) error {
	envelope := Envelope{
		Meta: map[string]interface{}{
			"trace_id":  h.TraceID,
			"tenant_id": h.TenantID,
			"timestamp": time.Now().Unix(),
		},
		Data: data,
	}
	return h.Writer.WriteEvent(event, envelope)
}

func (h *BaseStreamHandler) WriteErrorEnvelope(code, message string) error {
	envelope := Envelope{
		Meta: map[string]interface{}{
			"trace_id":  h.TraceID,
			"tenant_id": h.TenantID,
			"timestamp": time.Now().Unix(),
		},
		Error: &ErrorInfo{
			Code:    code,
			Message: message,
		},
	}
	return h.Writer.WriteEvent("error", envelope)
}
