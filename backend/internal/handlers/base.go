package handlers

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/gin-gonic/gin"
)

// ErrorInfo 错误信息结构
type ErrorInfo struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type BaseHandler struct {
	AlgoServiceURL string
}

func (h *BaseHandler) extractTraceInfo(c *gin.Context) (traceID, tenantID string) {
	traceID = c.GetHeader("X-Trace-ID")
	if traceID == "" {
		traceID = h.generateTraceID()
	}

	tenantID = c.GetHeader("X-Tenant-ID")
	if tenantID == "" {
		tenantID = "default"
	}

	return
}

func (h *BaseHandler) generateTraceID() string {
	bytes := make([]byte, 8)
	rand.Read(bytes)
	return fmt.Sprintf("trace_%s", hex.EncodeToString(bytes))
}

// 统一消息信封
type Envelope struct {
	Meta  map[string]interface{} `json:"meta,omitempty"`
	Data  interface{}            `json:"data,omitempty"`
	Error *ErrorInfo             `json:"error,omitempty"`
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

// WriteEnvelope 写入信封消息
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

// WriteErrorEnvelope 写入错误信封
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

func (h *BaseHandler) createStreamHandler(writer StreamWriter, traceID, tenantID string) *BaseStreamHandler {
	return &BaseStreamHandler{
		Writer:   writer,
		TraceID:  traceID,
		TenantID: tenantID,
	}
}
