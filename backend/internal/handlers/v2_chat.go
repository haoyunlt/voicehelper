package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
	"voicehelper/backend/internal/ssews"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

type V2ChatHandler struct {
	BaseHandler
}

type ChatRequest struct {
	Query     string                 `json:"query"`
	SessionID string                 `json:"session_id"`
	Context   map[string]interface{} `json:"context,omitempty"`
}

type CancelRequest struct {
	SessionID string `json:"session_id"`
}

// SSEWriter SSE写入器
type SSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	closed  bool
}

// ErrorInfo 错误信息
type ErrorInfo struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// NewSSEWriter 创建SSE写入器
func NewSSEWriter(w http.ResponseWriter) *SSEWriter {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")

	return &SSEWriter{w: w, flusher: flusher}
}

// WriteEvent 写入事件
func (s *SSEWriter) WriteEvent(event string, payload interface{}) error {
	if s.closed {
		return fmt.Errorf("writer closed")
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	fmt.Fprintf(s.w, "event: %s\n", event)
	fmt.Fprintf(s.w, "data: %s\n\n", data)
	s.flusher.Flush()

	return nil
}

// WriteError 写入错误
func (s *SSEWriter) WriteError(code, message string) error {
	return s.WriteEvent("error", ErrorInfo{Code: code, Message: message})
}

// Close 关闭写入器
func (s *SSEWriter) Close() error {
	s.closed = true
	return nil
}

func NewV2ChatHandler(algoServiceURL string) *V2ChatHandler {
	return &V2ChatHandler{
		BaseHandler: BaseHandler{
			AlgoServiceURL: algoServiceURL,
		},
	}
}

func (h *V2ChatHandler) StreamChat(c *gin.Context) {
	// 创建 SSE 写入器
	writer := NewSSEWriter(c.Writer)
	if writer == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "SSE not supported"})
		return
	}
	defer writer.Close()

	// 提取追踪信息
	traceID, tenantID := h.extractTraceInfo(c)
	streamHandler := h.createStreamHandler(writer, traceID, tenantID)

	// 解析请求
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		streamHandler.WriteErrorEnvelope("invalid_request", err.Error())
		return
	}

	// 验证请求
	if req.Query == "" {
		streamHandler.WriteErrorEnvelope("empty_query", "Query cannot be empty")
		return
	}

	if req.SessionID == "" {
		streamHandler.WriteErrorEnvelope("missing_session_id", "Session ID is required")
		return
	}

	logrus.WithFields(logrus.Fields{
		"trace_id":   traceID,
		"tenant_id":  tenantID,
		"session_id": req.SessionID,
		"query":      req.Query[:min(len(req.Query), 100)],
	}).Info("开始处理聊天请求")

	// 转发到算法服务
	h.forwardToAlgoService(streamHandler, req)
}

func (h *V2ChatHandler) CancelChat(c *gin.Context) {
	var req CancelRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	if req.SessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	// TODO: 实现取消逻辑，向算法服务发送取消请求
	logrus.WithField("session_id", req.SessionID).Info("取消聊天会话")

	c.JSON(http.StatusOK, gin.H{
		"status":     "cancelled",
		"session_id": req.SessionID,
	})
}

func (h *V2ChatHandler) forwardToAlgoService(handler *ssews.BaseStreamHandler, req ChatRequest) {
	// 构建到算法服务的请求
	reqBody, err := json.Marshal(req)
	if err != nil {
		handler.WriteErrorEnvelope("marshal_error", err.Error())
		return
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v1/chat/stream", h.AlgoServiceURL)
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		handler.WriteErrorEnvelope("request_error", err.Error())
		return
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Trace-ID", handler.TraceID)
	httpReq.Header.Set("X-Tenant-ID", handler.TenantID)

	// 发送请求
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		handler.WriteErrorEnvelope("algo_service_error", err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		handler.WriteErrorEnvelope("algo_service_error", fmt.Sprintf("Status: %d", resp.StatusCode))
		return
	}

	// 转发SSE流
	h.forwardSSEStream(handler, resp.Body)
}

func (h *V2ChatHandler) forwardSSEStream(handler *ssews.BaseStreamHandler, body io.Reader) {
	buf := make([]byte, 4096)
	var eventBuffer []byte

	for {
		n, err := body.Read(buf)
		if n > 0 {
			eventBuffer = append(eventBuffer, buf[:n]...)

			// 处理SSE事件
			for {
				lineEnd := bytes.Index(eventBuffer, []byte("\n\n"))
				if lineEnd == -1 {
					break
				}

				eventData := eventBuffer[:lineEnd]
				eventBuffer = eventBuffer[lineEnd+2:]

				h.processSSEEvent(handler, eventData)
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			logrus.WithError(err).Error("读取算法服务响应失败")
			handler.WriteErrorEnvelope("stream_error", err.Error())
			break
		}
	}

	// 处理剩余数据
	if len(eventBuffer) > 0 {
		h.processSSEEvent(handler, eventBuffer)
	}
}

func (h *V2ChatHandler) processSSEEvent(handler *ssews.BaseStreamHandler, eventData []byte) {
	lines := bytes.Split(eventData, []byte("\n"))
	var event string
	var data []byte

	for _, line := range lines {
		if bytes.HasPrefix(line, []byte("event: ")) {
			event = string(line[7:])
		} else if bytes.HasPrefix(line, []byte("data: ")) {
			data = line[6:]
		}
	}

	if len(data) > 0 {
		// 解析数据并重新封装
		var payload interface{}
		if err := json.Unmarshal(data, &payload); err != nil {
			logrus.WithError(err).Error("解析SSE数据失败")
			return
		}

		// 转发事件
		if event == "" {
			event = "message"
		}
		handler.WriteEnvelope(event, payload)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
