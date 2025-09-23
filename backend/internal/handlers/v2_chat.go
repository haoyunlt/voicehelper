package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

type V2ChatHandlerSimple struct {
	BaseHandler
}

type ChatRequest struct {
	Query     string                 `json:"query"`
	SessionID string                 `json:"session_id,omitempty"`
	UserID    string                 `json:"user_id,omitempty"`
	Context   map[string]interface{} `json:"context,omitempty"`
}

// AlgoChatRequest 算法服务聊天请求
type AlgoChatRequest struct {
	Query     string                 `json:"query"`
	SessionID string                 `json:"session_id"`
	Context   map[string]interface{} `json:"context,omitempty"`
}

type ChatResponse struct {
	Status    string      `json:"status"`
	Message   string      `json:"message,omitempty"`
	Data      interface{} `json:"data,omitempty"`
	SessionID string      `json:"session_id,omitempty"`
}

// SSEEvent SSE事件结构
type SSEEvent struct {
	Event string      `json:"event"`
	Data  interface{} `json:"data"`
}

// SSEEventData SSE事件数据
type SSEEventData struct {
	Meta map[string]interface{} `json:"meta,omitempty"`
	Data interface{}            `json:"data,omitempty"`
}

func NewV2ChatHandlerSimple(algoServiceURL string) *V2ChatHandlerSimple {
	return &V2ChatHandlerSimple{
		BaseHandler: BaseHandler{
			AlgoServiceURL: algoServiceURL,
		},
	}
}

// HandleChatQuery 处理聊天查询
func (h *V2ChatHandlerSimple) HandleChatQuery(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ChatResponse{
			Status:  "error",
			Message: "Invalid request format: " + err.Error(),
		})
		return
	}

	traceID, tenantID := h.extractTraceInfo(c)

	logrus.WithFields(logrus.Fields{
		"trace_id":   traceID,
		"tenant_id":  tenantID,
		"session_id": req.SessionID,
		"query":      req.Query,
	}).Info("处理聊天查询")

	// 调用算法服务
	response, err := h.callAlgoService(req, traceID)
	if err != nil {
		logrus.WithError(err).Error("算法服务调用失败")
		c.JSON(http.StatusInternalServerError, ChatResponse{
			Status:  "error",
			Message: "算法服务调用失败: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, response)
}

// HandleChatStream 处理聊天流
func (h *V2ChatHandlerSimple) HandleChatStream(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ChatResponse{
			Status:  "error",
			Message: "Invalid request format: " + err.Error(),
		})
		return
	}

	traceID, tenantID := h.extractTraceInfo(c)

	logrus.WithFields(logrus.Fields{
		"trace_id":   traceID,
		"tenant_id":  tenantID,
		"session_id": req.SessionID,
		"query":      req.Query,
	}).Info("处理流式聊天")

	// 设置SSE头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	// 调用算法服务流式接口
	err := h.streamFromAlgoService(c, req, traceID, tenantID)
	if err != nil {
		logrus.WithError(err).Error("流式聊天处理失败")
		// 发送错误事件
		h.sendSSEEvent(c, "error", map[string]interface{}{
			"error":   "stream_failed",
			"message": err.Error(),
		}, traceID, tenantID)
	}
}

// StreamChat 处理流式聊天
func (h *V2ChatHandlerSimple) StreamChat(c *gin.Context) {
	h.HandleChatStream(c)
}

// CancelChat 取消聊天
func (h *V2ChatHandlerSimple) CancelChat(c *gin.Context) {
	var req map[string]string
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ChatResponse{
			Status:  "error",
			Message: "Invalid request format: " + err.Error(),
		})
		return
	}

	sessionID := req["session_id"]
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, ChatResponse{
			Status:  "error",
			Message: "缺少session_id",
		})
		return
	}

	traceID, _ := h.extractTraceInfo(c)

	logrus.WithFields(logrus.Fields{
		"trace_id":   traceID,
		"session_id": sessionID,
	}).Info("取消聊天会话")

	// 调用算法服务取消接口
	err := h.cancelAlgoServiceChat(sessionID, traceID)
	if err != nil {
		logrus.WithError(err).Error("取消聊天失败")
		c.JSON(http.StatusInternalServerError, ChatResponse{
			Status:  "error",
			Message: "取消聊天失败: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, ChatResponse{
		Status:    "success",
		Message:   "Chat cancelled successfully",
		SessionID: sessionID,
	})
}

// callAlgoService 调用算法服务
func (h *V2ChatHandlerSimple) callAlgoService(req ChatRequest, traceID string) (*ChatResponse, error) {
	// 构建算法服务请求
	algoReq := AlgoChatRequest{
		Query:     req.Query,
		SessionID: req.SessionID,
		Context:   req.Context,
	}

	reqBody, err := json.Marshal(algoReq)
	if err != nil {
		return nil, fmt.Errorf("序列化请求失败: %w", err)
	}

	// 发送HTTP请求到算法服务
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	url := fmt.Sprintf("%s/api/v1/chat/stream", h.AlgoServiceURL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("请求算法服务失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("算法服务返回错误: %d, %s", resp.StatusCode, string(body))
	}

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
	}

	// 解析响应（简化处理，实际应该解析SSE流）
	return &ChatResponse{
		Status:    "success",
		Message:   "Chat completed",
		Data:      map[string]interface{}{"response": string(body)},
		SessionID: req.SessionID,
	}, nil
}

// streamFromAlgoService 从算法服务流式获取响应
func (h *V2ChatHandlerSimple) streamFromAlgoService(c *gin.Context, req ChatRequest, traceID, tenantID string) error {
	// 构建算法服务请求
	algoReq := AlgoChatRequest{
		Query:     req.Query,
		SessionID: req.SessionID,
		Context:   req.Context,
	}

	reqBody, err := json.Marshal(algoReq)
	if err != nil {
		return fmt.Errorf("序列化请求失败: %w", err)
	}

	// 创建HTTP客户端
	client := &http.Client{
		Timeout: 60 * time.Second,
	}

	url := fmt.Sprintf("%s/api/v1/chat/stream", h.AlgoServiceURL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("请求算法服务失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("算法服务返回错误: %d, %s", resp.StatusCode, string(body))
	}

	// 发送开始事件
	h.sendSSEEvent(c, "start", map[string]interface{}{
		"session_id": req.SessionID,
		"status":     "started",
	}, traceID, tenantID)

	// 读取SSE流并转发
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			dataStr := strings.TrimPrefix(line, "data: ")
			if dataStr == "[DONE]" {
				break
			}

			// 解析事件数据
			var eventData map[string]interface{}
			if err := json.Unmarshal([]byte(dataStr), &eventData); err != nil {
				logrus.WithError(err).Warn("解析SSE事件失败")
				continue
			}

			// 转发事件
			h.sendSSEEvent(c, "message", eventData, traceID, tenantID)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("读取流失败: %w", err)
	}

	// 发送完成事件
	h.sendSSEEvent(c, "done", map[string]interface{}{
		"session_id": req.SessionID,
		"status":     "completed",
	}, traceID, tenantID)

	return nil
}

// cancelAlgoServiceChat 取消算法服务聊天
func (h *V2ChatHandlerSimple) cancelAlgoServiceChat(sessionID, traceID string) error {
	reqBody, err := json.Marshal(map[string]string{"session_id": sessionID})
	if err != nil {
		return fmt.Errorf("序列化请求失败: %w", err)
	}

	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	url := fmt.Sprintf("%s/api/v1/chat/cancel", h.AlgoServiceURL)
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("请求算法服务失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("算法服务返回错误: %d, %s", resp.StatusCode, string(body))
	}

	return nil
}

// sendSSEEvent 发送SSE事件
func (h *V2ChatHandlerSimple) sendSSEEvent(c *gin.Context, event string, data interface{}, traceID, tenantID string) {
	eventData := SSEEventData{
		Meta: map[string]interface{}{
			"trace_id":  traceID,
			"tenant_id": tenantID,
			"timestamp": time.Now().Unix(),
		},
		Data: data,
	}

	c.SSEvent(event, eventData)
	c.Writer.Flush()
}
