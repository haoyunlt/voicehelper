package handler

import (
	"chatbot-backend/internal/service"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// ChatStream 流式对话接口
func (h *Handlers) ChatStream(c *gin.Context) {
	var req service.ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 设置SSE头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	// 调用对话服务
	responseCh, err := h.services.ChatService.StreamChat(c.Request.Context(), &req)
	if err != nil {
		logrus.WithError(err).Error("Failed to start chat stream")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	// 流式返回响应
	c.Stream(func(w gin.ResponseWriter) bool {
		select {
		case response, ok := <-responseCh:
			if !ok {
				// 通道关闭，发送结束事件
				fmt.Fprintf(w, "event: end\ndata: {}\n\n")
				return false
			}

			// 序列化响应
			data, err := json.Marshal(response)
			if err != nil {
				logrus.WithError(err).Error("Failed to marshal response")
				return false
			}

			// 发送SSE事件
			fmt.Fprintf(w, "data: %s\n\n", data)
			return true

		case <-c.Request.Context().Done():
			// 客户端断开连接
			return false
		}
	})
}
