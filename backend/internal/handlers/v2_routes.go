package handlers

import (
	"os"

	"github.com/gin-gonic/gin"
)

// SetupV2Routes 设置V2架构路由
func SetupV2Routes(router *gin.Engine) {
	// 获取算法服务URL
	algoServiceURL := os.Getenv("ALGO_SERVICE_URL")
	if algoServiceURL == "" {
		algoServiceURL = "http://localhost:8070"
	}

	// 创建处理器
	chatHandler := NewV2ChatHandlerSimple(algoServiceURL)
	voiceHandler := NewV2VoiceHandler(algoServiceURL)
	voiceWSHandler := NewVoiceWSHandler(algoServiceURL)
	webrtcHandler := NewWebRTCSignalingHandler(algoServiceURL)

	// 启动语音处理器的清理例程
	voiceHandler.StartCleanupRoutine()

	// V2 API路由组
	v2 := router.Group("/api/v2")
	{
		// 聊天相关路由
		chat := v2.Group("/chat")
		{
			chat.POST("/stream", chatHandler.StreamChat)
			chat.POST("/cancel", chatHandler.CancelChat)
		}

		// 语音相关路由
		voice := v2.Group("/voice")
		{
			voice.GET("/stream", voiceHandler.HandleWebSocket)
			voice.GET("/ws", voiceWSHandler.HandleVoiceWebSocket)
		}

		// WebRTC信令路由
		webrtc := v2.Group("/webrtc")
		{
			webrtc.GET("/signaling", webrtcHandler.HandleWebRTCSignaling)
		}
	}

	// 健康检查
	v2.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":  "healthy",
			"version": "2.0.0",
			"service": "voicehelper-gateway",
		})
	})
}
