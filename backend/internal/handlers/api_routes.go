package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"voicehelper/backend/pkg/middleware"
)

// APIHandler API路由处理器
type APIHandler struct {
	chatSSEHandler   *ChatSSEHandler
	voiceWSHandler   *VoiceWSHandler
	authMiddleware   *middleware.AuthMiddleware
	rbacMiddleware   *middleware.RBACMiddleware
	tenantMiddleware *middleware.TenantMiddleware
}

// NewAPIHandler 创建API处理器
func NewAPIHandler(
	chatSSE *ChatSSEHandler,
	voiceWS *VoiceWSHandler,
	auth *middleware.AuthMiddleware,
	rbac *middleware.RBACMiddleware,
	tenant *middleware.TenantMiddleware,
) *APIHandler {
	return &APIHandler{
		chatSSEHandler:   chatSSE,
		voiceWSHandler:   voiceWS,
		authMiddleware:   auth,
		rbacMiddleware:   rbac,
		tenantMiddleware: tenant,
	}
}

// SetupRoutes 设置路由
func (h *APIHandler) SetupRoutes(r *gin.Engine) {
	// 健康检查（无需认证）
	r.GET("/health", h.healthCheck)
	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// API v1 路由组
	v1 := r.Group("/api/v1")
	{
		// 认证相关（无需JWT）
		auth := v1.Group("/auth")
		{
			auth.POST("/wechat/miniprogram/login", h.wechatMiniProgramLogin)
			auth.POST("/refresh", h.refreshToken)
			auth.POST("/logout", h.logout)
		}

		// 需要认证的路由
		protected := v1.Group("")
		protected.Use(h.authMiddleware.Handle())
		protected.Use(h.tenantMiddleware.Handle())
		{
			// 聊天相关
			chat := protected.Group("/chat")
			{
				chat.GET("/stream", h.adaptHTTPHandler(h.chatSSEHandler.HandleStream))
				chat.POST("/stream", h.adaptHTTPHandler(h.chatSSEHandler.ProcessChatRequest))
				chat.POST("/cancel", h.cancelChat)
				chat.GET("/stats/:stream_id", h.getChatStats)
			}

			// 语音相关
			voice := protected.Group("/voice")
			{
				voice.GET("/stream", h.adaptHTTPHandler(h.voiceWSHandler.HandleConnection))
				voice.POST("/transcribe", h.transcribeAudio)
				voice.POST("/synthesize", h.synthesizeText)
				voice.GET("/stats/:session_id", h.getVoiceStats)
			}

			// 检索相关
			search := protected.Group("/search")
			{
				search.POST("", h.searchDocuments)
				search.GET("/suggestions", h.getSearchSuggestions)
			}

			// 文档管理
			docs := protected.Group("/documents")
			docs.Use(h.rbacMiddleware.RequirePermission("document:read"))
			{
				docs.GET("", h.listDocuments)
				docs.GET("/:id", h.getDocument)

				// 需要写权限
				writeGroup := docs.Group("")
				writeGroup.Use(h.rbacMiddleware.RequirePermission("document:write"))
				{
					writeGroup.POST("", h.uploadDocument)
					writeGroup.PUT("/:id", h.updateDocument)
					writeGroup.DELETE("/:id", h.deleteDocument)
				}
			}

			// 会话管理
			conversations := protected.Group("/conversations")
			{
				conversations.GET("", h.listConversations)
				conversations.POST("", h.createConversation)
				conversations.GET("/:id", h.getConversation)
				conversations.PUT("/:id", h.updateConversation)
				conversations.DELETE("/:id", h.deleteConversation)
				conversations.GET("/:id/messages", h.getConversationMessages)
			}

			// Agent相关
			agent := protected.Group("/agent")
			{
				agent.GET("/stream", h.agentStream)
				agent.POST("/tools/execute", h.executeAgentTool)
				agent.GET("/capabilities", h.getAgentCapabilities)
			}

			// 管理接口（需要管理员权限）
			admin := protected.Group("/admin")
			admin.Use(h.rbacMiddleware.RequirePermission("admin:*"))
			{
				admin.GET("/stats", h.getSystemStats)
				admin.GET("/sessions", h.getActiveSessions)
				admin.POST("/reload", h.reloadConfiguration)
				admin.POST("/maintenance", h.setMaintenanceMode)
			}
		}
	}
}

// 健康检查
func (h *APIHandler) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"version":   "1.0.0",
		"services": gin.H{
			"chat_sse": "active",
			"voice_ws": "active",
		},
	})
}

// 微信小程序登录
func (h *APIHandler) wechatMiniProgramLogin(c *gin.Context) {
	var req struct {
		Code     string                 `json:"code" binding:"required"`
		UserInfo map[string]interface{} `json:"user_info"`
		TenantID string                 `json:"tenant_id"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: 实现微信登录逻辑
	// 1. 使用code换取openid和session_key
	// 2. 验证用户信息
	// 3. 生成JWT token

	c.JSON(http.StatusOK, gin.H{
		"token": "mock_jwt_token",
		"user": gin.H{
			"id":        "user_123",
			"openid":    "mock_openid",
			"tenant_id": req.TenantID,
		},
	})
}

// 刷新Token
func (h *APIHandler) refreshToken(c *gin.Context) {
	// TODO: 实现token刷新逻辑
	c.JSON(http.StatusOK, gin.H{
		"token": "new_jwt_token",
	})
}

// 登出
func (h *APIHandler) logout(c *gin.Context) {
	tokenString := c.GetHeader("Authorization")
	if tokenString != "" {
		// 将token加入黑名单
		h.authMiddleware.RevokeToken(tokenString)
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "Logged out successfully",
	})
}

// 取消聊天
func (h *APIHandler) cancelChat(c *gin.Context) {
	var req struct {
		StreamID string `json:"stream_id" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: 实现取消聊天逻辑
	c.JSON(http.StatusOK, gin.H{
		"message":   "Chat cancelled",
		"stream_id": req.StreamID,
	})
}

// 获取聊天统计
func (h *APIHandler) getChatStats(c *gin.Context) {
	streamID := c.Param("stream_id")
	stats := h.chatSSEHandler.GetStreamStats(streamID)

	if stats == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Stream not found"})
		return
	}

	c.JSON(http.StatusOK, stats)
}

// 语音转写
func (h *APIHandler) transcribeAudio(c *gin.Context) {
	// TODO: 实现语音转写逻辑
	c.JSON(http.StatusOK, gin.H{
		"text":       "这是转写结果",
		"confidence": 0.95,
		"language":   "zh-CN",
	})
}

// 语音合成
func (h *APIHandler) synthesizeText(c *gin.Context) {
	var req struct {
		Text   string `json:"text" binding:"required"`
		Voice  string `json:"voice"`
		Format string `json:"format"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: 实现语音合成逻辑
	c.JSON(http.StatusOK, gin.H{
		"audio_url": "https://example.com/audio.mp3",
		"duration":  3.5,
		"format":    req.Format,
	})
}

// 获取语音统计
func (h *APIHandler) getVoiceStats(c *gin.Context) {
	sessionID := c.Param("session_id")
	stats := h.voiceWSHandler.GetSessionStats(sessionID)

	if stats == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	c.JSON(http.StatusOK, stats)
}

// 搜索文档
func (h *APIHandler) searchDocuments(c *gin.Context) {
	var req struct {
		Query  string                 `json:"query" binding:"required"`
		TopK   int                    `json:"top_k"`
		Filter map[string]interface{} `json:"filter"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: 实现文档搜索逻辑
	c.JSON(http.StatusOK, gin.H{
		"results": []gin.H{
			{
				"id":      "doc_1",
				"title":   "示例文档",
				"content": "这是搜索结果",
				"score":   0.95,
			},
		},
		"total": 1,
	})
}

// 获取搜索建议
func (h *APIHandler) getSearchSuggestions(c *gin.Context) {
	query := c.Query("q")

	// TODO: 实现搜索建议逻辑
	c.JSON(http.StatusOK, gin.H{
		"suggestions": []string{
			query + "相关建议1",
			query + "相关建议2",
		},
	})
}

// 文档管理相关方法
func (h *APIHandler) listDocuments(c *gin.Context) {
	// TODO: 实现文档列表逻辑
	c.JSON(http.StatusOK, gin.H{
		"documents": []gin.H{},
		"total":     0,
	})
}

func (h *APIHandler) getDocument(c *gin.Context) {
	id := c.Param("id")
	// TODO: 实现获取文档逻辑
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"title":   "文档标题",
		"content": "文档内容",
	})
}

func (h *APIHandler) uploadDocument(c *gin.Context) {
	// TODO: 实现文档上传逻辑
	c.JSON(http.StatusCreated, gin.H{
		"id":      "new_doc_id",
		"message": "Document uploaded successfully",
	})
}

func (h *APIHandler) updateDocument(c *gin.Context) {
	id := c.Param("id")
	// TODO: 实现文档更新逻辑
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"message": "Document updated successfully",
	})
}

func (h *APIHandler) deleteDocument(c *gin.Context) {
	id := c.Param("id")
	// TODO: 实现文档删除逻辑
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"message": "Document deleted successfully",
	})
}

// 会话管理相关方法
func (h *APIHandler) listConversations(c *gin.Context) {
	// TODO: 实现会话列表逻辑
	c.JSON(http.StatusOK, gin.H{
		"conversations": []gin.H{},
		"total":         0,
	})
}

func (h *APIHandler) createConversation(c *gin.Context) {
	// TODO: 实现创建会话逻辑
	c.JSON(http.StatusCreated, gin.H{
		"id":      "new_conversation_id",
		"message": "Conversation created successfully",
	})
}

func (h *APIHandler) getConversation(c *gin.Context) {
	id := c.Param("id")
	// TODO: 实现获取会话逻辑
	c.JSON(http.StatusOK, gin.H{
		"id":    id,
		"title": "会话标题",
	})
}

func (h *APIHandler) updateConversation(c *gin.Context) {
	id := c.Param("id")
	// TODO: 实现更新会话逻辑
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"message": "Conversation updated successfully",
	})
}

func (h *APIHandler) deleteConversation(c *gin.Context) {
	id := c.Param("id")
	// TODO: 实现删除会话逻辑
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"message": "Conversation deleted successfully",
	})
}

func (h *APIHandler) getConversationMessages(c *gin.Context) {
	id := c.Param("id")
	// TODO: 实现获取会话消息逻辑
	c.JSON(http.StatusOK, gin.H{
		"conversation_id": id,
		"messages":        []gin.H{},
		"total":           0,
	})
}

// Agent相关方法
func (h *APIHandler) agentStream(c *gin.Context) {
	// TODO: 实现Agent流式处理
	c.JSON(http.StatusOK, gin.H{
		"message": "Agent stream endpoint",
	})
}

func (h *APIHandler) executeAgentTool(c *gin.Context) {
	// TODO: 实现Agent工具执行
	c.JSON(http.StatusOK, gin.H{
		"result": "Tool execution result",
	})
}

func (h *APIHandler) getAgentCapabilities(c *gin.Context) {
	// TODO: 实现获取Agent能力
	c.JSON(http.StatusOK, gin.H{
		"capabilities": []string{
			"reasoning",
			"planning",
			"tool_use",
			"memory",
		},
	})
}

// 管理接口
func (h *APIHandler) getSystemStats(c *gin.Context) {
	chatStats := h.chatSSEHandler.GetAllStreamsStats()

	c.JSON(http.StatusOK, gin.H{
		"chat_streams": chatStats,
		"voice_sessions": gin.H{
			"total": 0, // TODO: 从voiceWSHandler获取
		},
		"system": gin.H{
			"uptime":       time.Since(time.Now()).Seconds(),
			"memory_usage": "TODO",
			"cpu_usage":    "TODO",
		},
	})
}

func (h *APIHandler) getActiveSessions(c *gin.Context) {
	// TODO: 实现获取活跃会话
	c.JSON(http.StatusOK, gin.H{
		"sessions": []gin.H{},
		"total":    0,
	})
}

func (h *APIHandler) reloadConfiguration(c *gin.Context) {
	// TODO: 实现配置重载
	c.JSON(http.StatusOK, gin.H{
		"message": "Configuration reloaded successfully",
	})
}

func (h *APIHandler) setMaintenanceMode(c *gin.Context) {
	var req struct {
		Enabled bool   `json:"enabled"`
		Message string `json:"message"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// TODO: 实现维护模式设置
	c.JSON(http.StatusOK, gin.H{
		"maintenance_mode": req.Enabled,
		"message":          req.Message,
	})
}

// adaptHTTPHandler 将标准HTTP处理器适配为Gin处理器
func (h *APIHandler) adaptHTTPHandler(handler func(http.ResponseWriter, *http.Request)) gin.HandlerFunc {
	return gin.WrapH(http.HandlerFunc(handler))
}
