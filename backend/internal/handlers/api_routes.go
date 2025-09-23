package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"voicehelper/backend/common/logger"
	"voicehelper/backend/internal/repository"
	"voicehelper/backend/pkg/middleware"
)

// APIHandler API路由处理器
type APIHandler struct {
	authMiddleware   *middleware.AuthMiddleware
	rbacMiddleware   *middleware.RBACMiddleware
	tenantMiddleware *middleware.TenantMiddleware
	conversationRepo repository.ConversationRepository
	algoServiceURL   string
}

// NewAPIHandler 创建API处理器
func NewAPIHandler(
	auth *middleware.AuthMiddleware,
	rbac *middleware.RBACMiddleware,
	tenant *middleware.TenantMiddleware,
	conversationRepo repository.ConversationRepository,
	algoServiceURL string,
) *APIHandler {
	return &APIHandler{
		authMiddleware:   auth,
		rbacMiddleware:   rbac,
		tenantMiddleware: tenant,
		conversationRepo: conversationRepo,
		algoServiceURL:   algoServiceURL,
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
			// 聊天相关 - 使用V2版本
			chat := protected.Group("/chat")
			{
				chat.POST("/cancel", h.cancelChat)
			}

			// 语音相关 - 使用V2版本
			voice := protected.Group("/voice")
			{
				voice.POST("/transcribe", h.transcribeAudio)
				voice.POST("/synthesize", h.synthesizeText)
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

// 获取聊天统计 - 已移至V2版本
// func (h *APIHandler) getChatStats(c *gin.Context) {
//	streamID := c.Param("stream_id")
//	// TODO: 实现V2版本的统计获取
//	c.JSON(http.StatusOK, gin.H{"stream_id": streamID})
// }

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

// 获取语音统计 - 已移至V2版本
// func (h *APIHandler) getVoiceStats(c *gin.Context) {
//	sessionID := c.Param("session_id")
//	// TODO: 实现V2版本的统计获取
//	c.JSON(http.StatusOK, gin.H{"session_id": sessionID})
// }

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

	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 设置默认值
	if req.TopK == 0 {
		req.TopK = 10
	}

	// 调用算法服务进行文档搜索
	searchReq := map[string]interface{}{
		"query":     req.Query,
		"top_k":     req.TopK,
		"filter":    req.Filter,
		"user_id":   userID,
		"tenant_id": tenantID,
	}

	results, err := h.callDocumentSearchService(c.Request.Context(), searchReq)
	if err != nil {
		logger.Error("文档搜索失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "query", Value: req.Query})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "搜索失败"})
		return
	}

	c.JSON(http.StatusOK, results)
}

// 获取搜索建议
func (h *APIHandler) getSearchSuggestions(c *gin.Context) {
	query := c.Query("q")
	userID := c.GetString("user_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "查询参数不能为空"})
		return
	}

	// 调用算法服务获取搜索建议
	suggestions, err := h.getSearchSuggestionsFromService(c.Request.Context(), query, userID)
	if err != nil {
		logger.Error("获取搜索建议失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "query", Value: query})
		// 返回基础建议作为降级方案
		c.JSON(http.StatusOK, gin.H{
			"suggestions": []string{
				query + " 相关内容",
				query + " 使用指南",
				query + " 最佳实践",
			},
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"suggestions": suggestions,
		"query":       query,
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
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析查询参数
	page := c.DefaultQuery("page", "1")
	limit := c.DefaultQuery("limit", "20")
	sortBy := c.DefaultQuery("sort_by", "last_msg_at")
	order := c.DefaultQuery("order", "desc")

	pageInt, _ := strconv.Atoi(page)
	limitInt, _ := strconv.Atoi(limit)
	offset := (pageInt - 1) * limitInt

	opts := repository.ListOptions{
		Limit:  limitInt,
		Offset: offset,
		SortBy: sortBy,
		Order:  order,
	}

	// 从仓库获取会话列表
	conversations, total, err := h.conversationRepo.GetByUserID(c.Request.Context(), userID, opts)
	if err != nil {
		logger.Error("获取会话列表失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "user_id", Value: userID})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取会话列表失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"conversations": conversations,
		"total":         total,
		"page":          pageInt,
		"limit":         limitInt,
	})
}

func (h *APIHandler) createConversation(c *gin.Context) {
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析请求体
	var req struct {
		Title    string                 `json:"title"`
		Summary  string                 `json:"summary"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求参数错误"})
		return
	}

	// 创建会话对象
	conv := &repository.Conversation{
		UserID:   userID,
		TenantID: tenantID,
		Title:    req.Title,
		Summary:  req.Summary,
		Metadata: req.Metadata,
	}

	// 保存到数据库
	if err := h.conversationRepo.Create(c.Request.Context(), conv); err != nil {
		logger.Error("创建会话失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "user_id", Value: userID})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建会话失败"})
		return
	}

	logger.Info("会话创建成功", logger.Field{Key: "conversation_id", Value: conv.ID}, logger.Field{Key: "user_id", Value: userID})
	c.JSON(http.StatusCreated, gin.H{
		"id":           conv.ID,
		"message":      "会话创建成功",
		"conversation": conv,
	})
}

func (h *APIHandler) getConversation(c *gin.Context) {
	id := c.Param("id")
	userID := c.GetString("user_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 从数据库获取会话
	conv, err := h.conversationRepo.Get(c.Request.Context(), id)
	if err != nil {
		logger.Error("获取会话失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "conversation_id", Value: id})
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}

	// 验证用户权限
	if conv.UserID != userID {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权访问此会话"})
		return
	}

	c.JSON(http.StatusOK, conv)
}

func (h *APIHandler) updateConversation(c *gin.Context) {
	id := c.Param("id")
	userID := c.GetString("user_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析请求体
	var req struct {
		Title    string                 `json:"title"`
		Summary  string                 `json:"summary"`
		Status   string                 `json:"status"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求参数错误"})
		return
	}

	// 获取现有会话
	conv, err := h.conversationRepo.Get(c.Request.Context(), id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}

	// 验证用户权限
	if conv.UserID != userID {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权修改此会话"})
		return
	}

	// 更新字段
	if req.Title != "" {
		conv.Title = req.Title
	}
	if req.Summary != "" {
		conv.Summary = req.Summary
	}
	if req.Status != "" {
		conv.Status = req.Status
	}
	if req.Metadata != nil {
		conv.Metadata = req.Metadata
	}

	// 保存更新
	if err := h.conversationRepo.Update(c.Request.Context(), conv); err != nil {
		logger.Error("更新会话失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "conversation_id", Value: id})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "更新会话失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":           id,
		"message":      "会话更新成功",
		"conversation": conv,
	})
}

func (h *APIHandler) deleteConversation(c *gin.Context) {
	id := c.Param("id")
	userID := c.GetString("user_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 获取会话验证权限
	conv, err := h.conversationRepo.Get(c.Request.Context(), id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}

	if conv.UserID != userID {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权删除此会话"})
		return
	}

	// 软删除会话
	if err := h.conversationRepo.Delete(c.Request.Context(), id); err != nil {
		logger.Error("删除会话失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "conversation_id", Value: id})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "删除会话失败"})
		return
	}

	logger.Info("会话删除成功", logger.Field{Key: "conversation_id", Value: id}, logger.Field{Key: "user_id", Value: userID})
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"message": "会话删除成功",
	})
}

func (h *APIHandler) getConversationMessages(c *gin.Context) {
	id := c.Param("id")
	userID := c.GetString("user_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 验证会话权限
	conv, err := h.conversationRepo.Get(c.Request.Context(), id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}

	if conv.UserID != userID {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权访问此会话"})
		return
	}

	// 解析查询参数
	page := c.DefaultQuery("page", "1")
	limit := c.DefaultQuery("limit", "50")

	pageInt, _ := strconv.Atoi(page)
	limitInt, _ := strconv.Atoi(limit)
	offset := (pageInt - 1) * limitInt

	opts := repository.ListOptions{
		Limit:  limitInt,
		Offset: offset,
		SortBy: "created_at",
		Order:  "asc",
	}

	// 获取消息列表
	messages, total, err := h.conversationRepo.GetMessages(c.Request.Context(), id, opts)
	if err != nil {
		logger.Error("获取会话消息失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "conversation_id", Value: id})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取消息失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"conversation_id": id,
		"messages":        messages,
		"total":           total,
		"page":            pageInt,
		"limit":           limitInt,
	})
}

// Agent相关方法
func (h *APIHandler) agentStream(c *gin.Context) {
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析请求体
	var req struct {
		ConversationID string                 `json:"conversation_id"`
		Message        string                 `json:"message"`
		Tools          []string               `json:"tools"`
		Context        map[string]interface{} `json:"context"`
		MaxTokens      int                    `json:"max_tokens"`
		Temperature    float64                `json:"temperature"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求参数错误"})
		return
	}

	if req.Message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "消息内容不能为空"})
		return
	}

	// 设置SSE响应头
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	// 创建流式响应通道
	eventChan := make(chan map[string]interface{}, 10)
	done := make(chan bool)

	// 启动Agent处理协程
	go func() {
		defer close(eventChan)
		defer func() { done <- true }()

		// 发送开始事件
		eventChan <- map[string]interface{}{
			"event": "agent_start",
			"data": map[string]interface{}{
				"conversation_id": req.ConversationID,
				"timestamp":       time.Now().Unix(),
			},
		}

		// 调用算法服务进行Agent处理
		agentReq := map[string]interface{}{
			"message":         req.Message,
			"conversation_id": req.ConversationID,
			"user_id":         userID,
			"tools":           req.Tools,
			"context":         req.Context,
			"max_tokens":      req.MaxTokens,
			"temperature":     req.Temperature,
		}

		// 这里应该调用算法服务的Agent接口
		if err := h.callAgentService(c.Request.Context(), agentReq, eventChan); err != nil {
			eventChan <- map[string]interface{}{
				"event": "error",
				"data": map[string]interface{}{
					"error": err.Error(),
				},
			}
			return
		}

		// 发送完成事件
		eventChan <- map[string]interface{}{
			"event": "agent_complete",
			"data": map[string]interface{}{
				"conversation_id": req.ConversationID,
				"timestamp":       time.Now().Unix(),
			},
		}
	}()

	// 发送SSE事件流
	c.Stream(func(w io.Writer) bool {
		select {
		case event, ok := <-eventChan:
			if !ok {
				return false
			}

			eventData, _ := json.Marshal(event["data"])
			fmt.Fprintf(w, "event: %s\n", event["event"])
			fmt.Fprintf(w, "data: %s\n\n", eventData)
			return true

		case <-done:
			return false

		case <-c.Request.Context().Done():
			return false
		}
	})
}

func (h *APIHandler) executeAgentTool(c *gin.Context) {
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析请求体
	var req struct {
		ToolName       string                 `json:"tool_name"`
		Parameters     map[string]interface{} `json:"parameters"`
		ConversationID string                 `json:"conversation_id"`
		Context        map[string]interface{} `json:"context"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求参数错误"})
		return
	}

	if req.ToolName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "工具名称不能为空"})
		return
	}

	// 验证工具权限
	if !h.isToolAllowed(req.ToolName, userID) {
		c.JSON(http.StatusForbidden, gin.H{"error": "无权使用此工具"})
		return
	}

	// 执行工具
	result, err := h.executeToolInternal(c.Request.Context(), req.ToolName, req.Parameters, userID)
	if err != nil {
		logger.Error("工具执行失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "tool_name", Value: req.ToolName}, logger.Field{Key: "user_id", Value: userID})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "工具执行失败: " + err.Error()})
		return
	}

	// 记录工具使用
	h.logToolUsage(userID, req.ToolName, req.ConversationID, result)

	c.JSON(http.StatusOK, gin.H{
		"tool_name": req.ToolName,
		"result":    result,
		"timestamp": time.Now().Unix(),
		"status":    "success",
	})
}

func (h *APIHandler) getAgentCapabilities(c *gin.Context) {
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	// 获取用户可用的工具和能力
	capabilities := map[string]interface{}{
		"reasoning": map[string]interface{}{
			"enabled": true,
			"types": []string{
				"logical_reasoning",
				"mathematical_reasoning",
				"causal_reasoning",
				"common_sense_reasoning",
			},
		},
		"planning": map[string]interface{}{
			"enabled": true,
			"features": []string{
				"task_decomposition",
				"goal_setting",
				"step_by_step_execution",
				"adaptive_planning",
			},
		},
		"tool_use": map[string]interface{}{
			"enabled":         true,
			"available_tools": h.getAvailableTools(userID, tenantID),
		},
		"memory": map[string]interface{}{
			"enabled": true,
			"types": []string{
				"conversation_memory",
				"long_term_memory",
				"episodic_memory",
				"semantic_memory",
			},
		},
		"multimodal": map[string]interface{}{
			"enabled": true,
			"supported_formats": []string{
				"text",
				"image",
				"audio",
				"video",
			},
		},
		"code_understanding": map[string]interface{}{
			"enabled": true,
			"languages": []string{
				"python", "javascript", "go", "java",
				"cpp", "rust", "typescript", "sql",
			},
		},
	}

	// 根据用户权限过滤能力
	filteredCapabilities := h.filterCapabilitiesByPermission(capabilities, userID, tenantID)

	c.JSON(http.StatusOK, gin.H{
		"capabilities": filteredCapabilities,
		"user_id":      userID,
		"tenant_id":    tenantID,
		"timestamp":    time.Now().Unix(),
	})
}

// 管理接口
func (h *APIHandler) getSystemStats(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"chat_streams": gin.H{
			"total": 0, // TODO: 实现V2版本的统计
		},
		"voice_sessions": gin.H{
			"total": 0, // TODO: 实现V2版本的统计
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

// callAgentService 调用算法服务的Agent接口
func (h *APIHandler) callAgentService(ctx context.Context, request map[string]interface{}, eventChan chan<- map[string]interface{}) error {
	// 模拟Agent服务调用
	go func() {
		// 发送思考事件
		eventChan <- map[string]interface{}{
			"event": "agent_thinking",
			"data": map[string]interface{}{
				"content":   "正在分析您的问题...",
				"timestamp": time.Now().Unix(),
			},
		}

		// 模拟处理延迟
		time.Sleep(1 * time.Second)

		// 发送响应事件
		eventChan <- map[string]interface{}{
			"event": "agent_response",
			"data": map[string]interface{}{
				"content":   "基于您的问题，我建议采用以下方案...",
				"timestamp": time.Now().Unix(),
			},
		}
	}()

	return nil
}

// isToolAllowed 检查用户是否有权限使用指定工具
func (h *APIHandler) isToolAllowed(toolName, userID string) bool {
	allowedTools := map[string]bool{
		"web_search":       true,
		"calculator":       true,
		"code_interpreter": true,
		"document_search":  true,
		"knowledge_query":  true,
		"reasoning_engine": true,
	}

	return allowedTools[toolName]
}

// executeToolInternal 内部工具执行逻辑
func (h *APIHandler) executeToolInternal(ctx context.Context, toolName string, parameters map[string]interface{}, userID string) (interface{}, error) {
	switch toolName {
	case "web_search":
		query, _ := parameters["query"].(string)
		return map[string]interface{}{
			"query": query,
			"results": []map[string]interface{}{
				{"title": "搜索结果1", "url": "https://example.com/1", "snippet": "搜索结果摘要..."},
			},
			"timestamp": time.Now().Unix(),
		}, nil

	case "calculator":
		expression, _ := parameters["expression"].(string)
		return map[string]interface{}{
			"expression": expression,
			"result":     "42",
			"timestamp":  time.Now().Unix(),
		}, nil

	case "document_search":
		query, _ := parameters["query"].(string)
		return map[string]interface{}{
			"query":     query,
			"documents": []map[string]interface{}{},
			"total":     0,
			"timestamp": time.Now().Unix(),
		}, nil

	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

// logToolUsage 记录工具使用情况
func (h *APIHandler) logToolUsage(userID, toolName, conversationID string, result interface{}) {
	logger.Info("工具使用记录",
		logger.Field{Key: "user_id", Value: userID},
		logger.Field{Key: "tool_name", Value: toolName},
		logger.Field{Key: "conversation_id", Value: conversationID},
		logger.Field{Key: "timestamp", Value: time.Now().Unix()},
	)
}

// getAvailableTools 获取用户可用的工具列表
func (h *APIHandler) getAvailableTools(userID, tenantID string) []map[string]interface{} {
	return []map[string]interface{}{
		{
			"name":        "web_search",
			"description": "网络搜索工具",
			"parameters": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "搜索查询",
					"required":    true,
				},
			},
		},
		{
			"name":        "calculator",
			"description": "数学计算器",
			"parameters": map[string]interface{}{
				"expression": map[string]interface{}{
					"type":        "string",
					"description": "数学表达式",
					"required":    true,
				},
			},
		},
		{
			"name":        "document_search",
			"description": "文档搜索工具",
			"parameters": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "搜索查询",
					"required":    true,
				},
			},
		},
	}
}

// filterCapabilitiesByPermission 根据用户权限过滤能力
func (h *APIHandler) filterCapabilitiesByPermission(capabilities map[string]interface{}, userID, tenantID string) map[string]interface{} {
	// 基础用户拥有所有基本能力
	return capabilities
}

// callDocumentSearchService 调用文档搜索服务
func (h *APIHandler) callDocumentSearchService(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	// 模拟调用算法服务的文档搜索接口
	query, _ := request["query"].(string)
	topK, _ := request["top_k"].(int)

	// 模拟搜索结果
	results := []map[string]interface{}{
		{
			"id":      "doc_001",
			"title":   "VoiceHelper 用户指南",
			"content": "这是关于 " + query + " 的详细说明...",
			"score":   0.95,
			"source":  "user_guide.pdf",
			"page":    1,
			"metadata": map[string]interface{}{
				"category": "documentation",
				"tags":     []string{"guide", "tutorial"},
			},
		},
		{
			"id":      "doc_002",
			"title":   "API 参考文档",
			"content": "关于 " + query + " 的API使用方法...",
			"score":   0.87,
			"source":  "api_reference.md",
			"page":    5,
			"metadata": map[string]interface{}{
				"category": "api",
				"tags":     []string{"api", "reference"},
			},
		},
	}

	// 根据topK限制结果数量
	if topK > 0 && topK < len(results) {
		results = results[:topK]
	}

	return map[string]interface{}{
		"results":   results,
		"total":     len(results),
		"query":     query,
		"timestamp": time.Now().Unix(),
	}, nil
}

// getSearchSuggestionsFromService 从服务获取搜索建议
func (h *APIHandler) getSearchSuggestionsFromService(ctx context.Context, query, userID string) ([]string, error) {
	// 模拟智能搜索建议
	suggestions := []string{
		query + " 使用指南",
		query + " 最佳实践",
		query + " 常见问题",
		query + " API 文档",
		query + " 配置说明",
	}

	// 根据查询内容提供更智能的建议
	if strings.Contains(strings.ToLower(query), "api") {
		suggestions = append([]string{
			query + " 接口文档",
			query + " 调用示例",
			query + " 错误码说明",
		}, suggestions...)
	}

	if strings.Contains(strings.ToLower(query), "配置") || strings.Contains(strings.ToLower(query), "config") {
		suggestions = append([]string{
			query + " 环境配置",
			query + " 参数设置",
			query + " 配置文件",
		}, suggestions...)
	}

	// 限制建议数量
	if len(suggestions) > 8 {
		suggestions = suggestions[:8]
	}

	return suggestions, nil
}
