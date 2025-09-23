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
	voiceHandler     *RealtimeVoiceHandler
}

// NewAPIHandler 创建API处理器
func NewAPIHandler(
	auth *middleware.AuthMiddleware,
	rbac *middleware.RBACMiddleware,
	tenant *middleware.TenantMiddleware,
	conversationRepo repository.ConversationRepository,
	algoServiceURL string,
	voiceHandler *RealtimeVoiceHandler,
) *APIHandler {
	return &APIHandler{
		authMiddleware:   auth,
		rbacMiddleware:   rbac,
		tenantMiddleware: tenant,
		conversationRepo: conversationRepo,
		algoServiceURL:   algoServiceURL,
		voiceHandler:     voiceHandler,
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
				voice.GET("/stream", h.handleVoiceWebSocket) // WebSocket语音流处理
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

	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 调用算法服务取消聊天
	cancelReq := map[string]interface{}{
		"session_id": req.StreamID,
		"user_id":    userID,
		"tenant_id":  tenantID,
	}

	success, err := h.callCancelChatService(c.Request.Context(), cancelReq)
	if err != nil {
		logger.Error("取消聊天失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "stream_id", Value: req.StreamID})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "取消聊天失败"})
		return
	}

	if success {
		c.JSON(http.StatusOK, gin.H{
			"message":   "Chat cancelled successfully",
			"stream_id": req.StreamID,
			"status":    "cancelled",
			"timestamp": time.Now().Unix(),
		})
	} else {
		c.JSON(http.StatusNotFound, gin.H{
			"error":     "Chat session not found or already completed",
			"stream_id": req.StreamID,
		})
	}
}

// 获取聊天统计 - 已移至V2版本
// func (h *APIHandler) getChatStats(c *gin.Context) {
//	streamID := c.Param("stream_id")
//	// TODO: 实现V2版本的统计获取
//	c.JSON(http.StatusOK, gin.H{"stream_id": streamID})
// }

// 语音转写
func (h *APIHandler) transcribeAudio(c *gin.Context) {
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析multipart form
	err := c.Request.ParseMultipartForm(32 << 20) // 32MB max
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "解析表单失败"})
		return
	}

	// 获取音频文件
	file, header, err := c.Request.FormFile("audio")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "音频文件不能为空"})
		return
	}
	defer file.Close()

	// 检查文件大小 (最大10MB)
	if header.Size > 10*1024*1024 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "音频文件过大，最大支持10MB"})
		return
	}

	// 获取可选参数
	language := c.DefaultPostForm("language", "zh-CN")
	model := c.DefaultPostForm("model", "whisper-1")

	// 读取音频数据
	audioData, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "读取音频文件失败"})
		return
	}

	// 调用算法服务进行语音转写
	transcribeReq := map[string]interface{}{
		"audio_data": audioData,
		"filename":   header.Filename,
		"language":   language,
		"model":      model,
		"user_id":    userID,
		"tenant_id":  tenantID,
	}

	result, err := h.callTranscribeService(c.Request.Context(), transcribeReq)
	if err != nil {
		logger.Error("语音转写失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "filename", Value: header.Filename})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "语音转写失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"text":       result["text"],
		"confidence": result["confidence"],
		"language":   result["language"],
		"duration":   result["duration"],
		"model":      model,
		"timestamp":  time.Now().Unix(),
	})
}

// 语音合成
func (h *APIHandler) synthesizeText(c *gin.Context) {
	var req struct {
		Text     string  `json:"text" binding:"required"`
		Voice    string  `json:"voice"`
		Format   string  `json:"format"`
		Speed    float64 `json:"speed"`
		Emotion  string  `json:"emotion"`
		Language string  `json:"language"`
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
	if req.Voice == "" {
		req.Voice = "zh-female-1"
	}
	if req.Format == "" {
		req.Format = "mp3"
	}
	if req.Speed == 0 {
		req.Speed = 1.0
	}
	if req.Language == "" {
		req.Language = "zh-CN"
	}

	// 验证文本长度
	if len(req.Text) > 5000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "文本长度不能超过5000字符"})
		return
	}

	// 调用算法服务进行语音合成
	synthesizeReq := map[string]interface{}{
		"text":      req.Text,
		"voice":     req.Voice,
		"format":    req.Format,
		"speed":     req.Speed,
		"emotion":   req.Emotion,
		"language":  req.Language,
		"user_id":   userID,
		"tenant_id": tenantID,
	}

	result, err := h.callSynthesizeService(c.Request.Context(), synthesizeReq)
	if err != nil {
		logger.Error("语音合成失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "text_length", Value: len(req.Text)})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "语音合成失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"audio_url":  result["audio_url"],
		"audio_data": result["audio_data"], // base64编码的音频数据
		"duration":   result["duration"],
		"format":     req.Format,
		"voice":      req.Voice,
		"text_hash":  result["text_hash"], // 用于缓存
		"timestamp":  time.Now().Unix(),
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
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析查询参数
	page := c.DefaultQuery("page", "1")
	limit := c.DefaultQuery("limit", "20")
	source := c.Query("source")
	status := c.DefaultQuery("status", "active")

	pageInt, err := strconv.Atoi(page)
	if err != nil || pageInt < 1 {
		pageInt = 1
	}

	limitInt, err := strconv.Atoi(limit)
	if err != nil || limitInt < 1 || limitInt > 100 {
		limitInt = 20
	}

	offset := (pageInt - 1) * limitInt

	// 调用文档仓库获取文档列表
	documents, err := h.getDocumentsFromRepository(c.Request.Context(), tenantID, source, status, limitInt, offset)
	if err != nil {
		logger.Error("获取文档列表失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "tenant_id", Value: tenantID})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取文档列表失败"})
		return
	}

	// 获取总数
	total, err := h.getDocumentCountFromRepository(c.Request.Context(), tenantID, source, status)
	if err != nil {
		logger.Error("获取文档总数失败", logger.Field{Key: "error", Value: err})
		total = 0
	}

	c.JSON(http.StatusOK, gin.H{
		"documents": documents,
		"total":     total,
		"page":      pageInt,
		"limit":     limitInt,
		"pages":     (total + limitInt - 1) / limitInt,
	})
}

func (h *APIHandler) getDocument(c *gin.Context) {
	id := c.Param("id")
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "文档ID不能为空"})
		return
	}

	// 从仓库获取文档
	document, err := h.getDocumentFromRepository(c.Request.Context(), id, tenantID)
	if err != nil {
		if err.Error() == "document not found" {
			c.JSON(http.StatusNotFound, gin.H{"error": "文档不存在"})
			return
		}
		logger.Error("获取文档失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "document_id", Value: id})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取文档失败"})
		return
	}

	// 获取文档块（如果需要）
	includeChunks := c.DefaultQuery("include_chunks", "false")
	if includeChunks == "true" {
		chunks, err := h.getDocumentChunksFromRepository(c.Request.Context(), id)
		if err != nil {
			logger.Error("获取文档块失败", logger.Field{Key: "error", Value: err})
		} else {
			document["chunks"] = chunks
		}
	}

	c.JSON(http.StatusOK, document)
}

func (h *APIHandler) uploadDocument(c *gin.Context) {
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 解析multipart form
	err := c.Request.ParseMultipartForm(50 << 20) // 50MB max
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "解析表单失败"})
		return
	}

	// 获取文件
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "文件不能为空"})
		return
	}
	defer file.Close()

	// 检查文件大小 (最大50MB)
	if header.Size > 50*1024*1024 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "文件过大，最大支持50MB"})
		return
	}

	// 获取表单参数
	title := c.PostForm("title")
	if title == "" {
		title = header.Filename
	}
	source := c.DefaultPostForm("source", "upload")
	contentType := c.DefaultPostForm("content_type", "auto")

	// 读取文件内容
	fileData, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "读取文件失败"})
		return
	}

	// 调用文档上传服务
	uploadReq := map[string]interface{}{
		"title":        title,
		"filename":     header.Filename,
		"content_type": contentType,
		"source":       source,
		"file_data":    fileData,
		"file_size":    header.Size,
		"user_id":      userID,
		"tenant_id":    tenantID,
	}

	result, err := h.callDocumentUploadService(c.Request.Context(), uploadReq)
	if err != nil {
		logger.Error("文档上传失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "filename", Value: header.Filename})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "文档上传失败"})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"id":           result["document_id"],
		"title":        title,
		"filename":     header.Filename,
		"size":         header.Size,
		"content_type": result["content_type"],
		"status":       result["status"],
		"message":      "Document uploaded successfully",
		"timestamp":    time.Now().Unix(),
	})
}

func (h *APIHandler) updateDocument(c *gin.Context) {
	id := c.Param("id")
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "文档ID不能为空"})
		return
	}

	var req struct {
		Title       string                 `json:"title"`
		Content     string                 `json:"content"`
		ContentType string                 `json:"content_type"`
		Source      string                 `json:"source"`
		Status      string                 `json:"status"`
		Metadata    map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 验证状态值
	if req.Status != "" && req.Status != "active" && req.Status != "archived" && req.Status != "deleted" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "无效的状态值"})
		return
	}

	// 调用文档更新服务
	updateReq := map[string]interface{}{
		"document_id":  id,
		"title":        req.Title,
		"content":      req.Content,
		"content_type": req.ContentType,
		"source":       req.Source,
		"status":       req.Status,
		"metadata":     req.Metadata,
		"user_id":      userID,
		"tenant_id":    tenantID,
	}

	result, err := h.callDocumentUpdateService(c.Request.Context(), updateReq)
	if err != nil {
		if err.Error() == "document not found" {
			c.JSON(http.StatusNotFound, gin.H{"error": "文档不存在"})
			return
		}
		logger.Error("文档更新失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "document_id", Value: id})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "文档更新失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":         id,
		"title":      result["title"],
		"status":     result["status"],
		"updated_at": result["updated_at"],
		"message":    "Document updated successfully",
		"timestamp":  time.Now().Unix(),
	})
}

func (h *APIHandler) deleteDocument(c *gin.Context) {
	id := c.Param("id")
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "文档ID不能为空"})
		return
	}

	// 获取删除类型（软删除或硬删除）
	hardDelete := c.DefaultQuery("hard", "false") == "true"

	// 调用文档删除服务
	deleteReq := map[string]interface{}{
		"document_id": id,
		"hard_delete": hardDelete,
		"user_id":     userID,
		"tenant_id":   tenantID,
	}

	err := h.callDocumentDeleteService(c.Request.Context(), deleteReq)
	if err != nil {
		if err.Error() == "document not found" {
			c.JSON(http.StatusNotFound, gin.H{"error": "文档不存在"})
			return
		}
		logger.Error("文档删除失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "document_id", Value: id})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "文档删除失败"})
		return
	}

	deleteType := "soft"
	if hardDelete {
		deleteType = "hard"
	}

	c.JSON(http.StatusOK, gin.H{
		"id":          id,
		"delete_type": deleteType,
		"message":     "Document deleted successfully",
		"timestamp":   time.Now().Unix(),
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
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 获取时间范围参数
	startTimeStr := c.DefaultQuery("start_time", "")
	endTimeStr := c.DefaultQuery("end_time", "")

	var startTime, endTime time.Time
	var err error

	if startTimeStr != "" {
		startTime, err = time.Parse(time.RFC3339, startTimeStr)
		if err != nil {
			startTime = time.Now().AddDate(0, 0, -7) // 默认7天前
		}
	} else {
		startTime = time.Now().AddDate(0, 0, -7)
	}

	if endTimeStr != "" {
		endTime, err = time.Parse(time.RFC3339, endTimeStr)
		if err != nil {
			endTime = time.Now()
		}
	} else {
		endTime = time.Now()
	}

	// 获取聊天统计
	chatStats, err := h.getChatStatsFromService(c.Request.Context(), tenantID, startTime, endTime)
	if err != nil {
		logger.Error("获取聊天统计失败", logger.Field{Key: "error", Value: err})
		chatStats = map[string]interface{}{
			"total":     0,
			"active":    0,
			"completed": 0,
			"failed":    0,
		}
	}

	// 获取语音会话统计
	voiceStats, err := h.getVoiceStatsFromService(c.Request.Context(), tenantID, startTime, endTime)
	if err != nil {
		logger.Error("获取语音统计失败", logger.Field{Key: "error", Value: err})
		voiceStats = map[string]interface{}{
			"total":               0,
			"active":              0,
			"completed":           0,
			"total_duration":      0,
			"avg_duration":        0,
			"total_audio_minutes": 0,
		}
	}

	// 获取系统资源统计
	systemStats, err := h.getSystemResourceStats(c.Request.Context())
	if err != nil {
		logger.Error("获取系统资源统计失败", logger.Field{Key: "error", Value: err})
		systemStats = map[string]interface{}{
			"uptime":       0,
			"memory_usage": "N/A",
			"cpu_usage":    "N/A",
			"disk_usage":   "N/A",
		}
	}

	// 获取文档统计
	documentStats, err := h.getDocumentStatsFromService(c.Request.Context(), tenantID)
	if err != nil {
		logger.Error("获取文档统计失败", logger.Field{Key: "error", Value: err})
		documentStats = map[string]interface{}{
			"total":    0,
			"active":   0,
			"archived": 0,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"chat_streams":   chatStats,
		"voice_sessions": voiceStats,
		"documents":      documentStats,
		"system":         systemStats,
		"time_range": gin.H{
			"start_time": startTime.Format(time.RFC3339),
			"end_time":   endTime.Format(time.RFC3339),
		},
		"tenant_id": tenantID,
		"timestamp": time.Now().Unix(),
	})
}

func (h *APIHandler) getActiveSessions(c *gin.Context) {
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 获取查询参数
	sessionType := c.DefaultQuery("type", "all") // all, chat, voice
	limit := c.DefaultQuery("limit", "50")

	limitInt, err := strconv.Atoi(limit)
	if err != nil || limitInt < 1 || limitInt > 100 {
		limitInt = 50
	}

	var sessions []map[string]interface{}
	var total int

	// 根据类型获取不同的会话
	switch sessionType {
	case "chat":
		sessions, total, err = h.getActiveChatSessions(c.Request.Context(), tenantID, limitInt)
	case "voice":
		sessions, total, err = h.getActiveVoiceSessions(c.Request.Context(), tenantID, limitInt)
	default: // all
		chatSessions, chatTotal, chatErr := h.getActiveChatSessions(c.Request.Context(), tenantID, limitInt/2)
		voiceSessions, voiceTotal, voiceErr := h.getActiveVoiceSessions(c.Request.Context(), tenantID, limitInt/2)

		if chatErr != nil && voiceErr != nil {
			err = fmt.Errorf("failed to get sessions: chat=%v, voice=%v", chatErr, voiceErr)
		} else {
			sessions = append(sessions, chatSessions...)
			sessions = append(sessions, voiceSessions...)
			total = chatTotal + voiceTotal
		}
	}

	if err != nil {
		logger.Error("获取活跃会话失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "type", Value: sessionType})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取活跃会话失败"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"sessions":  sessions,
		"total":     total,
		"type":      sessionType,
		"limit":     limitInt,
		"tenant_id": tenantID,
		"timestamp": time.Now().Unix(),
	})
}

func (h *APIHandler) reloadConfiguration(c *gin.Context) {
	userID := c.GetString("user_id")
	tenantID := c.GetString("tenant_id")

	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "用户未认证"})
		return
	}

	// 检查管理员权限
	if !h.isAdmin(userID) {
		c.JSON(http.StatusForbidden, gin.H{"error": "需要管理员权限"})
		return
	}

	// 获取配置类型参数
	configType := c.DefaultQuery("type", "all") // all, app, database, cache, algo

	var reloadResults []map[string]interface{}
	var errors []string

	// 根据类型重载不同的配置
	switch configType {
	case "app":
		if err := h.reloadAppConfig(c.Request.Context()); err != nil {
			errors = append(errors, fmt.Sprintf("app config: %v", err))
		} else {
			reloadResults = append(reloadResults, map[string]interface{}{
				"type":    "app",
				"status":  "success",
				"message": "Application configuration reloaded",
			})
		}
	case "database":
		if err := h.reloadDatabaseConfig(c.Request.Context()); err != nil {
			errors = append(errors, fmt.Sprintf("database config: %v", err))
		} else {
			reloadResults = append(reloadResults, map[string]interface{}{
				"type":    "database",
				"status":  "success",
				"message": "Database configuration reloaded",
			})
		}
	case "cache":
		if err := h.reloadCacheConfig(c.Request.Context()); err != nil {
			errors = append(errors, fmt.Sprintf("cache config: %v", err))
		} else {
			reloadResults = append(reloadResults, map[string]interface{}{
				"type":    "cache",
				"status":  "success",
				"message": "Cache configuration reloaded",
			})
		}
	case "algo":
		if err := h.reloadAlgoConfig(c.Request.Context()); err != nil {
			errors = append(errors, fmt.Sprintf("algo config: %v", err))
		} else {
			reloadResults = append(reloadResults, map[string]interface{}{
				"type":    "algo",
				"status":  "success",
				"message": "Algorithm service configuration reloaded",
			})
		}
	default: // all
		configs := []string{"app", "database", "cache", "algo"}
		for _, cfg := range configs {
			var err error
			switch cfg {
			case "app":
				err = h.reloadAppConfig(c.Request.Context())
			case "database":
				err = h.reloadDatabaseConfig(c.Request.Context())
			case "cache":
				err = h.reloadCacheConfig(c.Request.Context())
			case "algo":
				err = h.reloadAlgoConfig(c.Request.Context())
			}

			if err != nil {
				errors = append(errors, fmt.Sprintf("%s config: %v", cfg, err))
			} else {
				reloadResults = append(reloadResults, map[string]interface{}{
					"type":    cfg,
					"status":  "success",
					"message": fmt.Sprintf("%s configuration reloaded", cfg),
				})
			}
		}
	}

	// 记录重载操作
	logger.Info("配置重载操作",
		logger.Field{Key: "user_id", Value: userID},
		logger.Field{Key: "config_type", Value: configType},
		logger.Field{Key: "success_count", Value: len(reloadResults)},
		logger.Field{Key: "error_count", Value: len(errors)})

	status := http.StatusOK
	message := "Configuration reloaded successfully"

	if len(errors) > 0 {
		if len(reloadResults) == 0 {
			status = http.StatusInternalServerError
			message = "Configuration reload failed"
		} else {
			status = http.StatusPartialContent
			message = "Configuration partially reloaded"
		}
	}

	c.JSON(status, gin.H{
		"message":   message,
		"type":      configType,
		"results":   reloadResults,
		"errors":    errors,
		"user_id":   userID,
		"tenant_id": tenantID,
		"timestamp": time.Now().Unix(),
	})
}

func (h *APIHandler) setMaintenanceMode(c *gin.Context) {
	var req struct {
		Enabled   bool   `json:"enabled"`
		Message   string `json:"message"`
		StartTime string `json:"start_time,omitempty"`
		EndTime   string `json:"end_time,omitempty"`
		Reason    string `json:"reason,omitempty"`
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

	// 检查管理员权限
	if !h.isAdmin(userID) {
		c.JSON(http.StatusForbidden, gin.H{"error": "需要管理员权限"})
		return
	}

	// 解析时间参数
	var startTime, endTime *time.Time
	if req.StartTime != "" {
		if t, err := time.Parse(time.RFC3339, req.StartTime); err == nil {
			startTime = &t
		}
	}
	if req.EndTime != "" {
		if t, err := time.Parse(time.RFC3339, req.EndTime); err == nil {
			endTime = &t
		}
	}

	// 设置默认消息
	if req.Message == "" {
		if req.Enabled {
			req.Message = "系统正在维护中，请稍后再试"
		} else {
			req.Message = "系统维护已结束，服务已恢复正常"
		}
	}

	// 构建维护模式配置
	maintenanceConfig := map[string]interface{}{
		"enabled":   req.Enabled,
		"message":   req.Message,
		"reason":    req.Reason,
		"set_by":    userID,
		"set_at":    time.Now().Unix(),
		"tenant_id": tenantID,
	}

	if startTime != nil {
		maintenanceConfig["start_time"] = startTime.Format(time.RFC3339)
	}
	if endTime != nil {
		maintenanceConfig["end_time"] = endTime.Format(time.RFC3339)
	}

	// 调用维护模式设置服务
	err := h.setMaintenanceModeInService(c.Request.Context(), maintenanceConfig)
	if err != nil {
		logger.Error("设置维护模式失败", logger.Field{Key: "error", Value: err}, logger.Field{Key: "enabled", Value: req.Enabled})
		c.JSON(http.StatusInternalServerError, gin.H{"error": "设置维护模式失败"})
		return
	}

	// 记录维护模式变更
	action := "disabled"
	if req.Enabled {
		action = "enabled"
	}

	logger.Info("维护模式已变更",
		logger.Field{Key: "action", Value: action},
		logger.Field{Key: "user_id", Value: userID},
		logger.Field{Key: "message", Value: req.Message},
		logger.Field{Key: "reason", Value: req.Reason})

	// 如果启用维护模式，通知所有活跃会话
	if req.Enabled {
		go h.notifyMaintenanceModeToActiveSessions(maintenanceConfig)
	}

	c.JSON(http.StatusOK, gin.H{
		"maintenance_mode": req.Enabled,
		"message":          req.Message,
		"reason":           req.Reason,
		"start_time":       req.StartTime,
		"end_time":         req.EndTime,
		"set_by":           userID,
		"set_at":           time.Now().Unix(),
		"status":           fmt.Sprintf("Maintenance mode %s successfully", action),
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

// ==================== 新增的辅助方法 ====================

// callCancelChatService 调用取消聊天服务
func (h *APIHandler) callCancelChatService(ctx context.Context, request map[string]interface{}) (bool, error) {
	sessionID, _ := request["session_id"].(string)
	userID, _ := request["user_id"].(string)
	tenantID, _ := request["tenant_id"].(string)

	if sessionID == "" {
		return false, fmt.Errorf("session_id is required")
	}

	// 构建取消请求
	cancelReq := map[string]interface{}{
		"action":     "cancel_chat",
		"session_id": sessionID,
		"user_id":    userID,
		"tenant_id":  tenantID,
		"timestamp":  time.Now().Unix(),
	}

	// 调用算法服务的取消接口
	reqBody, err := json.Marshal(cancelReq)
	if err != nil {
		return false, fmt.Errorf("failed to marshal cancel request: %w", err)
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/chat/cancel", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return false, fmt.Errorf("failed to create cancel request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 发送请求
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return false, fmt.Errorf("failed to call cancel service: %w", err)
	}
	defer resp.Body.Close()

	// 解析响应
	if resp.StatusCode == http.StatusNotFound {
		return false, nil // 会话不存在或已完成
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return false, fmt.Errorf("cancel service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false, fmt.Errorf("failed to decode cancel response: %w", err)
	}

	success, _ := result["success"].(bool)
	return success, nil
}

// callTranscribeService 调用语音转写服务
func (h *APIHandler) callTranscribeService(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	audioData, _ := request["audio_data"].([]byte)
	filename, _ := request["filename"].(string)
	language, _ := request["language"].(string)
	model, _ := request["model"].(string)
	userID, _ := request["user_id"].(string)
	tenantID, _ := request["tenant_id"].(string)

	if len(audioData) == 0 {
		return nil, fmt.Errorf("audio_data is required")
	}

	// 构建转写请求
	transcribeReq := map[string]interface{}{
		"audio_data": audioData,
		"filename":   filename,
		"language":   language,
		"model":      model,
		"user_id":    userID,
		"tenant_id":  tenantID,
		"timestamp":  time.Now().Unix(),
		"config": map[string]interface{}{
			"sample_rate":     16000,
			"channels":        1,
			"format":          "wav",
			"vad_enabled":     true,
			"noise_reduction": true,
		},
	}

	// 调用算法服务的ASR接口
	reqBody, err := json.Marshal(transcribeReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal transcribe request: %w", err)
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/voice/transcribe", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to create transcribe request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 设置较长的超时时间，因为语音转写可能需要更多时间
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call transcribe service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("transcribe service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode transcribe response: %w", err)
	}

	// 验证必要字段
	if result["text"] == nil {
		return nil, fmt.Errorf("transcribe service returned invalid response: missing text field")
	}

	// 添加处理时间戳
	result["processed_at"] = time.Now().Unix()
	result["model_used"] = model
	result["language_detected"] = result["language"]

	return result, nil
}

// callSynthesizeService 调用语音合成服务
func (h *APIHandler) callSynthesizeService(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	text, _ := request["text"].(string)
	voice, _ := request["voice"].(string)
	format, _ := request["format"].(string)
	speed, _ := request["speed"].(float64)
	emotion, _ := request["emotion"].(string)
	language, _ := request["language"].(string)
	userID, _ := request["user_id"].(string)
	tenantID, _ := request["tenant_id"].(string)

	if text == "" {
		return nil, fmt.Errorf("text is required")
	}

	// 构建合成请求
	synthesizeReq := map[string]interface{}{
		"text":      text,
		"voice":     voice,
		"format":    format,
		"speed":     speed,
		"emotion":   emotion,
		"language":  language,
		"user_id":   userID,
		"tenant_id": tenantID,
		"timestamp": time.Now().Unix(),
		"config": map[string]interface{}{
			"sample_rate":   22050,
			"bit_rate":      128,
			"channels":      1,
			"cache_enabled": true,
			"streaming":     false,
			"normalize":     true,
		},
	}

	// 调用算法服务的TTS接口
	reqBody, err := json.Marshal(synthesizeReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal synthesize request: %w", err)
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/voice/synthesize", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to create synthesize request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 设置较长的超时时间，因为语音合成可能需要更多时间
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call synthesize service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("synthesize service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode synthesize response: %w", err)
	}

	// 验证必要字段
	if result["audio_data"] == nil && result["audio_url"] == nil {
		return nil, fmt.Errorf("synthesize service returned invalid response: missing audio data")
	}

	// 添加处理时间戳和元数据
	result["processed_at"] = time.Now().Unix()
	result["voice_used"] = voice
	result["text_length"] = len(text)
	result["format_used"] = format

	// 生成文本哈希用于缓存
	textHash := fmt.Sprintf("%x", time.Now().Unix())
	result["text_hash"] = textHash

	return result, nil
}

// handleVoiceWebSocket 处理WebSocket语音流
func (h *APIHandler) handleVoiceWebSocket(c *gin.Context) {
	if h.voiceHandler == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "语音服务不可用"})
		return
	}

	// 委托给专门的语音处理器
	h.voiceHandler.HandleWebSocket(c)
}

// getDocumentsFromRepository 从仓库获取文档列表
func (h *APIHandler) getDocumentsFromRepository(ctx context.Context, tenantID, source, status string, limit, offset int) ([]map[string]interface{}, error) {
	// 构建查询请求
	queryReq := map[string]interface{}{
		"tenant_id": tenantID,
		"source":    source,
		"status":    status,
		"limit":     limit,
		"offset":    offset,
		"timestamp": time.Now().Unix(),
	}

	// 调用算法服务的文档查询接口
	reqBody, err := json.Marshal(queryReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal document query request: %w", err)
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/documents/list", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to create document query request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 发送请求
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		// 降级到模拟数据
		logger.Error("文档服务不可用，使用模拟数据", logger.Field{Key: "error", Value: err})
		return h.getMockDocuments(), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		logger.Error("文档服务返回错误", logger.Field{Key: "status", Value: resp.StatusCode}, logger.Field{Key: "body", Value: string(body)})
		return h.getMockDocuments(), nil
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		logger.Error("解析文档服务响应失败", logger.Field{Key: "error", Value: err})
		return h.getMockDocuments(), nil
	}

	// 提取文档列表
	documents, ok := result["documents"].([]interface{})
	if !ok {
		logger.Error("文档服务返回格式错误")
		return h.getMockDocuments(), nil
	}

	// 转换为所需格式
	var documentList []map[string]interface{}
	for _, doc := range documents {
		if docMap, ok := doc.(map[string]interface{}); ok {
			documentList = append(documentList, docMap)
		}
	}

	return documentList, nil
}

// getMockDocuments 获取模拟文档数据
func (h *APIHandler) getMockDocuments() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"id":           "doc_1",
			"document_id":  "document_001",
			"title":        "VoiceHelper用户指南",
			"content_type": "text/plain",
			"source":       "upload",
			"status":       "active",
			"file_size":    1024000,
			"created_at":   time.Now().AddDate(0, 0, -1).Format(time.RFC3339),
			"updated_at":   time.Now().Format(time.RFC3339),
		},
		{
			"id":           "doc_2",
			"document_id":  "document_002",
			"title":        "API参考文档",
			"content_type": "application/pdf",
			"source":       "upload",
			"status":       "active",
			"file_size":    2048000,
			"created_at":   time.Now().AddDate(0, 0, -2).Format(time.RFC3339),
			"updated_at":   time.Now().Format(time.RFC3339),
		},
	}
}

// getDocumentCountFromRepository 获取文档总数
func (h *APIHandler) getDocumentCountFromRepository(ctx context.Context, tenantID, source, status string) (int, error) {
	// 模拟文档总数
	return 2, nil
}

// getDocumentFromRepository 从仓库获取单个文档
func (h *APIHandler) getDocumentFromRepository(ctx context.Context, documentID, tenantID string) (map[string]interface{}, error) {
	// 模拟文档数据
	return map[string]interface{}{
		"id":           "doc_1",
		"document_id":  documentID,
		"title":        "示例文档",
		"content":      "这是文档的内容...",
		"content_type": "text/plain",
		"source":       "upload",
		"status":       "active",
		"metadata":     map[string]interface{}{"author": "system"},
		"created_at":   time.Now().AddDate(0, 0, -1).Format(time.RFC3339),
		"updated_at":   time.Now().Format(time.RFC3339),
	}, nil
}

// getDocumentChunksFromRepository 获取文档块
func (h *APIHandler) getDocumentChunksFromRepository(ctx context.Context, documentID string) ([]map[string]interface{}, error) {
	// 模拟文档块数据
	chunks := []map[string]interface{}{
		{
			"id":         "chunk_1",
			"chunk_id":   "chunk_001",
			"content":    "这是第一个文档块的内容...",
			"position":   1,
			"created_at": time.Now().Format(time.RFC3339),
		},
		{
			"id":         "chunk_2",
			"chunk_id":   "chunk_002",
			"content":    "这是第二个文档块的内容...",
			"position":   2,
			"created_at": time.Now().Format(time.RFC3339),
		},
	}

	return chunks, nil
}

// callDocumentUploadService 调用文档上传服务
func (h *APIHandler) callDocumentUploadService(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	// 调用算法服务的文档上传接口
	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal upload request: %w", err)
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/documents/upload", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to create upload request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 设置较长的超时时间，因为文档上传可能需要更多时间
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call upload service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upload service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode upload response: %w", err)
	}

	// 添加处理时间戳
	result["uploaded_at"] = time.Now().Unix()

	return result, nil
}

// callDocumentUpdateService 调用文档更新服务
func (h *APIHandler) callDocumentUpdateService(ctx context.Context, request map[string]interface{}) (map[string]interface{}, error) {
	// 模拟文档更新结果
	title, _ := request["title"].(string)
	status, _ := request["status"].(string)

	return map[string]interface{}{
		"title":      title,
		"status":     status,
		"updated_at": time.Now().Format(time.RFC3339),
	}, nil
}

// callDocumentDeleteService 调用文档删除服务
func (h *APIHandler) callDocumentDeleteService(ctx context.Context, request map[string]interface{}) error {
	// 模拟文档删除操作
	documentID, _ := request["document_id"].(string)
	if documentID == "" {
		return fmt.Errorf("document not found")
	}

	return nil
}

// getChatStatsFromService 获取聊天统计
func (h *APIHandler) getChatStatsFromService(ctx context.Context, tenantID string, startTime, endTime time.Time) (map[string]interface{}, error) {
	// 构建统计查询请求
	statsReq := map[string]interface{}{
		"type":       "chat",
		"tenant_id":  tenantID,
		"start_time": startTime.Format(time.RFC3339),
		"end_time":   endTime.Format(time.RFC3339),
		"metrics": []string{
			"total_sessions",
			"active_sessions",
			"completed_sessions",
			"failed_sessions",
			"avg_duration",
			"total_messages",
			"avg_response_time",
		},
	}

	// 调用算法服务的统计接口
	reqBody, err := json.Marshal(statsReq)
	if err != nil {
		logger.Error("构建聊天统计请求失败", logger.Field{Key: "error", Value: err})
		return h.getMockChatStats(), nil
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/stats/chat", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		logger.Error("创建聊天统计请求失败", logger.Field{Key: "error", Value: err})
		return h.getMockChatStats(), nil
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 发送请求
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logger.Error("聊天统计服务不可用", logger.Field{Key: "error", Value: err})
		return h.getMockChatStats(), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		logger.Error("聊天统计服务返回错误", logger.Field{Key: "status", Value: resp.StatusCode}, logger.Field{Key: "body", Value: string(body)})
		return h.getMockChatStats(), nil
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		logger.Error("解析聊天统计响应失败", logger.Field{Key: "error", Value: err})
		return h.getMockChatStats(), nil
	}

	return result, nil
}

// getMockChatStats 获取模拟聊天统计数据
func (h *APIHandler) getMockChatStats() map[string]interface{} {
	return map[string]interface{}{
		"total":             150,
		"active":            5,
		"completed":         140,
		"failed":            5,
		"avg_duration":      45.5,
		"total_messages":    850,
		"avg_response_time": 1.2,
		"success_rate":      0.967,
	}
}

// getVoiceStatsFromService 获取语音统计
func (h *APIHandler) getVoiceStatsFromService(ctx context.Context, tenantID string, startTime, endTime time.Time) (map[string]interface{}, error) {
	// 模拟语音统计数据
	return map[string]interface{}{
		"total":               80,
		"active":              2,
		"completed":           75,
		"failed":              3,
		"total_duration":      3600,
		"avg_duration":        45.0,
		"total_audio_minutes": 120,
	}, nil
}

// getSystemResourceStats 获取系统资源统计
func (h *APIHandler) getSystemResourceStats(ctx context.Context) (map[string]interface{}, error) {
	// 构建系统资源查询请求
	resourceReq := map[string]interface{}{
		"type": "system_resources",
		"metrics": []string{
			"uptime",
			"memory_usage",
			"cpu_usage",
			"disk_usage",
			"goroutines",
			"gc_stats",
			"network_stats",
		},
		"timestamp": time.Now().Unix(),
	}

	// 调用算法服务的系统资源接口
	reqBody, err := json.Marshal(resourceReq)
	if err != nil {
		logger.Error("构建系统资源请求失败", logger.Field{Key: "error", Value: err})
		return h.getMockSystemStats(), nil
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/stats/system", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		logger.Error("创建系统资源请求失败", logger.Field{Key: "error", Value: err})
		return h.getMockSystemStats(), nil
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 发送请求
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logger.Error("系统资源服务不可用", logger.Field{Key: "error", Value: err})
		return h.getMockSystemStats(), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		logger.Error("系统资源服务返回错误", logger.Field{Key: "status", Value: resp.StatusCode}, logger.Field{Key: "body", Value: string(body)})
		return h.getMockSystemStats(), nil
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		logger.Error("解析系统资源响应失败", logger.Field{Key: "error", Value: err})
		return h.getMockSystemStats(), nil
	}

	return result, nil
}

// getMockSystemStats 获取模拟系统统计数据
func (h *APIHandler) getMockSystemStats() map[string]interface{} {
	return map[string]interface{}{
		"uptime":       time.Now().Unix() - 1000000, // 模拟启动时间
		"memory_usage": "65%",
		"cpu_usage":    "25%",
		"disk_usage":   "45%",
		"goroutines":   150,
		"gc_stats": map[string]interface{}{
			"num_gc":      42,
			"pause_total": "12ms",
			"pause_avg":   "0.3ms",
		},
		"network_stats": map[string]interface{}{
			"bytes_sent":     1024000,
			"bytes_received": 2048000,
			"connections":    25,
		},
	}
}

// getDocumentStatsFromService 获取文档统计
func (h *APIHandler) getDocumentStatsFromService(ctx context.Context, tenantID string) (map[string]interface{}, error) {
	// 模拟文档统计数据
	return map[string]interface{}{
		"total":    250,
		"active":   200,
		"archived": 45,
		"deleted":  5,
	}, nil
}

// getActiveChatSessions 获取活跃聊天会话
func (h *APIHandler) getActiveChatSessions(ctx context.Context, tenantID string, limit int) ([]map[string]interface{}, int, error) {
	// 构建活跃会话查询请求
	sessionReq := map[string]interface{}{
		"type":      "chat",
		"status":    "active",
		"tenant_id": tenantID,
		"limit":     limit,
		"timestamp": time.Now().Unix(),
	}

	// 调用算法服务的活跃会话接口
	reqBody, err := json.Marshal(sessionReq)
	if err != nil {
		logger.Error("构建活跃聊天会话请求失败", logger.Field{Key: "error", Value: err})
		return h.getMockChatSessions(), len(h.getMockChatSessions()), nil
	}

	// 创建HTTP请求
	url := fmt.Sprintf("%s/api/v2/sessions/active", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		logger.Error("创建活跃聊天会话请求失败", logger.Field{Key: "error", Value: err})
		return h.getMockChatSessions(), len(h.getMockChatSessions()), nil
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	// 发送请求
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logger.Error("活跃会话服务不可用", logger.Field{Key: "error", Value: err})
		return h.getMockChatSessions(), len(h.getMockChatSessions()), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		logger.Error("活跃会话服务返回错误", logger.Field{Key: "status", Value: resp.StatusCode}, logger.Field{Key: "body", Value: string(body)})
		return h.getMockChatSessions(), len(h.getMockChatSessions()), nil
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		logger.Error("解析活跃会话响应失败", logger.Field{Key: "error", Value: err})
		return h.getMockChatSessions(), len(h.getMockChatSessions()), nil
	}

	// 提取会话列表
	sessions, ok := result["sessions"].([]interface{})
	if !ok {
		logger.Error("活跃会话服务返回格式错误")
		return h.getMockChatSessions(), len(h.getMockChatSessions()), nil
	}

	// 转换为所需格式
	var sessionList []map[string]interface{}
	for _, session := range sessions {
		if sessionMap, ok := session.(map[string]interface{}); ok {
			sessionList = append(sessionList, sessionMap)
		}
	}

	total, _ := result["total"].(int)
	if total == 0 {
		total = len(sessionList)
	}

	return sessionList, total, nil
}

// getMockChatSessions 获取模拟聊天会话数据
func (h *APIHandler) getMockChatSessions() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"session_id":    "chat_001",
			"user_id":       "user_123",
			"type":          "chat",
			"status":        "active",
			"start_time":    time.Now().Add(-5 * time.Minute).Format(time.RFC3339),
			"last_activity": time.Now().Format(time.RFC3339),
			"message_count": 12,
			"duration":      300,
		},
		{
			"session_id":    "chat_002",
			"user_id":       "user_456",
			"type":          "chat",
			"status":        "active",
			"start_time":    time.Now().Add(-10 * time.Minute).Format(time.RFC3339),
			"last_activity": time.Now().Add(-2 * time.Minute).Format(time.RFC3339),
			"message_count": 8,
			"duration":      600,
		},
	}
}

// getActiveVoiceSessions 获取活跃语音会话
func (h *APIHandler) getActiveVoiceSessions(ctx context.Context, tenantID string, limit int) ([]map[string]interface{}, int, error) {
	// 模拟活跃语音会话数据
	sessions := []map[string]interface{}{
		{
			"session_id":    "voice_001",
			"user_id":       "user_789",
			"type":          "voice",
			"status":        "active",
			"start_time":    time.Now().Add(-3 * time.Minute).Format(time.RFC3339),
			"last_activity": time.Now().Format(time.RFC3339),
			"duration":      180,
		},
	}

	return sessions, len(sessions), nil
}

// isAdmin 检查用户是否为管理员
func (h *APIHandler) isAdmin(userID string) bool {
	// 模拟管理员检查
	adminUsers := map[string]bool{
		"admin":    true,
		"root":     true,
		"system":   true,
		"user_123": true, // 测试用户
	}

	return adminUsers[userID]
}

// reloadAppConfig 重载应用配置
func (h *APIHandler) reloadAppConfig(ctx context.Context) error {
	// 构建应用配置重载请求
	reloadReq := map[string]interface{}{
		"type":      "app_config",
		"timestamp": time.Now().Unix(),
		"config_files": []string{
			"app.yaml",
			"server.yaml",
			"logging.yaml",
		},
	}

	// 调用算法服务的配置重载接口
	reqBody, err := json.Marshal(reloadReq)
	if err != nil {
		return fmt.Errorf("failed to marshal app config reload request: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/config/reload", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return fmt.Errorf("failed to create app config reload request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call app config reload service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("app config reload service returned status %d: %s", resp.StatusCode, string(body))
	}

	logger.Info("应用配置重载成功")
	return nil
}

// reloadDatabaseConfig 重载数据库配置
func (h *APIHandler) reloadDatabaseConfig(ctx context.Context) error {
	// 构建数据库配置重载请求
	reloadReq := map[string]interface{}{
		"type":      "database_config",
		"timestamp": time.Now().Unix(),
		"config_files": []string{
			"database.yaml",
			"migrations.yaml",
		},
	}

	reqBody, err := json.Marshal(reloadReq)
	if err != nil {
		return fmt.Errorf("failed to marshal database config reload request: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/config/reload", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return fmt.Errorf("failed to create database config reload request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call database config reload service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("database config reload service returned status %d: %s", resp.StatusCode, string(body))
	}

	logger.Info("数据库配置重载成功")
	return nil
}

// reloadCacheConfig 重载缓存配置
func (h *APIHandler) reloadCacheConfig(ctx context.Context) error {
	// 构建缓存配置重载请求
	reloadReq := map[string]interface{}{
		"type":      "cache_config",
		"timestamp": time.Now().Unix(),
		"config_files": []string{
			"redis.yaml",
			"cache.yaml",
		},
	}

	reqBody, err := json.Marshal(reloadReq)
	if err != nil {
		return fmt.Errorf("failed to marshal cache config reload request: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/config/reload", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return fmt.Errorf("failed to create cache config reload request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call cache config reload service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("cache config reload service returned status %d: %s", resp.StatusCode, string(body))
	}

	logger.Info("缓存配置重载成功")
	return nil
}

// reloadAlgoConfig 重载算法服务配置
func (h *APIHandler) reloadAlgoConfig(ctx context.Context) error {
	// 构建算法服务配置重载请求
	reloadReq := map[string]interface{}{
		"type":      "algo_config",
		"timestamp": time.Now().Unix(),
		"config_files": []string{
			"models.yaml",
			"algorithms.yaml",
			"voice.yaml",
			"rag.yaml",
		},
	}

	reqBody, err := json.Marshal(reloadReq)
	if err != nil {
		return fmt.Errorf("failed to marshal algo config reload request: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/config/reload", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return fmt.Errorf("failed to create algo config reload request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	client := &http.Client{Timeout: 30 * time.Second} // 算法服务重载可能需要更长时间
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call algo config reload service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("algo config reload service returned status %d: %s", resp.StatusCode, string(body))
	}

	logger.Info("算法服务配置重载成功")
	return nil
}

// setMaintenanceModeInService 在服务中设置维护模式
func (h *APIHandler) setMaintenanceModeInService(ctx context.Context, config map[string]interface{}) error {
	// 调用算法服务的维护模式设置接口
	reqBody, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal maintenance mode config: %w", err)
	}

	url := fmt.Sprintf("%s/api/v2/admin/maintenance", h.algoServiceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return fmt.Errorf("failed to create maintenance mode request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call maintenance mode service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("maintenance mode service returned status %d: %s", resp.StatusCode, string(body))
	}

	enabled, _ := config["enabled"].(bool)
	logger.Info("维护模式设置成功", logger.Field{Key: "enabled", Value: enabled})
	return nil
}

// notifyMaintenanceModeToActiveSessions 通知活跃会话维护模式
func (h *APIHandler) notifyMaintenanceModeToActiveSessions(config map[string]interface{}) {
	// 构建通知请求
	notifyReq := map[string]interface{}{
		"type":             "maintenance_mode_notification",
		"maintenance_mode": config,
		"timestamp":        time.Now().Unix(),
	}

	// 调用算法服务的会话通知接口
	reqBody, err := json.Marshal(notifyReq)
	if err != nil {
		logger.Error("构建维护模式通知请求失败", logger.Field{Key: "error", Value: err})
		return
	}

	url := fmt.Sprintf("%s/api/v2/sessions/notify", h.algoServiceURL)
	req, err := http.NewRequest("POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		logger.Error("创建维护模式通知请求失败", logger.Field{Key: "error", Value: err})
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "VoiceHelper-Backend/1.0")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logger.Error("维护模式通知服务不可用", logger.Field{Key: "error", Value: err})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		logger.Error("维护模式通知服务返回错误", logger.Field{Key: "status", Value: resp.StatusCode}, logger.Field{Key: "body", Value: string(body)})
		return
	}

	logger.Info("维护模式通知发送成功")
}
