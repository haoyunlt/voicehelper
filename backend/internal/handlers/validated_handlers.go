package handlers

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/voicehelper/backend/pkg/validation"
)

// ValidatedAPIHandler 带参数验证的API处理器
type ValidatedAPIHandler struct {
	*APIHandler
	validator *validation.ParameterValidator
}

// NewValidatedAPIHandler 创建带验证的API处理器
func NewValidatedAPIHandler(apiHandler *APIHandler) *ValidatedAPIHandler {
	return &ValidatedAPIHandler{
		APIHandler: apiHandler,
		validator:  validation.NewParameterValidator(),
	}
}

// 语音相关的验证处理器

// ValidatedTranscribeAudio 验证语音转文字请求
func (h *ValidatedAPIHandler) ValidatedTranscribeAudio(c *gin.Context) {
	var req validation.TranscribeRequest

	// 验证multipart表单数据
	if result := h.validator.ValidateMultipart(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 额外的业务逻辑验证
	if req.AudioFile != nil {
		// 检查文件大小（最大50MB）
		maxSize := int64(50 * 1024 * 1024)
		if req.AudioFile.Size > maxSize {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error": gin.H{
					"code":    "FILE_TOO_LARGE",
					"message": "Audio file size exceeds maximum allowed size (50MB)",
					"details": gin.H{
						"file_size": req.AudioFile.Size,
						"max_size":  maxSize,
					},
				},
			})
			return
		}

		// 检查文件类型
		contentType := req.AudioFile.Header.Get("Content-Type")
		validTypes := []string{
			"audio/wav", "audio/wave", "audio/x-wav",
			"audio/mpeg", "audio/mp3",
			"audio/webm", "audio/x-m4a",
		}

		isValidType := false
		for _, validType := range validTypes {
			if strings.Contains(contentType, validType) {
				isValidType = true
				break
			}
		}

		if !isValidType {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error": gin.H{
					"code":    "INVALID_FILE_TYPE",
					"message": "Unsupported audio file type",
					"details": gin.H{
						"content_type":    contentType,
						"supported_types": validTypes,
					},
				},
			})
			return
		}
	}

	// 验证通过，调用原始处理器
	h.transcribeAudio(c)
}

// ValidatedSynthesizeText 验证文字转语音请求
func (h *ValidatedAPIHandler) ValidatedSynthesizeText(c *gin.Context) {
	var req validation.SynthesizeRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 额外验证：检查文本长度和内容
	textLength := len([]rune(req.Text))
	if textLength > 1000 {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "TEXT_TOO_LONG",
				"message": "Text length exceeds maximum allowed length",
				"details": gin.H{
					"text_length": textLength,
					"max_length":  1000,
				},
			},
		})
		return
	}

	// 检查是否包含敏感内容（简单示例）
	sensitiveWords := []string{"敏感词1", "敏感词2"} // 实际应该从配置或数据库加载
	for _, word := range sensitiveWords {
		if strings.Contains(req.Text, word) {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error": gin.H{
					"code":    "SENSITIVE_CONTENT",
					"message": "Text contains sensitive content",
				},
			})
			return
		}
	}

	// 验证通过，调用原始处理器
	h.synthesizeText(c)
}

// 搜索相关的验证处理器

// ValidatedSearchDocuments 验证文档搜索请求
func (h *ValidatedAPIHandler) ValidatedSearchDocuments(c *gin.Context) {
	var req validation.SearchRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 额外验证：检查查询长度和复杂度
	queryLength := len([]rune(req.Query))
	if queryLength < 2 {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "QUERY_TOO_SHORT",
				"message": "Search query is too short (minimum 2 characters)",
				"details": gin.H{
					"query_length": queryLength,
					"min_length":   2,
				},
			},
		})
		return
	}

	// 验证分页参数
	if req.Limit <= 0 {
		req.Limit = 10 // 默认值
	}
	if req.Limit > 100 {
		req.Limit = 100 // 最大值
	}

	if req.Offset < 0 {
		req.Offset = 0
	}

	// 验证通过，调用原始处理器
	h.searchDocuments(c)
}

// ValidatedGetSearchSuggestions 验证搜索建议请求
func (h *ValidatedAPIHandler) ValidatedGetSearchSuggestions(c *gin.Context) {
	var req validation.SearchSuggestionsRequest

	// 验证查询参数
	if result := h.validator.ValidateQueryParams(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 设置默认值
	if req.Limit <= 0 {
		req.Limit = 5
	}
	if req.Limit > 20 {
		req.Limit = 20
	}

	// 验证通过，调用原始处理器
	h.getSearchSuggestions(c)
}

// 文档管理的验证处理器

// ValidatedUploadDocument 验证文档上传请求
func (h *ValidatedAPIHandler) ValidatedUploadDocument(c *gin.Context) {
	var req validation.UploadDocumentRequest

	// 验证multipart表单数据
	if result := h.validator.ValidateMultipart(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证文件
	if req.File != nil {
		// 检查文件大小（最大100MB）
		maxSize := int64(100 * 1024 * 1024)
		if req.File.Size > maxSize {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error": gin.H{
					"code":    "FILE_TOO_LARGE",
					"message": "Document file size exceeds maximum allowed size (100MB)",
					"details": gin.H{
						"file_size": req.File.Size,
						"max_size":  maxSize,
					},
				},
			})
			return
		}

		// 检查文件类型
		filename := req.File.Filename
		validExtensions := []string{".pdf", ".doc", ".docx", ".txt", ".md", ".html"}

		isValidExtension := false
		for _, ext := range validExtensions {
			if strings.HasSuffix(strings.ToLower(filename), ext) {
				isValidExtension = true
				break
			}
		}

		if !isValidExtension {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error": gin.H{
					"code":    "INVALID_FILE_TYPE",
					"message": "Unsupported document file type",
					"details": gin.H{
						"filename":             filename,
						"supported_extensions": validExtensions,
					},
				},
			})
			return
		}
	}

	// 验证标题唯一性（这里应该查询数据库）
	// TODO: 实现标题唯一性检查

	// 验证通过，调用原始处理器
	h.uploadDocument(c)
}

// ValidatedUpdateDocument 验证文档更新请求
func (h *ValidatedAPIHandler) ValidatedUpdateDocument(c *gin.Context) {
	var req validation.UpdateDocumentRequest

	// 验证路径参数
	if result := h.validator.ValidatePathParams(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证文档是否存在
	// TODO: 实现文档存在性检查

	// 验证通过，调用原始处理器
	h.updateDocument(c)
}

// ValidatedDeleteDocument 验证文档删除请求
func (h *ValidatedAPIHandler) ValidatedDeleteDocument(c *gin.Context) {
	var req validation.DeleteDocumentRequest

	// 验证路径参数
	if result := h.validator.ValidatePathParams(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证文档是否存在
	// TODO: 实现文档存在性检查

	// 检查用户是否有删除权限
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "UNAUTHORIZED",
				"message": "User authentication required",
			},
		})
		return
	}

	// 验证通过，调用原始处理器
	h.deleteDocument(c)
}

// ValidatedGetDocument 验证获取文档请求
func (h *ValidatedAPIHandler) ValidatedGetDocument(c *gin.Context) {
	var req validation.GetDocumentRequest

	// 验证路径参数
	if result := h.validator.ValidatePathParams(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证查询参数
	if result := h.validator.ValidateQueryParams(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证通过，调用原始处理器
	h.getDocument(c)
}

// ValidatedListDocuments 验证列出文档请求
func (h *ValidatedAPIHandler) ValidatedListDocuments(c *gin.Context) {
	var req validation.ListDocumentsRequest

	// 验证查询参数
	if result := h.validator.ValidateQueryParams(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 设置默认值和限制
	if req.Page <= 0 {
		req.Page = 1
	}
	if req.PageSize <= 0 {
		req.PageSize = 20
	}
	if req.PageSize > 100 {
		req.PageSize = 100
	}

	// 验证通过，调用原始处理器
	h.listDocuments(c)
}

// 会话管理的验证处理器

// ValidatedCreateConversation 验证创建会话请求
func (h *ValidatedAPIHandler) ValidatedCreateConversation(c *gin.Context) {
	var req validation.CreateConversationRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证会话配置
	if req.Config != nil {
		// 验证配置项的有效性
		if maxTokens, exists := req.Config["max_tokens"]; exists {
			if tokens, ok := maxTokens.(float64); ok {
				if tokens < 1 || tokens > 8192 {
					c.JSON(http.StatusBadRequest, gin.H{
						"success": false,
						"error": gin.H{
							"code":    "INVALID_CONFIG",
							"message": "max_tokens must be between 1 and 8192",
							"details": gin.H{
								"field": "config.max_tokens",
								"value": tokens,
							},
						},
					})
					return
				}
			}
		}
	}

	// 验证通过，调用原始处理器
	h.createConversation(c)
}

// 认证相关的验证处理器

// ValidatedWechatMiniProgramLogin 验证微信小程序登录请求
func (h *ValidatedAPIHandler) ValidatedWechatMiniProgramLogin(c *gin.Context) {
	var req validation.WechatMiniProgramLoginRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证微信code格式
	if len(req.Code) < 10 {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "INVALID_WECHAT_CODE",
				"message": "Invalid WeChat authorization code format",
			},
		})
		return
	}

	// 验证通过，调用原始处理器
	h.wechatMiniProgramLogin(c)
}

// ValidatedRefreshToken 验证刷新令牌请求
func (h *ValidatedAPIHandler) ValidatedRefreshToken(c *gin.Context) {
	var req validation.RefreshTokenRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证refresh token格式
	if len(req.RefreshToken) < 20 {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "INVALID_REFRESH_TOKEN",
				"message": "Invalid refresh token format",
			},
		})
		return
	}

	// 验证通过，调用原始处理器
	h.refreshToken(c)
}

// Agent相关的验证处理器

// ValidatedAgentStream 验证Agent流式请求
func (h *ValidatedAPIHandler) ValidatedAgentStream(c *gin.Context) {
	var req validation.AgentStreamRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证工具列表
	if len(req.Tools) > 10 {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "TOO_MANY_TOOLS",
				"message": "Too many tools specified (maximum 10)",
				"details": gin.H{
					"tool_count": len(req.Tools),
					"max_tools":  10,
				},
			},
		})
		return
	}

	// 验证通过，调用原始处理器
	h.agentStream(c)
}

// ValidatedExecuteAgentTool 验证执行Agent工具请求
func (h *ValidatedAPIHandler) ValidatedExecuteAgentTool(c *gin.Context) {
	var req validation.ExecuteAgentToolRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证工具名称
	validTools := []string{"web_search", "calculator", "document_search", "code_executor"}
	isValidTool := false
	for _, tool := range validTools {
		if req.ToolName == tool {
			isValidTool = true
			break
		}
	}

	if !isValidTool {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "INVALID_TOOL",
				"message": "Invalid tool name",
				"details": gin.H{
					"tool_name":   req.ToolName,
					"valid_tools": validTools,
				},
			},
		})
		return
	}

	// 验证工具参数
	if len(req.Parameters) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "MISSING_PARAMETERS",
				"message": "Tool parameters are required",
			},
		})
		return
	}

	// 验证通过，调用原始处理器
	h.executeAgentTool(c)
}

// 管理相关的验证处理器

// ValidatedSetMaintenanceMode 验证设置维护模式请求
func (h *ValidatedAPIHandler) ValidatedSetMaintenanceMode(c *gin.Context) {
	var req validation.SetMaintenanceModeRequest

	// 验证JSON请求体
	if result := h.validator.ValidateJSON(c, &req); !result.Success {
		validation.SendValidationError(c, result)
		return
	}

	// 验证结束时间
	if req.EndTime > 0 && req.EndTime <= c.GetInt64("request_time") {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "INVALID_END_TIME",
				"message": "End time must be in the future",
				"details": gin.H{
					"end_time":     req.EndTime,
					"current_time": c.GetInt64("request_time"),
				},
			},
		})
		return
	}

	// 验证通过，调用原始处理器
	h.setMaintenanceMode(c)
}

// 通用验证辅助方法

// validateUserPermission 验证用户权限
func (h *ValidatedAPIHandler) validateUserPermission(c *gin.Context, requiredPermission string) bool {
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{
			"success": false,
			"error": gin.H{
				"code":    "UNAUTHORIZED",
				"message": "User authentication required",
			},
		})
		return false
	}

	// TODO: 实现权限检查逻辑
	// 这里应该查询用户权限并验证

	return true
}

// validateRateLimit 验证请求频率限制
func (h *ValidatedAPIHandler) validateRateLimit(c *gin.Context, key string, limit int, window int) bool {
	// TODO: 实现频率限制检查
	// 这里应该使用Redis或内存缓存来跟踪请求频率

	return true
}

// sanitizeInput 清理输入数据
func (h *ValidatedAPIHandler) sanitizeInput(input string) string {
	// 移除潜在的恶意字符
	input = strings.ReplaceAll(input, "<script>", "")
	input = strings.ReplaceAll(input, "</script>", "")
	input = strings.ReplaceAll(input, "javascript:", "")
	input = strings.ReplaceAll(input, "vbscript:", "")

	return strings.TrimSpace(input)
}
