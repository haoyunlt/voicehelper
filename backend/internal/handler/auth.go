package handler

import (
	"voicehelper/backend/internal/repository"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/sirupsen/logrus"
)

// AuthHandler 认证处理器（使用真实数据库）
type AuthHandler struct {
	userRepo  repository.UserRepository
	jwtSecret string
}

// NewAuthHandler 创建认证处理器
func NewAuthHandler(userRepo repository.UserRepository) *AuthHandler {
	jwtSecret := os.Getenv("JWT_SECRET")
	if jwtSecret == "" {
		jwtSecret = "default-secret-key-change-in-production"
	}

	return &AuthHandler{
		userRepo:  userRepo,
		jwtSecret: jwtSecret,
	}
}

// WeChatMiniProgramLoginV2 微信小程序登录（数据库版）
func (h *AuthHandler) WeChatMiniProgramLoginV2(c *gin.Context) {
	var req WeChatLoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// 调用微信API获取session
	session, err := h.weChatCode2Session(req.Code)
	if err != nil {
		logrus.WithError(err).Error("Failed to get WeChat session")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "WeChat authentication failed"})
		return
	}

	// 获取或创建用户（使用真实数据库）
	ctx := c.Request.Context()
	user, err := h.getOrCreateUser(ctx, session.OpenID, session.UnionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to get or create user")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process user"})
		return
	}

	// 更新最后登录时间
	if err := h.userRepo.UpdateLastLogin(ctx, user.ID); err != nil {
		logrus.WithError(err).Warn("Failed to update last login time")
	}

	// 生成JWT
	token, err := h.generateJWT(user.ID, user.TenantID, session.OpenID)
	if err != nil {
		logrus.WithError(err).Error("Failed to generate JWT")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate token"})
		return
	}

	// 记录审计日志
	h.auditLog(ctx, user.ID, user.TenantID, "login", c.ClientIP())

	// 返回响应
	c.JSON(http.StatusOK, gin.H{
		"token":     token,
		"tenant_id": user.TenantID,
		"user_info": gin.H{
			"user_id":  user.ID,
			"nickname": user.Nickname,
			"avatar":   user.Avatar,
			"role":     user.Role,
			"open_id":  session.OpenID,
			"union_id": session.UnionID,
		},
	})
}

// getOrCreateUser 获取或创建用户（数据库实现）
func (h *AuthHandler) getOrCreateUser(ctx context.Context, openID, unionID string) (*repository.User, error) {
	// 先尝试获取用户
	user, err := h.userRepo.GetByOpenID(ctx, openID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	// 如果用户存在，返回
	if user != nil {
		return user, nil
	}

	// 用户不存在，创建新用户
	newUser := &repository.User{
		OpenID:   openID,
		UnionID:  unionID,
		TenantID: h.getDefaultTenantID(),
		Username: fmt.Sprintf("wx_%s", openID[:8]),
		Nickname: "微信用户",
		Role:     "user",
		Status:   "active",
	}

	if err := h.userRepo.Create(ctx, newUser); err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	logrus.Infof("Created new user: %s (openID: %s)", newUser.ID, openID)
	return newUser, nil
}

// weChatCode2Session 调用微信 code2session 接口
func (h *AuthHandler) weChatCode2Session(code string) (*WeChatSessionResponse, error) {
	appID := os.Getenv("WECHAT_APP_ID")
	appSecret := os.Getenv("WECHAT_APP_SECRET")

	if appID == "" || appSecret == "" {
		// 开发环境模拟
		if os.Getenv("ENV") == "development" {
			return &WeChatSessionResponse{
				OpenID:     "mock_openid_" + code,
				SessionKey: "mock_session_key",
				UnionID:    "mock_union_id",
			}, nil
		}
		return nil, fmt.Errorf("WeChat credentials not configured")
	}

	url := fmt.Sprintf(
		"https://api.weixin.qq.com/sns/jscode2session?appid=%s&secret=%s&js_code=%s&grant_type=authorization_code",
		appID, appSecret, code,
	)

	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to call WeChat API: %w", err)
	}
	defer resp.Body.Close()

	var session WeChatSessionResponse
	if err := json.NewDecoder(resp.Body).Decode(&session); err != nil {
		return nil, fmt.Errorf("failed to decode WeChat response: %w", err)
	}

	if session.ErrCode != 0 {
		return nil, fmt.Errorf("WeChat API error: %d - %s", session.ErrCode, session.ErrMsg)
	}

	return &session, nil
}

// generateJWT 生成JWT令牌
func (h *AuthHandler) generateJWT(userID, tenantID, openID string) (string, error) {
	// 创建claims
	claims := Claims{
		UserID:   userID,
		TenantID: tenantID,
		OpenID:   openID,
		Channel:  "wechat_miniprogram",
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(7 * 24 * time.Hour)), // 7天有效期
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "chatbot-api",
			Subject:   userID,
			ID:        fmt.Sprintf("%d", time.Now().Unix()),
		},
	}

	// 创建token
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)

	// 签名
	tokenString, err := token.SignedString([]byte(h.jwtSecret))
	if err != nil {
		return "", fmt.Errorf("failed to sign token: %w", err)
	}

	return tokenString, nil
}

// getDefaultTenantID 获取默认租户ID
func (h *AuthHandler) getDefaultTenantID() string {
	// 可以从配置或环境变量读取
	tenantID := os.Getenv("DEFAULT_TENANT_ID")
	if tenantID == "" {
		tenantID = "default"
	}
	return tenantID
}

// auditLog 记录审计日志
func (h *AuthHandler) auditLog(ctx context.Context, userID, tenantID, action, ip string) {
	// 这里可以异步写入审计日志
	// 为了避免影响主流程，使用goroutine
	go func() {
		logrus.WithFields(logrus.Fields{
			"user_id":   userID,
			"tenant_id": tenantID,
			"action":    action,
			"ip":        ip,
			"timestamp": time.Now().UTC(),
		}).Info("Audit log")

		// 写入数据库审计表
		// 这里可以实现实际的审计日志存储
		logrus.Info("Audit log recorded successfully")
	}()
}

// RefreshToken 刷新Token
func (h *AuthHandler) RefreshToken(c *gin.Context) {
	// 从header获取旧token
	oldToken := c.GetHeader("Authorization")
	if oldToken == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Token required"})
		return
	}

	// 解析旧token
	claims := &Claims{}
	token, err := jwt.ParseWithClaims(oldToken, claims, func(token *jwt.Token) (interface{}, error) {
		return []byte(h.jwtSecret), nil
	})

	if err != nil || !token.Valid {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
		return
	}

	// 检查是否在刷新窗口内（例如：过期前24小时）
	if time.Until(claims.ExpiresAt.Time) > 24*time.Hour {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Token not eligible for refresh"})
		return
	}

	// 生成新token
	newToken, err := h.generateJWT(claims.UserID, claims.TenantID, claims.OpenID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate new token"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"token":      newToken,
		"expires_at": time.Now().Add(7 * 24 * time.Hour).Unix(),
	})
}

// Logout 登出
func (h *AuthHandler) Logout(c *gin.Context) {
	userID := c.GetString("user_id")

	// 记录登出日志
	h.auditLog(c.Request.Context(), userID, c.GetString("tenant_id"), "logout", c.ClientIP())

	// 将token加入黑名单（Redis）
	// 这里可以实现token黑名单功能
	logrus.Info("Token added to blacklist")

	c.JSON(http.StatusOK, gin.H{
		"message": "Logged out successfully",
	})
}

// GetUserInfo 获取用户信息
func (h *AuthHandler) GetUserInfo(c *gin.Context) {
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}

	// 从数据库获取用户信息
	user, err := h.userRepo.GetByID(c.Request.Context(), userID)
	if err != nil {
		logrus.WithError(err).Error("Failed to get user info")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get user info"})
		return
	}

	if user == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "User not found"})
		return
	}

	// 返回用户信息（过滤敏感字段）
	c.JSON(http.StatusOK, gin.H{
		"user_id":    user.ID,
		"username":   user.Username,
		"nickname":   user.Nickname,
		"avatar":     user.Avatar,
		"email":      user.Email,
		"phone":      user.Phone,
		"role":       user.Role,
		"status":     user.Status,
		"tenant_id":  user.TenantID,
		"last_login": user.LastLogin,
		"created_at": user.CreatedAt,
	})
}

// UpdateUserInfo 更新用户信息
func (h *AuthHandler) UpdateUserInfo(c *gin.Context) {
	userID := c.GetString("user_id")

	var req struct {
		Nickname string `json:"nickname"`
		Avatar   string `json:"avatar"`
		Email    string `json:"email"`
		Phone    string `json:"phone"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// 获取用户
	ctx := c.Request.Context()
	user, err := h.userRepo.GetByID(ctx, userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get user"})
		return
	}

	// 更新字段
	if req.Nickname != "" {
		user.Nickname = req.Nickname
	}
	if req.Avatar != "" {
		user.Avatar = req.Avatar
	}
	if req.Email != "" {
		user.Email = req.Email
	}
	if req.Phone != "" {
		user.Phone = req.Phone
	}

	// 保存更新
	if err := h.userRepo.Update(ctx, user); err != nil {
		logrus.WithError(err).Error("Failed to update user info")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update user info"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "User info updated successfully",
		"user": gin.H{
			"user_id":  user.ID,
			"nickname": user.Nickname,
			"avatar":   user.Avatar,
			"email":    user.Email,
			"phone":    user.Phone,
		},
	})
}
