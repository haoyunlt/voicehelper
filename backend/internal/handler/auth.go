package handler

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/sirupsen/logrus"
)

// WeChatLoginRequest 微信小程序登录请求
type WeChatLoginRequest struct {
	Code string `json:"code" binding:"required"`
}

// WeChatLoginResponse 微信小程序登录响应
type WeChatLoginResponse struct {
	Token    string `json:"token"`
	TenantID string `json:"tenant_id"`
	UserID   string `json:"user_id"`
}

// WeChatSessionResponse 微信code2session响应
type WeChatSessionResponse struct {
	OpenID     string `json:"openid"`
	SessionKey string `json:"session_key"`
	UnionID    string `json:"unionid"`
	ErrCode    int    `json:"errcode"`
	ErrMsg     string `json:"errmsg"`
}

// Claims JWT声明
type Claims struct {
	UserID   string `json:"user_id"`
	TenantID string `json:"tenant_id"`
	OpenID   string `json:"openid"`
	Channel  string `json:"channel"`
	jwt.RegisteredClaims
}

// WeChatMiniProgramLogin 微信小程序登录
func (h *Handlers) WeChatMiniProgramLogin(c *gin.Context) {
	var req WeChatLoginRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// 调用微信 code2session 接口
	session, err := h.weChatCode2Session(req.Code)
	if err != nil {
		logrus.WithError(err).Error("Failed to get WeChat session")
		c.JSON(http.StatusUnauthorized, gin.H{"error": "WeChat authentication failed"})
		return
	}

	// 根据 OpenID 获取或创建用户
	userID, tenantID, err := h.getOrCreateUser(session.OpenID, session.UnionID)
	if err != nil {
		logrus.WithError(err).Error("Failed to get or create user")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "User creation failed"})
		return
	}

	// 生成 JWT
	token, err := h.generateJWT(userID, tenantID, session.OpenID)
	if err != nil {
		logrus.WithError(err).Error("Failed to generate JWT")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Token generation failed"})
		return
	}

	// 返回登录结果
	c.JSON(http.StatusOK, WeChatLoginResponse{
		Token:    token,
		TenantID: tenantID,
		UserID:   userID,
	})
}

// weChatCode2Session 调用微信 code2session 接口
func (h *Handlers) weChatCode2Session(code string) (*WeChatSessionResponse, error) {
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

// getOrCreateUser 获取或创建用户
func (h *Handlers) getOrCreateUser(openID, unionID string) (string, string, error) {
	// 这里简化处理，实际应该查询数据库
	// 在生产环境中，应该：
	// 1. 先查询用户是否存在
	// 2. 如果不存在则创建新用户
	// 3. 返回用户ID和租户ID

	// 模拟用户ID和租户ID
	userID := "user_" + openID[:8]
	tenantID := "tenant_default"

	// TODO: 实际数据库操作
	// user, err := h.services.UserService.GetByOpenID(openID)
	// if err != nil {
	//     user, err = h.services.UserService.Create(openID, unionID)
	// }

	logrus.Infof("User logged in: %s (tenant: %s)", userID, tenantID)

	return userID, tenantID, nil
}

// generateJWT 生成JWT令牌
func (h *Handlers) generateJWT(userID, tenantID, openID string) (string, error) {
	secretKey := os.Getenv("JWT_SECRET")
	if secretKey == "" {
		secretKey = "default-secret-key-for-development"
	}

	claims := Claims{
		UserID:   userID,
		TenantID: tenantID,
		OpenID:   openID,
		Channel:  "wechat",
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "chatbot-api",
			Subject:   userID,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString([]byte(secretKey))
	if err != nil {
		return "", fmt.Errorf("failed to sign token: %w", err)
	}

	return tokenString, nil
}

// ValidateToken 验证JWT令牌（中间件使用）
func (h *Handlers) ValidateToken(tokenString string) (*Claims, error) {
	secretKey := os.Getenv("JWT_SECRET")
	if secretKey == "" {
		secretKey = "default-secret-key-for-development"
	}

	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(secretKey), nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}

	return nil, fmt.Errorf("invalid token")
}

// RefreshToken 刷新令牌
func (h *Handlers) RefreshToken(c *gin.Context) {
	// 从请求头获取当前令牌
	tokenString := c.GetHeader("Authorization")
	if tokenString == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "No token provided"})
		return
	}

	// 去掉 "Bearer " 前缀
	if len(tokenString) > 7 && tokenString[:7] == "Bearer " {
		tokenString = tokenString[7:]
	}

	// 验证当前令牌
	claims, err := h.ValidateToken(tokenString)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
		return
	}

	// 生成新令牌
	newToken, err := h.generateJWT(claims.UserID, claims.TenantID, claims.OpenID)
	if err != nil {
		logrus.WithError(err).Error("Failed to refresh token")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Token refresh failed"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"token": newToken,
	})
}

// Logout 登出（可选，主要用于清理服务端会话）
func (h *Handlers) Logout(c *gin.Context) {
	// 获取用户信息
	userID := c.GetString("user_id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "User not found"})
		return
	}

	// TODO: 清理服务端会话（如果有）
	// h.services.SessionService.ClearUserSession(userID)

	logrus.Infof("User logged out: %s", userID)

	c.JSON(http.StatusOK, gin.H{
		"message": "Logged out successfully",
	})
}
