package middleware

import (
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"github.com/sirupsen/logrus"
)

// AuthMiddleware JWT认证中间件
type AuthMiddleware struct {
	secretKey      []byte
	skipPaths      []string
	tokenBlacklist map[string]time.Time // 简单的黑名单实现
}

// NewAuthMiddleware 创建认证中间件
func NewAuthMiddleware(secretKey string, skipPaths []string) *AuthMiddleware {
	return &AuthMiddleware{
		secretKey:      []byte(secretKey),
		skipPaths:      skipPaths,
		tokenBlacklist: make(map[string]time.Time),
	}
}

// Claims JWT声明
type Claims struct {
	UserID   string   `json:"user_id"`
	TenantID string   `json:"tenant_id"`
	OpenID   string   `json:"openid"`
	Channel  string   `json:"channel"`
	Role     string   `json:"role"`
	Scopes   []string `json:"scopes"`
	jwt.RegisteredClaims
}

// Handle JWT验证中间件处理函数
func (a *AuthMiddleware) Handle() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 检查是否需要跳过验证
		if a.shouldSkip(c.Request.URL.Path) {
			c.Next()
			return
		}

		// 获取Token
		tokenString := a.extractToken(c)
		if tokenString == "" {
			c.JSON(401, gin.H{"error": "No token provided"})
			c.Abort()
			return
		}

		// 检查黑名单
		if a.isBlacklisted(tokenString) {
			c.JSON(401, gin.H{"error": "Token has been revoked"})
			c.Abort()
			return
		}

		// 验证Token
		claims, err := a.validateToken(tokenString)
		if err != nil {
			c.JSON(401, gin.H{"error": "Invalid token: " + err.Error()})
			c.Abort()
			return
		}

		// 检查Token是否过期
		if claims.ExpiresAt != nil && claims.ExpiresAt.Before(time.Now()) {
			c.JSON(401, gin.H{"error": "Token has expired"})
			c.Abort()
			return
		}

		// 将用户信息存入上下文
		c.Set("user_id", claims.UserID)
		c.Set("tenant_id", claims.TenantID)
		c.Set("role", claims.Role)
		c.Set("scopes", claims.Scopes)
		c.Set("channel", claims.Channel)
		c.Set("token", tokenString)

		// 自动续期（如果Token快过期）
		if a.shouldRenew(claims) {
			newToken, err := a.renewToken(claims)
			if err == nil {
				c.Header("X-New-Token", newToken)
			}
		}

		c.Next()
	}
}

// extractToken 从请求中提取Token
func (a *AuthMiddleware) extractToken(c *gin.Context) string {
	// 从Header获取
	authHeader := c.GetHeader("Authorization")
	if authHeader != "" {
		parts := strings.SplitN(authHeader, " ", 2)
		if len(parts) == 2 && strings.ToLower(parts[0]) == "bearer" {
			return parts[1]
		}
	}

	// 从Query参数获取（用于WebSocket）
	if token := c.Query("token"); token != "" {
		return token
	}

	// 从Cookie获取
	if cookie, err := c.Cookie("token"); err == nil {
		return cookie
	}

	return ""
}

// validateToken 验证Token
func (a *AuthMiddleware) validateToken(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		return a.secretKey, nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}

	return nil, jwt.ErrSignatureInvalid
}

// shouldSkip 检查路径是否需要跳过验证
func (a *AuthMiddleware) shouldSkip(path string) bool {
	for _, skipPath := range a.skipPaths {
		if strings.HasPrefix(path, skipPath) {
			return true
		}
	}
	return false
}

// shouldRenew 检查是否需要续期
func (a *AuthMiddleware) shouldRenew(claims *Claims) bool {
	if claims.ExpiresAt == nil {
		return false
	}

	// 如果Token在10分钟内过期，则续期
	return time.Until(claims.ExpiresAt.Time) < 10*time.Minute
}

// renewToken 续期Token
func (a *AuthMiddleware) renewToken(oldClaims *Claims) (string, error) {
	// 创建新的Claims，延长过期时间
	newClaims := &Claims{
		UserID:   oldClaims.UserID,
		TenantID: oldClaims.TenantID,
		OpenID:   oldClaims.OpenID,
		Channel:  oldClaims.Channel,
		Role:     oldClaims.Role,
		Scopes:   oldClaims.Scopes,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(24 * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    oldClaims.Issuer,
			Subject:   oldClaims.Subject,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, newClaims)
	return token.SignedString(a.secretKey)
}

// RevokeToken 撤销Token（加入黑名单）
func (a *AuthMiddleware) RevokeToken(tokenString string) {
	a.tokenBlacklist[tokenString] = time.Now().Add(24 * time.Hour)

	// 清理过期的黑名单项
	a.cleanupBlacklist()
}

// isBlacklisted 检查Token是否在黑名单中
func (a *AuthMiddleware) isBlacklisted(tokenString string) bool {
	expiry, exists := a.tokenBlacklist[tokenString]
	if !exists {
		return false
	}

	// 如果黑名单项已过期，删除它
	if time.Now().After(expiry) {
		delete(a.tokenBlacklist, tokenString)
		return false
	}

	return true
}

// cleanupBlacklist 清理过期的黑名单项
func (a *AuthMiddleware) cleanupBlacklist() {
	now := time.Now()
	for token, expiry := range a.tokenBlacklist {
		if now.After(expiry) {
			delete(a.tokenBlacklist, token)
		}
	}
}

// ==================== RBAC权限中间件 ====================

// RBACMiddleware 基于角色的访问控制中间件
type RBACMiddleware struct {
	permissions map[string][]string // role -> permissions
}

// NewRBACMiddleware 创建RBAC中间件
func NewRBACMiddleware() *RBACMiddleware {
	return &RBACMiddleware{
		permissions: map[string][]string{
			"super_admin": {"*"}, // 所有权限
			"admin": {
				"conversation:*",
				"document:*",
				"user:read",
				"user:update",
				"analytics:*",
			},
			"user": {
				"conversation:create",
				"conversation:read",
				"conversation:update",
				"document:read",
				"analytics:read",
			},
		},
	}
}

// RequirePermission 需要特定权限
func (r *RBACMiddleware) RequirePermission(permission string) gin.HandlerFunc {
	return func(c *gin.Context) {
		role, exists := c.Get("role")
		if !exists {
			c.JSON(403, gin.H{"error": "No role found"})
			c.Abort()
			return
		}

		if !r.hasPermission(role.(string), permission) {
			logrus.WithFields(logrus.Fields{
				"user_id":    c.GetString("user_id"),
				"role":       role,
				"permission": permission,
			}).Warn("Permission denied")

			c.JSON(403, gin.H{"error": "Permission denied"})
			c.Abort()
			return
		}

		c.Next()
	}
}

// hasPermission 检查角色是否有权限
func (r *RBACMiddleware) hasPermission(role, permission string) bool {
	perms, exists := r.permissions[role]
	if !exists {
		return false
	}

	for _, perm := range perms {
		if perm == "*" || perm == permission {
			return true
		}

		// 支持通配符匹配
		if strings.HasSuffix(perm, ":*") {
			prefix := strings.TrimSuffix(perm, ":*")
			if strings.HasPrefix(permission, prefix+":") {
				return true
			}
		}
	}

	return false
}

// ==================== 租户隔离中间件 ====================

// TenantMiddleware 租户隔离中间件
type TenantMiddleware struct{}

// NewTenantMiddleware 创建租户中间件
func NewTenantMiddleware() *TenantMiddleware {
	return &TenantMiddleware{}
}

// Handle 租户验证处理
func (t *TenantMiddleware) Handle() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 从JWT中获取租户ID
		tenantIDFromToken := c.GetString("tenant_id")

		// 从Header中获取租户ID
		tenantIDFromHeader := c.GetHeader("X-Tenant-ID")

		// 验证租户ID一致性
		if tenantIDFromHeader != "" && tenantIDFromToken != "" {
			if tenantIDFromHeader != tenantIDFromToken {
				c.JSON(403, gin.H{"error": "Tenant ID mismatch"})
				c.Abort()
				return
			}
		}

		// 设置最终的租户ID
		finalTenantID := tenantIDFromToken
		if finalTenantID == "" {
			finalTenantID = tenantIDFromHeader
		}

		if finalTenantID == "" {
			c.JSON(400, gin.H{"error": "Tenant ID required"})
			c.Abort()
			return
		}

		c.Set("final_tenant_id", finalTenantID)
		c.Next()
	}
}

// ==================== API Key认证中间件 ====================

// APIKeyAuth API Key认证中间件
type APIKeyAuth struct {
	apiKeys map[string]string // apiKey -> tenantID
}

// NewAPIKeyAuth 创建API Key认证
func NewAPIKeyAuth() *APIKeyAuth {
	return &APIKeyAuth{
		apiKeys: make(map[string]string),
	}
}

// Handle API Key验证处理
func (a *APIKeyAuth) Handle() gin.HandlerFunc {
	return func(c *gin.Context) {
		apiKey := c.GetHeader("X-API-Key")
		if apiKey == "" {
			// 尝试JWT认证
			c.Next()
			return
		}

		tenantID, valid := a.validateAPIKey(apiKey)
		if !valid {
			c.JSON(401, gin.H{"error": "Invalid API key"})
			c.Abort()
			return
		}

		// 设置租户信息
		c.Set("tenant_id", tenantID)
		c.Set("auth_type", "api_key")
		c.Next()
	}
}

// validateAPIKey 验证API Key
func (a *APIKeyAuth) validateAPIKey(apiKey string) (string, bool) {
	// TODO: 从数据库或缓存中验证
	tenantID, exists := a.apiKeys[apiKey]
	return tenantID, exists
}

// RegisterAPIKey 注册API Key
func (a *APIKeyAuth) RegisterAPIKey(apiKey, tenantID string) {
	a.apiKeys[apiKey] = tenantID
}
