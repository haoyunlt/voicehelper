package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// Simplified zero trust middleware without external dependencies

// ZeroTrustMiddleware implements zero trust security middleware
type ZeroTrustMiddleware struct {
	logger *logrus.Logger
	config *ZeroTrustConfig
}

// ZeroTrustConfig contains configuration for zero trust middleware
type ZeroTrustConfig struct {
	EnableRiskAssessment bool     `json:"enable_risk_assessment"`
	EnableCompliance     bool     `json:"enable_compliance"`
	ExemptPaths          []string `json:"exempt_paths"`
	RequiredClaims       []string `json:"required_claims"`
	MaxRiskScore         float64  `json:"max_risk_score"`
	ChallengeEndpoint    string   `json:"challenge_endpoint"`
}

// AccessRequest represents a simplified access request
type AccessRequest struct {
	UserID    string                 `json:"user_id"`
	Resource  string                 `json:"resource"`
	Action    string                 `json:"action"`
	IPAddress string                 `json:"ip_address"`
	UserAgent string                 `json:"user_agent"`
	Claims    map[string]interface{} `json:"claims"`
	Timestamp time.Time              `json:"timestamp"`
}

// AccessDecision represents a simplified access decision
type AccessDecision struct {
	Allowed   bool      `json:"allowed"`
	Reason    string    `json:"reason"`
	Action    string    `json:"action"`
	Timestamp time.Time `json:"timestamp"`
}

// NewZeroTrustMiddleware creates a new zero trust middleware
func NewZeroTrustMiddleware() *ZeroTrustMiddleware {
	config := &ZeroTrustConfig{
		EnableRiskAssessment: true,
		EnableCompliance:     true,
		ExemptPaths: []string{
			"/health",
			"/metrics",
			"/api/v1/auth/login",
			"/api/v1/auth/wechat/login",
		},
		RequiredClaims:    []string{"user_id", "roles"},
		MaxRiskScore:      0.7,
		ChallengeEndpoint: "/api/v1/auth/challenge",
	}

	return &ZeroTrustMiddleware{
		logger: logrus.New(),
		config: config,
	}
}

// Middleware returns the Gin middleware function
func (ztm *ZeroTrustMiddleware) Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check if path is exempt
		if ztm.isExemptPath(c.Request.URL.Path) {
			c.Next()
			return
		}

		// Create access request
		accessRequest := ztm.createAccessRequest(c)

		// Evaluate access (simplified)
		decision := ztm.evaluateAccess(accessRequest)

		// Handle access decision
		switch decision.Action {
		case "allow":
			ztm.handleAllow(c, decision)
		case "deny":
			ztm.handleDeny(c, decision)
		case "challenge":
			ztm.handleChallenge(c, decision)
		default:
			ztm.handleDeny(c, decision)
		}

		if decision.Allowed {
			// Add security headers
			ztm.addSecurityHeaders(c)
			c.Next()
		}
	}
}

// createAccessRequest creates an access request from the Gin context
func (ztm *ZeroTrustMiddleware) createAccessRequest(c *gin.Context) *AccessRequest {
	userID := ztm.extractUserID(c)
	claims := ztm.extractClaims(c)

	return &AccessRequest{
		UserID:    userID,
		Resource:  c.Request.URL.Path,
		Action:    c.Request.Method,
		IPAddress: ztm.getClientIP(c),
		UserAgent: c.GetHeader("User-Agent"),
		Claims:    claims,
		Timestamp: time.Now(),
	}
}

// evaluateAccess performs simplified access evaluation
func (ztm *ZeroTrustMiddleware) evaluateAccess(request *AccessRequest) *AccessDecision {
	// Simplified access control logic
	if request.UserID == "" {
		return &AccessDecision{
			Allowed:   false,
			Reason:    "User not authenticated",
			Action:    "deny",
			Timestamp: time.Now(),
		}
	}

	// Check for required claims
	for _, claim := range ztm.config.RequiredClaims {
		if _, exists := request.Claims[claim]; !exists {
			return &AccessDecision{
				Allowed:   false,
				Reason:    fmt.Sprintf("Missing required claim: %s", claim),
				Action:    "deny",
				Timestamp: time.Now(),
			}
		}
	}

	// Check for high-risk indicators
	if ztm.isHighRiskRequest(request) {
		return &AccessDecision{
			Allowed:   false,
			Reason:    "High risk access attempt",
			Action:    "challenge",
			Timestamp: time.Now(),
		}
	}

	return &AccessDecision{
		Allowed:   true,
		Reason:    "Access granted",
		Action:    "allow",
		Timestamp: time.Now(),
	}
}

// isHighRiskRequest checks for high-risk indicators
func (ztm *ZeroTrustMiddleware) isHighRiskRequest(request *AccessRequest) bool {
	// Check for suspicious user agents
	suspiciousAgents := []string{"bot", "crawler", "spider", "scraper"}
	userAgent := strings.ToLower(request.UserAgent)
	for _, agent := range suspiciousAgents {
		if strings.Contains(userAgent, agent) {
			return true
		}
	}

	// Check for off-hours access
	hour := request.Timestamp.Hour()
	if hour < 6 || hour > 22 {
		return true
	}

	return false
}

// Handle different access decisions

func (ztm *ZeroTrustMiddleware) handleAllow(c *gin.Context, decision *AccessDecision) {
	ztm.logger.Debugf("Access allowed for user %s to %s",
		ztm.extractUserID(c), c.Request.URL.Path)
}

func (ztm *ZeroTrustMiddleware) handleDeny(c *gin.Context, decision *AccessDecision) {
	ztm.logger.Warnf("Access denied for user %s to %s: %s",
		ztm.extractUserID(c), c.Request.URL.Path, decision.Reason)

	c.JSON(http.StatusForbidden, gin.H{
		"error":  "Access denied",
		"reason": decision.Reason,
	})
	c.Abort()
}

func (ztm *ZeroTrustMiddleware) handleChallenge(c *gin.Context, decision *AccessDecision) {
	ztm.logger.Infof("Access challenge required for user %s to %s: %s",
		ztm.extractUserID(c), c.Request.URL.Path, decision.Reason)

	c.JSON(http.StatusUnauthorized, gin.H{
		"error":  "Additional verification required",
		"reason": decision.Reason,
		"challenge": map[string]interface{}{
			"type":     "mfa",
			"endpoint": ztm.config.ChallengeEndpoint,
		},
	})
	c.Abort()
}

// Helper methods

func (ztm *ZeroTrustMiddleware) isExemptPath(path string) bool {
	for _, exemptPath := range ztm.config.ExemptPaths {
		if strings.HasPrefix(path, exemptPath) {
			return true
		}
	}
	return false
}

func (ztm *ZeroTrustMiddleware) extractUserID(c *gin.Context) string {
	// Try to get user ID from JWT token
	if userID, exists := c.Get("user_id"); exists {
		if userIDStr, ok := userID.(string); ok {
			return userIDStr
		}
	}

	// Try to get from header
	if userID := c.GetHeader("X-User-ID"); userID != "" {
		return userID
	}

	return ""
}

func (ztm *ZeroTrustMiddleware) extractClaims(c *gin.Context) map[string]interface{} {
	claims := make(map[string]interface{})

	// Try to get claims from JWT token
	if claimsValue, exists := c.Get("claims"); exists {
		if claimsMap, ok := claimsValue.(map[string]interface{}); ok {
			return claimsMap
		}
	}

	// Extract from headers (simplified)
	if roles := c.GetHeader("X-User-Roles"); roles != "" {
		var roleList []string
		if err := json.Unmarshal([]byte(roles), &roleList); err == nil {
			claims["roles"] = roleList
		}
	}

	// Check for MFA
	if mfa := c.GetHeader("X-MFA-Verified"); mfa == "true" {
		claims["mfa"] = true
	}

	return claims
}

func (ztm *ZeroTrustMiddleware) getClientIP(c *gin.Context) string {
	// Check X-Forwarded-For header
	if xff := c.GetHeader("X-Forwarded-For"); xff != "" {
		ips := strings.Split(xff, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// Check X-Real-IP header
	if xri := c.GetHeader("X-Real-IP"); xri != "" {
		return xri
	}

	// Fall back to RemoteAddr
	return c.ClientIP()
}

func (ztm *ZeroTrustMiddleware) addSecurityHeaders(c *gin.Context) {
	// Add security headers
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("X-Frame-Options", "DENY")
	c.Header("X-XSS-Protection", "1; mode=block")
	c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
	c.Header("Referrer-Policy", "strict-origin-when-cross-origin")
}
