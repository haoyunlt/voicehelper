package middleware

import (
	"voicehelper/backend/pkg/security"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// ZeroTrustMiddleware implements zero trust security middleware
type ZeroTrustMiddleware struct {
	engine            *security.ZeroTrustEngine
	complianceManager *security.ComplianceManager
	logger            *logrus.Logger
	config            *ZeroTrustConfig
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
		engine:            security.NewZeroTrustEngine(),
		complianceManager: security.NewComplianceManager(),
		logger:            logrus.New(),
		config:            config,
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

		// Evaluate access using zero trust engine
		decision, err := ztm.engine.EvaluateAccess(c.Request.Context(), accessRequest)
		if err != nil {
			ztm.logger.Errorf("Failed to evaluate access: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Access evaluation failed",
			})
			c.Abort()
			return
		}

		// Handle access decision
		switch decision.Action {
		case security.ActionAllow:
			ztm.handleAllow(c, decision)
		case security.ActionDeny:
			ztm.handleDeny(c, decision)
		case security.ActionChallenge:
			ztm.handleChallenge(c, decision)
		case security.ActionLog:
			ztm.handleLog(c, decision)
		case security.ActionAlert:
			ztm.handleAlert(c, decision)
		default:
			ztm.handleDeny(c, decision)
		}

		// Check compliance if enabled
		if ztm.config.EnableCompliance && decision.Allowed {
			violations, err := ztm.complianceManager.CheckCompliance(accessRequest)
			if err != nil {
				ztm.logger.Errorf("Failed to check compliance: %v", err)
			} else if len(violations) > 0 {
				ztm.handleComplianceViolations(c, violations)
			}
		}

		if decision.Allowed {
			// Add security headers
			ztm.addSecurityHeaders(c, decision)
			c.Next()
		}
	}
}

// createAccessRequest creates an access request from the Gin context
func (ztm *ZeroTrustMiddleware) createAccessRequest(c *gin.Context) *security.AccessRequest {
	// Extract user information from JWT token or session
	userID := ztm.extractUserID(c)
	claims := ztm.extractClaims(c)

	// Extract location information (simplified)
	location := ztm.extractLocation(c)

	// Create request context
	requestContext := security.RequestContext{
		IPAddress: ztm.getClientIP(c),
		UserAgent: c.GetHeader("User-Agent"),
		DeviceID:  c.GetHeader("X-Device-ID"),
		Location:  location,
		SessionID: ztm.extractSessionID(c),
		RequestID: ztm.generateRequestID(),
		Headers:   ztm.extractHeaders(c),
	}

	return &security.AccessRequest{
		UserID:    userID,
		Resource:  c.Request.URL.Path,
		Action:    c.Request.Method,
		Context:   requestContext,
		Claims:    claims,
		Timestamp: time.Now(),
	}
}

// Handle different access decisions

func (ztm *ZeroTrustMiddleware) handleAllow(c *gin.Context, decision *security.AccessDecision) {
	ztm.logger.Debugf("Access allowed for user %s to %s",
		ztm.extractUserID(c), c.Request.URL.Path)

	// Set session token if provided
	if decision.SessionToken != "" {
		c.Header("X-Session-Token", decision.SessionToken)
	}
}

func (ztm *ZeroTrustMiddleware) handleDeny(c *gin.Context, decision *security.AccessDecision) {
	ztm.logger.Warnf("Access denied for user %s to %s: %s",
		ztm.extractUserID(c), c.Request.URL.Path, decision.Reason)

	c.JSON(http.StatusForbidden, gin.H{
		"error":     "Access denied",
		"reason":    decision.Reason,
		"policy_id": decision.PolicyID,
		"rule_id":   decision.RuleID,
	})
	c.Abort()
}

func (ztm *ZeroTrustMiddleware) handleChallenge(c *gin.Context, decision *security.AccessDecision) {
	ztm.logger.Infof("Access challenge required for user %s to %s: %s",
		ztm.extractUserID(c), c.Request.URL.Path, decision.Reason)

	c.JSON(http.StatusUnauthorized, gin.H{
		"error":  "Additional verification required",
		"reason": decision.Reason,
		"challenge": map[string]interface{}{
			"type":     "mfa",
			"endpoint": ztm.config.ChallengeEndpoint,
			"token":    decision.SessionToken,
		},
	})
	c.Abort()
}

func (ztm *ZeroTrustMiddleware) handleLog(c *gin.Context, decision *security.AccessDecision) {
	ztm.logger.Infof("Access logged for user %s to %s: %s",
		ztm.extractUserID(c), c.Request.URL.Path, decision.Reason)

	// Continue processing the request
}

func (ztm *ZeroTrustMiddleware) handleAlert(c *gin.Context, decision *security.AccessDecision) {
	ztm.logger.Warnf("Security alert for user %s to %s: %s",
		ztm.extractUserID(c), c.Request.URL.Path, decision.Reason)

	// Send alert to security team (implementation depends on alerting system)
	ztm.sendSecurityAlert(decision)

	// Continue processing the request
}

func (ztm *ZeroTrustMiddleware) handleComplianceViolations(c *gin.Context, violations []*security.ComplianceViolation) {
	ztm.logger.Warnf("Compliance violations detected: %d violations", len(violations))

	// For high severity violations, deny access
	for _, violation := range violations {
		if violation.Severity == security.ViolationSeverityCritical ||
			violation.Severity == security.ViolationSeverityHigh {
			c.JSON(http.StatusForbidden, gin.H{
				"error":     "Compliance violation",
				"violation": violation.Title,
				"standard":  violation.StandardID,
				"severity":  violation.Severity,
			})
			c.Abort()
			return
		}
	}

	// For lower severity violations, add warning headers
	c.Header("X-Compliance-Warning", "true")
	c.Header("X-Compliance-Violations", string(len(violations)))
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

	if permissions := c.GetHeader("X-User-Permissions"); permissions != "" {
		var permList []string
		if err := json.Unmarshal([]byte(permissions), &permList); err == nil {
			claims["permissions"] = permList
		}
	}

	// Check for MFA
	if mfa := c.GetHeader("X-MFA-Verified"); mfa == "true" {
		claims["mfa"] = true
	}

	// Check for consent
	if consent := c.GetHeader("X-User-Consent"); consent != "" {
		claims["consent"] = map[string]interface{}{
			"valid": consent == "true",
		}
	}

	return claims
}

func (ztm *ZeroTrustMiddleware) extractLocation(c *gin.Context) *security.Location {
	// Try to get location from headers (set by a geolocation service)
	country := c.GetHeader("X-Country")
	region := c.GetHeader("X-Region")
	city := c.GetHeader("X-City")

	if country == "" {
		return nil
	}

	location := &security.Location{
		Country: country,
		Region:  region,
		City:    city,
	}

	// Try to parse coordinates
	if lat := c.GetHeader("X-Latitude"); lat != "" {
		if lon := c.GetHeader("X-Longitude"); lon != "" {
			// Parse coordinates (simplified)
			location.Latitude = 0.0  // Would parse from lat string
			location.Longitude = 0.0 // Would parse from lon string
		}
	}

	return location
}

func (ztm *ZeroTrustMiddleware) extractSessionID(c *gin.Context) string {
	// Try to get session ID from various sources
	if sessionID := c.GetHeader("X-Session-ID"); sessionID != "" {
		return sessionID
	}

	if cookie, err := c.Cookie("session_id"); err == nil {
		return cookie
	}

	return ""
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

func (ztm *ZeroTrustMiddleware) extractHeaders(c *gin.Context) map[string]string {
	headers := make(map[string]string)

	// Extract relevant headers
	relevantHeaders := []string{
		"User-Agent",
		"Accept",
		"Accept-Language",
		"Accept-Encoding",
		"Referer",
		"X-Forwarded-Proto",
		"X-Device-ID",
		"X-App-Version",
	}

	for _, header := range relevantHeaders {
		if value := c.GetHeader(header); value != "" {
			headers[strings.ToLower(header)] = value
		}
	}

	return headers
}

func (ztm *ZeroTrustMiddleware) generateRequestID() string {
	return "req_" + time.Now().Format("20060102150405") + "_" +
		string(time.Now().UnixNano()%1000000)
}

func (ztm *ZeroTrustMiddleware) addSecurityHeaders(c *gin.Context, decision *security.AccessDecision) {
	// Add security headers
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("X-Frame-Options", "DENY")
	c.Header("X-XSS-Protection", "1; mode=block")
	c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
	c.Header("Referrer-Policy", "strict-origin-when-cross-origin")

	// Add zero trust specific headers
	c.Header("X-ZT-Policy-ID", decision.PolicyID)
	c.Header("X-ZT-Risk-Score", fmt.Sprintf("%.3f", decision.Confidence))
	c.Header("X-ZT-Decision-Time", decision.Timestamp.Format(time.RFC3339))
}

func (ztm *ZeroTrustMiddleware) sendSecurityAlert(decision *security.AccessDecision) {
	// Implementation would depend on the alerting system
	// Could send to Slack, email, PagerDuty, etc.
	ztm.logger.Warnf("Security alert: %s (Policy: %s, Rule: %s)",
		decision.Reason, decision.PolicyID, decision.RuleID)
}

// Policy management methods

// AddPolicy adds a new zero trust policy
func (ztm *ZeroTrustMiddleware) AddPolicy(policy *security.ZeroTrustPolicy) error {
	return ztm.engine.AddPolicy(policy)
}

// GetPolicies returns all zero trust policies
func (ztm *ZeroTrustMiddleware) GetPolicies() map[string]*security.ZeroTrustPolicy {
	return ztm.engine.GetPolicies()
}

// UpdatePolicy updates an existing policy
func (ztm *ZeroTrustMiddleware) UpdatePolicy(policyID string, updates map[string]interface{}) error {
	return ztm.engine.UpdatePolicy(policyID, updates)
}

// DeletePolicy removes a policy
func (ztm *ZeroTrustMiddleware) DeletePolicy(policyID string) error {
	return ztm.engine.DeletePolicy(policyID)
}

// GetComplianceStatus returns compliance status
func (ztm *ZeroTrustMiddleware) GetComplianceStatus() map[string]security.ComplianceStatus {
	return ztm.complianceManager.GetComplianceStatus()
}

// GetViolations returns current compliance violations
func (ztm *ZeroTrustMiddleware) GetViolations() []*security.ComplianceViolation {
	return ztm.complianceManager.GetViolations()
}

// ResolveViolation marks a violation as resolved
func (ztm *ZeroTrustMiddleware) ResolveViolation(violationID, resolution string) error {
	return ztm.complianceManager.ResolveViolation(violationID, resolution)
}
